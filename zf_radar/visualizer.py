#!/usr/bin/env python3
"""Radar point cloud PCD visualizer with frame slider."""

import argparse
import glob
import os
import signal
import sys

import numpy as np

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QSurfaceFormat
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except ImportError as e:
    raise SystemExit(
        f"Missing GUI dependency: {e.name}. The visualizer needs the [viz] extras:\n"
        '  pip install "zf-frgen21-radar-dataset[viz]"\n'
        "or, from a checkout:  pip install -r requirements.txt"
    ) from e

from zf_radar.loaders import discover_sequences, load_extrinsics, load_pcd, load_poses


COLOR_LEVELS = {
    "snr": (10.0, 40.0),
    "rcs": (-45.0, 20.0),
}


def turbo_cmap():
    """Turbo-ish colormap: blue → cyan → green → yellow → red."""
    ns = np.linspace(0.0, 1.0, 256)
    rgba = np.ones((256, 4))
    rgba[:, 0] = np.clip(1.5 - np.abs(ns - 0.75) * 4, 0, 1)
    rgba[:, 1] = np.clip(1.5 - np.abs(ns - 0.5) * 4, 0, 1)
    rgba[:, 2] = np.clip(1.5 - np.abs(ns - 0.25) * 4, 0, 1)
    return pg.ColorMap(ns, (rgba * 255).astype(np.ubyte))


class RadarVisualizer(QMainWindow):
    def __init__(self, dataset_dir):
        super().__init__()
        self.setWindowTitle("Radar Point Cloud Visualizer")
        self.resize(1800, 800)

        self.dataset_dir = dataset_dir
        self.sequences = discover_sequences(dataset_dir)
        if not self.sequences:
            print(f"No sequences with a radar/ folder found in {dataset_dir}")
            sys.exit(1)

        self.files = []
        self.poses = None  # (N,4,4) or None
        self.T_sensor_vehicle = np.eye(4)  # sensor -> vehicle (4,4)

        self.current = 0
        self.playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        # Color-by field, fixed color scale from COLOR_LEVELS
        self.color_field = "snr"
        self.cmap = turbo_cmap()

        self._build_ui()
        self.load_sequence(self.sequences[0])

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top row: 3D view (left) + azimuth-doppler plot (right)
        plots_layout = QHBoxLayout()

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=80, elevation=30, azimuth=-90)
        plots_layout.addWidget(self.view, stretch=1)

        # Grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(400, 400)
        self.grid.setSpacing(10, 10)
        self.view.addItem(self.grid)

        # Scatter plot (radar points in global frame)
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=3, pxMode=True)
        self.view.addItem(self.scatter)

        # GPS trajectory + markers (per-sequence data set in load_sequence)
        self.traj_colors = np.zeros((2, 4))
        self.traj_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=self.traj_colors, width=2, antialias=True
        )
        self.view.addItem(self.traj_line)

        # Vehicle position
        self.pos_marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=(1, 0, 0, 1), size=10, pxMode=True
        )
        self.view.addItem(self.pos_marker)

        # Sensor position
        self.sensor_marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=(1, 1, 0, 1), size=8, pxMode=True
        )
        self.view.addItem(self.sensor_marker)

        # Ego vehicle wireframe
        self.vehicle_boxes = []
        for x0, x1, y0, y1, z0, z1 in [
            (-1.0, 3.95, -0.95, 0.95, -0.35, 1.2),
        ]:
            box = gl.GLBoxItem(color=(255, 255, 255, 150))
            box.setSize(x1 - x0, y1 - y0, z1 - z0)
            offset = np.eye(4)
            offset[:3, 3] = [x0, y0, z0]
            self.vehicle_boxes.append((box, offset))
            self.view.addItem(box)

        entries = [
            '<span style="color:#33cc33;">━</span>  Trajectory',
            '<span style="color:#ff3333;">●</span>  Rear Axle',
            '<span style="color:#ffff00;">●</span>  Sensor',
        ]
        view_legend = QLabel("<br>".join(entries), self.view)
        view_legend.setStyleSheet(
            "color: #dddddd; background: rgba(0, 0, 0, 130); padding: 6px;"
            "border: 1px solid rgba(150, 150, 150, 180); border-radius: 4px;"
        )
        view_legend.move(10, 10)
        view_legend.adjustSize()

        # Azimuth-Doppler 2D plot
        self.az_dop_widget = pg.PlotWidget(title="Azimuth vs Doppler")
        self.az_dop_widget.setLabel("bottom", "Azimuth", units="rad")
        self.az_dop_widget.setLabel("left", "Velocity", units="m/s")
        self.az_dop_widget.setXRange(-np.pi / 2, np.pi / 2)
        self.az_dop_widget.setYRange(-25, 25)
        self.az_dop_widget.setLimits(xMin=-np.pi / 2, xMax=np.pi / 2)
        # Driver's-eye view: negative azimuth (right in SCS) on the right
        self.az_dop_widget.getPlotItem().invertX(True)
        self.az_dop_widget.showGrid(x=True, y=True, alpha=0.3)
        self.az_dop_scatter = pg.ScatterPlotItem(size=4, pen=None)
        self.az_dop_widget.addItem(self.az_dop_scatter)

        # Legend
        legend = self.az_dop_widget.addLegend(
            offset=(-10, 10), pen=pg.mkPen(150, 150, 150, 180), brush=pg.mkBrush(0, 0, 0, 130)
        )
        legend.addItem(
            pg.PlotDataItem(pen=pg.mkPen("r")),
            "Velocity Ambiguity",
        )
        plots_layout.addWidget(self.az_dop_widget, stretch=1)

        # Colorbar
        cbar_widget = pg.GraphicsLayoutWidget()
        cbar_widget.setFixedWidth(110)
        self.cbar = pg.ColorBarItem(
            colorMap=self.cmap, interactive=False, width=20, label=self.color_field.upper()
        )
        cbar_widget.addItem(self.cbar)
        plots_layout.addWidget(cbar_widget)

        layout.addLayout(plots_layout, stretch=1)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Seq:"))
        self.seq_combo = QComboBox()
        self.seq_combo.addItems(self.sequences)
        self.seq_combo.currentTextChanged.connect(self.load_sequence)
        self.seq_combo.setFixedWidth(100)
        controls.addWidget(self.seq_combo)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedWidth(60)
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.load_frame)
        controls.addWidget(self.slider)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(120)
        controls.addWidget(self.frame_label)

        controls.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems([f.upper() for f in COLOR_LEVELS])
        self.color_combo.currentTextChanged.connect(self.on_color_changed)
        self.color_combo.setCurrentText(self.color_field.upper())
        self.color_combo.setFixedWidth(100)
        controls.addWidget(self.color_combo)

        self.info_label = QLabel("")
        self.info_label.setFixedWidth(80)
        controls.addWidget(self.info_label)

        layout.addLayout(controls)

    def load_sequence(self, name):
        seq_dir = os.path.join(self.dataset_dir, name)
        files = sorted(glob.glob(os.path.join(seq_dir, "radar", "frame_*.pcd")))
        if not files:
            print(f"No PCD files found in {os.path.join(seq_dir, 'radar')}")
            return
        if self.playing:
            self.toggle_play()

        self.files = files
        self.poses = load_poses(seq_dir)
        self.T_sensor_vehicle = load_extrinsics(seq_dir)
        if self.poses is not None:
            print(f"Loaded {len(self.poses)} poses")
            if len(self.poses) != len(self.files):
                print(f"Warning: {name}: {len(self.files)} frames but {len(self.poses)} poses; "
                      "frames beyond the pose count will render untransformed")
        self.setWindowTitle(f"Radar Point Cloud Visualizer — {name}")

        self.slider.blockSignals(True)
        self.slider.setMaximum(len(self.files) - 1)
        self.slider.blockSignals(False)

        # Pose-dependent scene items
        has_poses = self.poses is not None
        for item in [self.traj_line, self.pos_marker, self.sensor_marker] + [b for b, _ in self.vehicle_boxes]:
            item.setVisible(has_poses)
        if has_poses:
            traj = self.poses[:, :3, 3]
            # Per-vertex colors: traversed part solid, upcoming part faded
            self.traj_colors = np.tile([0.2, 0.8, 0.2, 0.2], (len(traj), 1))
            self.traj_line.setData(pos=traj, color=self.traj_colors)
            self.view.setCameraPosition(
                pos=pg.Vector(*traj[0]), distance=80, elevation=60, azimuth=-90
            )

        self.load_frame(0)

    def load_frame(self, idx):
        self.current = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

        data = load_pcd(self.files[idx])
        self.frame_data = data

        n = len(data.get("x", []))
        self.frame_label.setText(f"{idx} / {len(self.files) - 1}")
        self.info_label.setText(f"{n} pts")

        # Plot velocity ambiguity interval as horizontal lines (header meta is
        # present even for empty frames)
        v_amb = data.get("_meta", {}).get("velocity_ambiguity", 25.0)
        if hasattr(self, "_v_amb_lines"):
            for line in self._v_amb_lines:
                self.az_dop_widget.removeItem(line)
        self._v_amb_lines = [
            self.az_dop_widget.addLine(y=0.5 * v_amb, pen=pg.mkPen("r")),
            self.az_dop_widget.addLine(y=-0.5 * v_amb, pen=pg.mkPen("r")),
        ]

        # Pose-dependent scene updates (only need the pose, so they run even
        # for empty frames)
        T_sensor_global = None
        if self.poses is not None and idx < len(self.poses):
            T_sensor_global = self.poses[idx] @ self.T_sensor_vehicle  # vehicle_global @ vehicle_sensor

            # Update markers
            vehicle_pos = self.poses[idx, :3, 3]
            self.pos_marker.setData(pos=vehicle_pos.reshape(1, 3))
            self.sensor_marker.setData(pos=T_sensor_global[:3, 3].reshape(1, 3))

            # Update ego vehicle wireframe
            for box, offset in self.vehicle_boxes:
                box.setTransform(pg.Transform3D(self.poses[idx] @ offset))

            # Trajectory line
            colors = self.traj_colors.copy()
            colors[: idx + 1, 3] = 0.8
            self.traj_line.setData(color=colors)

            # Ego-centric grid
            self.grid.resetTransform()
            self.grid.translate(*(np.round(vehicle_pos / 10.0) * 10.0))

            opts = self.view.cameraParams()
            self.view.setCameraPosition(
                pos=pg.Vector(*vehicle_pos),
                distance=opts['distance'],
                elevation=opts['elevation'],
                azimuth=opts['azimuth'],
            )

        if n == 0:
            self.scatter.setData(pos=np.zeros((1, 3)), color=(1, 1, 1, 0))
            self.az_dop_scatter.setData([], [])
            return

        pos = np.column_stack([data["x"], data["y"], data["z"]])

        # Transform to global frame
        if T_sensor_global is not None:
            R_sg = T_sensor_global[:3, :3]
            t_sg = T_sensor_global[:3, 3]
            pos = (R_sg @ pos.T).T + t_sg

        norm = self._field_norm(data)
        if norm is None:
            colors, brushes = np.ones((n, 4)), None
        else:
            normed, vmin, vmax = norm
            colors = self.cmap.map(normed, mode="float")
            rgba = self.cmap.map(normed, mode="byte")
            rgba[:, 3] = 200
            brushes = [pg.mkBrush(*c) for c in rgba]
            self.cbar.setLevels((vmin, vmax))
        self.scatter.setData(pos=pos, color=colors, size=3)

        # Azimuth-Doppler 2D plot (skip if the PCD lacks either field)
        az_rad = data.get("azimuth", np.array([]))
        vel = data.get("velocity", np.array([]))
        if len(az_rad) == n and len(vel) == n:
            # brush=None would set NoBrush and render invisibly; fall back to
            # the 3D view's white when there is no color field
            brush = brushes if brushes is not None else pg.mkBrush(255, 255, 255, 200)
            self.az_dop_scatter.setData(x=az_rad, y=vel, brush=brush)
        else:
            self.az_dop_scatter.setData([], [])

    def _field_norm(self, data):
        """Normalize the active color field to [0,1] using COLOR_LEVELS.
        Returns (normed, vmin, vmax), or None if the field is missing/empty."""
        field = self.color_field
        if field not in data or len(data[field]) == 0 or field not in COLOR_LEVELS:
            return None
        vmin, vmax = COLOR_LEVELS[field]
        return np.clip((data[field] - vmin) / (vmax - vmin), 0, 1), vmin, vmax

    def on_color_changed(self, text):
        self.color_field = text.lower()
        self.cbar.getAxis("left").setLabel(text)
        self.load_frame(self.current)

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("Pause")
            self.play_timer.start(25)
        else:
            self.play_btn.setText("Play")
            self.play_timer.stop()

    def next_frame(self):
        nxt = self.current + 1
        if nxt >= len(self.files):
            nxt = 0
        self.load_frame(nxt)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_Right:
            self.load_frame(min(self.current + 1, len(self.files) - 1))
        elif key == Qt.Key.Key_Left:
            self.load_frame(max(self.current - 1, 0))
        else:
            super().keyPressEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Radar point cloud PCD visualizer")
    parser.add_argument(
        "-p", "--path",
        required=True,
        help="Dataset root containing sequence directories.",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.path):
        parser.error(f"dataset path is not a directory: {args.path}")

    fmt = QSurfaceFormat()
    fmt.setSamples(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    pg.setConfigOptions(antialias=True)

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    window = RadarVisualizer(args.path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()