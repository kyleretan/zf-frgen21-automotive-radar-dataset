#!/usr/bin/env python3
"""Capture visualizer frames as PNGs for the README GIF.

Run headless:
    xvfb-run -a -s "-screen 0 1800x850x24" python scripts/make_gif.py \
        -p /path/to/ZF_Dataset --seq zf_01 --start 200 --frames 120 --out /tmp/gif_frames

Then assemble (15 fps, 900 px wide):
    ffmpeg -framerate 15 -i /tmp/gif_frames/cap_%04d.png \
        -vf "scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" \
        docs/visualizer.gif
"""

import argparse
import os
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from zf_radar.visualizer import RadarVisualizer


def main():
    parser = argparse.ArgumentParser(description="Capture visualizer frames to PNGs")
    parser.add_argument("-p", "--path", required=True, help="Dataset root")
    parser.add_argument("--seq", help="Sequence name (default: first)")
    parser.add_argument("--start", type=int, default=0, help="First dataset frame")
    parser.add_argument("--frames", type=int, default=120, help="Number of captures")
    parser.add_argument("--stride", type=int, default=2, help="Dataset frames per capture")
    parser.add_argument("--out", default="gif_frames", help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    app = QApplication(sys.argv)
    window = RadarVisualizer(args.path)
    if args.seq:
        window.seq_combo.setCurrentText(args.seq)
    window.show()

    state = {"i": 0}

    def step():
        i = state["i"]
        if i >= args.frames:
            app.quit()
            return
        window.load_frame(args.start + i * args.stride)
        app.processEvents()
        img = app.primaryScreen().grabWindow(window.winId())
        img.save(os.path.join(args.out, f"cap_{i:04d}.png"))
        state["i"] += 1
        QTimer.singleShot(0, step)

    # Give the GL context a moment to initialize before the first grab
    QTimer.singleShot(500, step)
    app.exec()
    print(f"Wrote {state['i']} frames to {args.out}")


if __name__ == "__main__":
    main()
