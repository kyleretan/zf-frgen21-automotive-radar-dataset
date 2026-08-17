"""Loader utilities for the ZF FRGen21 radar dataset.

Pure numpy/stdlib — no GUI dependencies, safe to import on headless machines.
"""

import os
import re

import numpy as np


def load_pcd(path: str) -> dict[str, np.ndarray]:
    """
    Load an ASCII PCD file to dictionary.

    Args:
        path: Path to the PCD file.

    Returns:
        dict: Dictionary containing the point cloud data.
    """
    fields = []
    num_points = 0
    header_lines = 0
    meta = {}
    data_format = None

    # errors="replace" keeps the header scan alive on binary payloads, which the
    # line iterator may read ahead into before we break at DATA
    with open(path, "r", errors="replace") as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("# ") and " " in stripped[2:]:
                parts = stripped[2:].split(None, 1)
                if len(parts) == 2:
                    # ints (e.g. timestamp_ns) stay exact — float64 would
                    # lose precision on nanosecond epochs
                    try:
                        meta[parts[0]] = int(parts[1])
                    except ValueError:
                        try:
                            meta[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
            elif stripped.startswith("FIELDS"):
                fields = stripped.split()[1:]
            elif stripped.startswith("POINTS"):
                num_points = int(stripped.split()[1])
            elif stripped.startswith("DATA"):
                parts = stripped.split()
                data_format = parts[1] if len(parts) > 1 else None
                break

    if data_format != "ascii":
        raise ValueError(f"{path}: only 'DATA ascii' PCD files are supported, got 'DATA {data_format}'")

    if num_points == 0:
        data = np.zeros((0, len(fields)))
    else:
        data = np.loadtxt(path, skiprows=header_lines, max_rows=num_points)
        if data.ndim == 1:
            data = data.reshape(1, -1)

    result = {"_meta": meta}
    for i, name in enumerate(fields):
        result[name] = data[:, i] if data.shape[0] > 0 else np.array([])
    return result


def load_extrinsics(seq_dir: str) -> np.ndarray:
    """
    Load the sensor-to-vehicle transform from ``extrinsics.csv``.

    Args:
        seq_dir: Sequence directory, for example ``ZF_Dataset/zf_01``.

    The loader first checks the sequence directory and then its parent dataset
    directory. The CSV must contain a header followed by one KITTI-style row
    with the 12 values ``T00`` through ``T23``.

    Returns:
        A 4x4 transform mapping sensor-frame points into the vehicle frame.

    Raises:
        FileNotFoundError: If ``extrinsics.csv`` is absent from both expected
            locations.
        ValueError: If the file does not contain exactly 12 numeric values.
    """
    seq_dir = os.path.abspath(os.fspath(seq_dir))
    candidates = [
        os.path.join(seq_dir, "extrinsics.csv"),
        os.path.join(os.path.dirname(seq_dir), "extrinsics.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            data = np.loadtxt(path, delimiter=",", skiprows=1).reshape(-1)
            if data.size != 12:
                raise ValueError(f"{path}: expected 12 values (T00..T23), got {data.size}")
            extrinsic = np.eye(4)
            extrinsic[:3, :4] = data.reshape(3, 4)
            return extrinsic

    searched = "\n  - ".join(candidates)
    raise FileNotFoundError(
        "Could not find the required sensor-to-vehicle extrinsics file. "
        f"Searched:\n  - {searched}"
    )


def load_poses(seq_dir):
    """
    Load ground_truth.csv if it exists. Returns (N,4,4) transforms or None.

    Expects KITTI format rows: t, T00..T23, vx, vy, vz, wx, wy, wz.
    """
    csv_path = os.path.join(seq_dir, "ground_truth.csv")
    if not os.path.exists(csv_path):
        return None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 19:
        raise ValueError(
            f"{csv_path}: expected 19 columns (t, T00..T23, vx..wz), got {data.shape[1]}"
        )

    pose = np.zeros((len(data), 4, 4))
    pose[:, :3, :4] = data[:, 1:13].reshape(-1, 3, 4)
    pose[:, 3, 3] = 1.0
    return pose


def discover_sequences(dataset_dir):
    """Sequence subdirs of dataset_dir that contain a radar/ folder, naturally sorted."""
    seqs = [n for n in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, n, "radar"))]
    return sorted(seqs, key=lambda s: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)])
