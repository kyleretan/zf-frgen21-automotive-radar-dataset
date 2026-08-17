"""Tests for the numpy-only dataset loaders. Fixtures are built in tmp_path."""

import numpy as np
import pytest

from zf_radar.loaders import discover_sequences, load_extrinsics, load_pcd, load_poses


PCD_HEADER = """\
# .PCD v0.7 - Point Cloud Data file format
# timestamp_ns 1624006954674470229
# velocity_ambiguity 32.499187
VERSION 0.7
FIELDS x y z snr
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA {fmt}
"""


def write_pcd(path, rows, fmt="ascii"):
    body = "\n".join(" ".join(str(v) for v in row) for row in rows)
    path.write_text(PCD_HEADER.format(n=len(rows), fmt=fmt) + body + ("\n" if body else ""))


class TestLoadPcd:
    def test_fields_and_meta(self, tmp_path):
        pcd = tmp_path / "frame_000000.pcd"
        write_pcd(pcd, [[1.0, 2.0, 3.0, 25.0], [4.0, 5.0, 6.0, 30.0]])
        data = load_pcd(str(pcd))
        np.testing.assert_array_equal(data["x"], [1.0, 4.0])
        np.testing.assert_array_equal(data["snr"], [25.0, 30.0])
        assert data["_meta"]["velocity_ambiguity"] == pytest.approx(32.499187)

    def test_int_meta_stays_exact(self, tmp_path):
        pcd = tmp_path / "f.pcd"
        write_pcd(pcd, [[1.0, 2.0, 3.0, 25.0]])
        meta = load_pcd(str(pcd))["_meta"]
        assert meta["timestamp_ns"] == 1624006954674470229
        assert isinstance(meta["timestamp_ns"], int)

    def test_single_point(self, tmp_path):
        pcd = tmp_path / "f.pcd"
        write_pcd(pcd, [[1.0, 2.0, 3.0, 25.0]])
        data = load_pcd(str(pcd))
        assert data["x"].shape == (1,)

    def test_empty_frame(self, tmp_path):
        pcd = tmp_path / "f.pcd"
        write_pcd(pcd, [])
        data = load_pcd(str(pcd))
        assert len(data["x"]) == 0
        assert data["_meta"]["velocity_ambiguity"] == pytest.approx(32.499187)

    def test_binary_pcd_rejected(self, tmp_path):
        pcd = tmp_path / "f.pcd"
        header = PCD_HEADER.format(n=1, fmt="binary").encode()
        pcd.write_bytes(header + b"\xd9\xfe\x00\x12" * 4)
        with pytest.raises(ValueError, match="DATA binary"):
            load_pcd(str(pcd))


class TestLoadExtrinsics:
    HEADER = "T00,T01,T02,T03,T10,T11,T12,T13,T20,T21,T22,T23\n"
    ROW = "1,0,0,3.925,0,1,0,0,0,0,1,0\n"

    def test_valid(self, tmp_path):
        seq = tmp_path / "zf_01"
        seq.mkdir()
        (seq / "extrinsics.csv").write_text(self.HEADER + self.ROW)
        T = load_extrinsics(str(seq))
        assert T.shape == (4, 4)
        np.testing.assert_array_equal(T[:3, 3], [3.925, 0, 0])
        np.testing.assert_array_equal(T[3], [0, 0, 0, 1])

    def test_falls_back_to_dataset_root(self, tmp_path):
        seq = tmp_path / "zf_01"
        seq.mkdir()
        (tmp_path / "extrinsics.csv").write_text(self.HEADER + self.ROW)
        T = load_extrinsics(str(seq))
        np.testing.assert_array_equal(T[:3, 3], [3.925, 0, 0])

    def test_missing_raises(self, tmp_path):
        seq = tmp_path / "zf_01"
        seq.mkdir()
        with pytest.raises(FileNotFoundError, match="required sensor-to-vehicle extrinsics"):
            load_extrinsics(str(seq))

    def test_wrong_size_raises(self, tmp_path):
        seq = tmp_path / "zf_01"
        seq.mkdir()
        (seq / "extrinsics.csv").write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="expected 12"):
            load_extrinsics(str(seq))


class TestLoadPoses:
    @staticmethod
    def row(t, tx):
        return f"{t},1,0,0,{tx},0,1,0,0,0,0,1,0,5,0,0,0,0,0.1\n"

    def test_valid(self, tmp_path):
        header = "t," + ",".join(f"c{i}" for i in range(18)) + "\n"
        (tmp_path / "ground_truth.csv").write_text(header + self.row(0.0, 0.0) + self.row(0.09, 0.45))
        poses = load_poses(str(tmp_path))
        assert poses.shape == (2, 4, 4)
        np.testing.assert_array_equal(poses[1, :3, 3], [0.45, 0, 0])
        np.testing.assert_array_equal(poses[0, 3], [0, 0, 0, 1])

    def test_single_row(self, tmp_path):
        header = "t," + ",".join(f"c{i}" for i in range(18)) + "\n"
        (tmp_path / "ground_truth.csv").write_text(header + self.row(0.0, 0.0))
        assert load_poses(str(tmp_path)).shape == (1, 4, 4)

    def test_missing_returns_none(self, tmp_path):
        assert load_poses(str(tmp_path)) is None

    def test_wrong_columns_raises(self, tmp_path):
        (tmp_path / "ground_truth.csv").write_text("a,b\n1,2\n3,4\n")
        with pytest.raises(ValueError, match="expected 19"):
            load_poses(str(tmp_path))


class TestDiscoverSequences:
    def test_natural_sort_and_filtering(self, tmp_path):
        for name in ["zf_10", "zf_2", "zf_1"]:
            (tmp_path / name / "radar").mkdir(parents=True)
        (tmp_path / "no_radar_here").mkdir()
        (tmp_path / "stray_file.txt").write_text("x")
        assert discover_sequences(str(tmp_path)) == ["zf_1", "zf_2", "zf_10"]