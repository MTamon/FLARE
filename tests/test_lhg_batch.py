"""LHG 頭部特徴量抽出バッチパイプライン ``flare.pipeline.lhg_batch`` のテスト。

テスト方針:
    - 実 DECA / BFM モデルは使わず、``MockExtractor`` でテンソル生成を代替
    - ``MockFaceDetector`` で顔検出を固定 bbox に置き換え、欠損パターンを注入可能
    - 合成動画 (OpenCV で書き出した小サイズ mp4) を入力に使用
    - npz 出力スキーマ・section 値・speaker_id 等の下流互換性を確認
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pytest
import torch

from flare.config import (
    DeviceMapConfig,
    ExtractorConfig,
    LHGExtractConfig,
    LHGInterpolationConfig,
    LHGSequenceConfig,
    PipelineConfig,
)
from flare.extractors.base import BaseExtractor
from flare.pipeline.lhg_batch import (
    LHGBatchPipeline,
    _aggregate_shape,
    _extract_features_from_params,
    _infer_prefix,
)


# =============================================================================
# モッククラス
# =============================================================================


class MockDECAExtractor(BaseExtractor):
    """DECA 互換の出力形式を返すモック Extractor。

    入力画像に依存せず、呼び出し回数に応じた決定論的な出力を生成する。
    これにより Extractor 本体を動かさずに下流パイプラインのテストができる。
    """

    def __init__(self) -> None:
        self._counter = 0

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = self._counter
        self._counter += 1
        base = float(idx) * 0.01
        return {
            "shape": torch.full((1, 100), base * 0.1),
            "tex": torch.zeros(1, 50),
            "exp": torch.full((1, 50), base),
            "pose": torch.tensor([[base, base * 0.5, 0.0, base * 0.2, 0.0, 0.0]]),
            "cam": torch.tensor([[1.0 + base, base * 0.1, base * 0.1]]),
            "light": torch.zeros(1, 27),
            "detail": torch.zeros(1, 128),
        }

    def extract_batch(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        results = [self.extract(img.unsqueeze(0)) for img in images]
        return {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}

    @property
    def param_dim(self) -> int:
        return 364

    @property
    def param_keys(self) -> list[str]:
        return ["shape", "tex", "exp", "pose", "cam", "light", "detail"]


class MockDeep3DExtractor(BaseExtractor):
    """Deep3DFaceRecon 互換の出力形式を返すモック Extractor。"""

    def __init__(self) -> None:
        self._counter = 0

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = self._counter
        self._counter += 1
        base = float(idx) * 0.01
        return {
            "id": torch.full((1, 80), base * 0.1),
            "exp": torch.full((1, 64), base),
            "tex": torch.zeros(1, 80),
            "pose": torch.tensor([[base, base * 0.5, 0.0, base * 10, base * 5, 0.0]]),
            "lighting": torch.zeros(1, 27),
        }

    def extract_batch(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        results = [self.extract(img.unsqueeze(0)) for img in images]
        return {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}

    @property
    def param_dim(self) -> int:
        return 257

    @property
    def param_keys(self) -> list[str]:
        return ["id", "exp", "tex", "pose", "lighting"]


class MockFaceDetector:
    """固定 bbox を返すモック FaceDetector。

    フレーム番号に応じて検出失敗を注入できる。
    """

    def __init__(self, fail_frames: Optional[set[int]] = None) -> None:
        self._fail_frames = fail_frames or set()
        self._counter = 0

    def detect(
        self, frame: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        idx = self._counter
        self._counter += 1
        if idx in self._fail_frames:
            return None
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = min(w, h) // 4
        return (cx - half, cy - half, cx + half, cy + half)

    def crop_and_align(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        size: int = 224,
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cropped = frame[max(0, y1) : y2, max(0, x1) : x2]
        if cropped.size == 0:
            cropped = np.zeros((size, size, 3), dtype=np.uint8)
        return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)

    def release(self) -> None:
        pass


# =============================================================================
# Fixture
# =============================================================================


def _make_synthetic_video(
    path: Path, num_frames: int, width: int = 128, height: int = 128, fps: int = 30
) -> None:
    """テスト用の合成動画を作成する。フレームごとに色が変わる単純な画像。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        pytest.skip(f"Failed to open VideoWriter for {path}")
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), (i * 3) % 255, dtype=np.uint8)
            cv2.circle(frame, (width // 2, height // 2), 20, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _write_participant_json(data_dir: Path, host_name: str, comp_name: str) -> None:
    participant = {
        "host": host_name,
        "comp": comp_name,
        "host_no": 1,
        "comp_no": 7,
    }
    with (data_dir / "participant.json").open("w", encoding="utf-8") as f:
        json.dump(participant, f, ensure_ascii=False)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """2 つの dataXXX ディレクトリを含む合成データセットを作成する。"""
    root = tmp_path / "multimodal_dialogue_formed"
    root.mkdir()

    for data_id in (1, 42):
        data_dir = root / f"data{data_id:03d}"
        data_dir.mkdir()
        _make_synthetic_video(data_dir / "comp.mp4", num_frames=150)
        _make_synthetic_video(data_dir / "host.mp4", num_frames=150)
        _write_participant_json(data_dir, host_name=f"alice{data_id}", comp_name="bob")

    return root


_CPU_DEVICE_MAP = DeviceMapConfig(extractor="cpu", lhg_model="cpu", renderer="cpu")


@pytest.fixture
def deca_config() -> PipelineConfig:
    return PipelineConfig(
        extractor=ExtractorConfig(type="deca", input_size=64),
        device_map=_CPU_DEVICE_MAP,
        lhg_extract=LHGExtractConfig(
            interpolation=LHGInterpolationConfig(max_gap_sec=0.2),
            sequence=LHGSequenceConfig(min_length=30),
        ),
    )


@pytest.fixture
def deep3d_config() -> PipelineConfig:
    return PipelineConfig(
        extractor=ExtractorConfig(type="deep3d", input_size=64),
        device_map=_CPU_DEVICE_MAP,
        lhg_extract=LHGExtractConfig(
            interpolation=LHGInterpolationConfig(max_gap_sec=0.2),
            sequence=LHGSequenceConfig(min_length=30),
        ),
    )


# =============================================================================
# ヘルパ関数のテスト
# =============================================================================


class TestInferPrefix:
    def test_deca(self) -> None:
        assert _infer_prefix("deca", None) == "deca"

    def test_deep3d_to_bfm(self) -> None:
        assert _infer_prefix("deep3d", None) == "bfm"

    def test_3ddfa_to_bfm(self) -> None:
        assert _infer_prefix("3ddfa", None) == "bfm"

    def test_smirk(self) -> None:
        assert _infer_prefix("smirk", None) == "smirk"

    def test_override(self) -> None:
        assert _infer_prefix("deca", "custom") == "custom"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown extractor"):
            _infer_prefix("mystery", None)


class TestExtractFeaturesFromParams:
    def test_deca_shapes(self) -> None:
        params = MockDECAExtractor().extract(torch.zeros(1, 3, 64, 64))
        feat = _extract_features_from_params(params, "deca")
        assert feat.angle.shape == (3,)
        assert feat.centroid.shape == (3,)
        assert feat.expression.shape == (50,)
        assert feat.shape.shape == (100,)
        assert feat.jaw_pose is not None and feat.jaw_pose.shape == (3,)
        assert feat.face_size is not None

    def test_deep3d_shapes(self) -> None:
        params = MockDeep3DExtractor().extract(torch.zeros(1, 3, 64, 64))
        feat = _extract_features_from_params(params, "deep3d")
        assert feat.angle.shape == (3,)
        assert feat.centroid.shape == (3,)
        assert feat.expression.shape == (64,)
        assert feat.shape.shape == (80,)
        assert feat.jaw_pose is None
        assert feat.face_size is None

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            _extract_features_from_params({}, "unknown_type")


class TestAggregateShape:
    def test_median(self) -> None:
        shapes = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float32)
        mask = np.array([True, True, True])
        result = _aggregate_shape(shapes, mask, "median")
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_first(self) -> None:
        shapes = np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32)
        mask = np.array([True, True])
        result = _aggregate_shape(shapes, mask, "first")
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_mean(self) -> None:
        shapes = np.array([[1.0], [3.0]], dtype=np.float32)
        mask = np.array([True, True])
        result = _aggregate_shape(shapes, mask, "mean")
        np.testing.assert_allclose(result, [2.0])

    def test_masked(self) -> None:
        shapes = np.array([[1.0], [999.0], [3.0]], dtype=np.float32)
        mask = np.array([True, False, True])
        result = _aggregate_shape(shapes, mask, "mean")
        np.testing.assert_allclose(result, [2.0])

    def test_all_invalid(self) -> None:
        shapes = np.zeros((3, 5), dtype=np.float32)
        mask = np.array([False, False, False])
        result = _aggregate_shape(shapes, mask, "median")
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_unknown_method_raises(self) -> None:
        shapes = np.zeros((2, 1), dtype=np.float32)
        mask = np.array([True, True])
        with pytest.raises(ValueError, match="Unknown shape"):
            _aggregate_shape(shapes, mask, "mode")


# =============================================================================
# パイプライン統合テスト
# =============================================================================


class TestLHGBatchPipelineDECA:
    def test_end_to_end(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )

        stats = pipeline.run(
            input_root=synthetic_dataset,
            output_root=output_root,
        )

        assert stats["num_data_dirs"] == 2
        assert stats["num_sequences"] >= 4  # 2 dirs × 2 roles 最低

        # 出力構造検証
        assert (output_root / "data001").is_dir()
        assert (output_root / "data042").is_dir()
        assert (output_root / "data001" / "participant.json").exists()
        assert (output_root / "data001" / "comp").is_dir()
        assert (output_root / "data001" / "host").is_dir()

    def test_file_naming_convention(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        comp_files = list((output_root / "data001" / "comp").glob("*.npz"))
        assert len(comp_files) >= 1
        name = comp_files[0].name
        assert name.startswith("deca_comp_")
        assert name.endswith(".npz")
        # SSSSS_EEEEE 形式の検証
        stem = name[len("deca_comp_") : -len(".npz")]
        start_s, end_s = stem.split("_")
        assert len(start_s) == 5 and len(end_s) == 5
        assert start_s.isdigit() and end_s.isdigit()

    def test_npz_schema(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        npz_files = list((output_root / "data001" / "host").glob("*.npz"))
        assert len(npz_files) >= 1
        data = np.load(npz_files[0], allow_pickle=False)

        # 下流必須キー
        assert "section" in data
        assert "speaker_id" in data
        assert "fps" in data
        # 特徴量
        assert "angle" in data and data["angle"].ndim == 2 and data["angle"].shape[1] == 3
        assert "centroid" in data and data["centroid"].shape[1] == 3
        assert "expression" in data and data["expression"].shape[1] == 50
        # 統計量
        assert "angle_mean" in data and data["angle_mean"].shape == (3,)
        assert "angle_std" in data
        assert "centroid_mean" in data
        assert "expression_mean" in data and data["expression_mean"].shape == (50,)
        # shape
        assert "shape" in data and data["shape"].shape == (100,)
        # メタ
        assert str(data["extractor_type"]) == "deca"
        assert str(data["param_version"]) == "flare-v2.2"
        # DECA 固有
        assert "jaw_pose" in data
        assert "face_size" in data

    def test_speaker_id_from_participant(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        host_files = list((output_root / "data001" / "host").glob("*.npz"))
        comp_files = list((output_root / "data001" / "comp").glob("*.npz"))
        host_data = np.load(host_files[0], allow_pickle=False)
        comp_data = np.load(comp_files[0], allow_pickle=False)
        assert int(host_data["speaker_id"]) == 1  # host_no
        assert int(comp_data["speaker_id"]) == 7  # comp_no

    def test_section_indices_valid(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        for npz_path in (output_root / "data001" / "comp").glob("*.npz"):
            data = np.load(npz_path, allow_pickle=False)
            section = data["section"]
            assert section.shape == (2,)
            assert section[0] <= section[1]
            # T = len(angle) should match end - start + 1
            assert data["angle"].shape[0] == int(section[1]) - int(section[0]) + 1

    def test_redo_overwrites(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)
        # 2 回目: redo=False なら全スキップ
        stats2 = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        ).run(
            input_root=synthetic_dataset,
            output_root=output_root,
            redo=False,
        )
        assert stats2["num_sequences"] == 0
        assert stats2["num_skipped"] >= 4

        # 3 回目: redo=True なら再生成
        stats3 = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        ).run(
            input_root=synthetic_dataset,
            output_root=output_root,
            redo=True,
        )
        assert stats3["num_sequences"] >= 4


class TestLHGBatchPipelineDeep3D:
    def test_end_to_end_bfm_prefix(
        self, synthetic_dataset: Path, tmp_path: Path, deep3d_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deep3d_config,
            extractor=MockDeep3DExtractor(),
            face_detector=MockFaceDetector(),
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        comp_files = list((output_root / "data001" / "comp").glob("*.npz"))
        assert len(comp_files) >= 1
        assert comp_files[0].name.startswith("bfm_comp_")

    def test_deep3d_schema_no_jaw(
        self, synthetic_dataset: Path, tmp_path: Path, deep3d_config: PipelineConfig
    ) -> None:
        output_root = tmp_path / "movements"
        pipeline = LHGBatchPipeline(
            config=deep3d_config,
            extractor=MockDeep3DExtractor(),
            face_detector=MockFaceDetector(),
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        host_files = list((output_root / "data001" / "host").glob("*.npz"))
        data = np.load(host_files[0], allow_pickle=False)

        assert "expression" in data and data["expression"].shape[1] == 64
        assert "shape" in data and data["shape"].shape == (80,)
        assert "jaw_pose" not in data.files
        assert "face_size" not in data.files


class TestLHGBatchPipelineGapHandling:
    def test_short_gap_interpolated(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        """短いギャップは補間され、シーケンスは分割されない。"""
        output_root = tmp_path / "movements"
        # 中央付近 3 フレームを欠損（30fps で max_gap_sec=0.2 → 6 frame まで補間可能）
        mock_detector = MockFaceDetector(fail_frames={50, 51, 52})
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=mock_detector,
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        comp_files = sorted((output_root / "data001" / "comp").glob("*.npz"))
        # 補間されたので分割されず 1 ファイル
        assert len(comp_files) == 1

    def test_long_gap_splits_sequence(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        """長いギャップではシーケンスが分割される。"""
        output_root = tmp_path / "movements"
        # 30fps × max_gap_sec=0.2 = 6frame 以下しか補間できない
        # 40 フレームの巨大ギャップ → 分割される
        mock_detector = MockFaceDetector(fail_frames=set(range(50, 90)))
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=mock_detector,
        )
        pipeline.run(input_root=synthetic_dataset, output_root=output_root)

        comp_files = sorted((output_root / "data001" / "comp").glob("*.npz"))
        assert len(comp_files) == 2
        # 前半と後半の section が不連続
        data0 = np.load(comp_files[0], allow_pickle=False)
        data1 = np.load(comp_files[1], allow_pickle=False)
        assert int(data0["section"][1]) < int(data1["section"][0])


class TestLHGBatchPipelineErrors:
    def test_missing_input_raises(
        self, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        with pytest.raises(FileNotFoundError):
            pipeline.run(
                input_root=tmp_path / "nonexistent",
                output_root=tmp_path / "out",
            )

    def test_missing_participant_json_skipped(
        self, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        root = tmp_path / "input"
        root.mkdir()
        data_dir = root / "data001"
        data_dir.mkdir()
        _make_synthetic_video(data_dir / "comp.mp4", num_frames=60)
        _make_synthetic_video(data_dir / "host.mp4", num_frames=60)
        # participant.json 不在

        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        stats = pipeline.run(input_root=root, output_root=tmp_path / "out")
        assert stats["num_sequences"] == 0
        assert stats["num_skipped"] >= 2

    def test_no_face_detected_empty_output(
        self,
        synthetic_dataset: Path,
        tmp_path: Path,
        deca_config: PipelineConfig,
    ) -> None:
        output_root = tmp_path / "movements"
        # 全フレームで検出失敗
        all_fail = set(range(10_000))
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(fail_frames=all_fail),
        )
        stats = pipeline.run(input_root=synthetic_dataset, output_root=output_root)
        assert stats["num_sequences"] == 0

    def test_extractor_not_injected_raises(
        self, synthetic_dataset: Path, tmp_path: Path, deca_config: PipelineConfig
    ) -> None:
        # extractor=None のまま run すると RuntimeError
        pipeline = LHGBatchPipeline(
            config=deca_config,
            extractor=None,
            face_detector=MockFaceDetector(),
        )
        # _ensure_components が RuntimeError を投げる（video 処理時）
        # run() は例外をキャッチして skip にするので統計で確認
        stats = pipeline.run(
            input_root=synthetic_dataset, output_root=tmp_path / "out"
        )
        assert stats["num_sequences"] == 0


class TestDataDirEnumeration:
    def test_only_matching_pattern(self, tmp_path: Path) -> None:
        root = tmp_path / "input"
        root.mkdir()
        (root / "data001").mkdir()
        (root / "data042").mkdir()
        (root / "readme.txt").touch()
        (root / "backup").mkdir()  # 非 data*
        (root / "data").mkdir()  # data だけ（数字無し）
        (root / "dataABC").mkdir()  # 数字でない

        config = PipelineConfig(
            extractor=ExtractorConfig(type="deca"),
            device_map=_CPU_DEVICE_MAP,
        )
        pipeline = LHGBatchPipeline(
            config=config,
            extractor=MockDECAExtractor(),
            face_detector=MockFaceDetector(),
        )
        dirs = pipeline._enumerate_data_dirs(root)
        names = [d.name for d in dirs]
        assert names == ["data001", "data042"]
