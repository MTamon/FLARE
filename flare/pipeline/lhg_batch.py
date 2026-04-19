"""LHG 頭部特徴量抽出バッチパイプライン。

旧 MediaPipe + ``extract_angle_cent.py`` を FLARE ベースに置換した、
対面対話動画からの頭部姿勢・位置・表情係数抽出パイプライン。
DECA (FLAME 系) と Deep3DFaceRecon / 3DDFA (BFM 系) の両方をサポートし、
``databuild_nx8.py`` が参照する npz スキーマで出力する。

設計の全体像・数学的根拠・スキーマ定義は以下を参照:
    - ``docs/design/lhg_extract_pipeline.md``
    - ``docs/design/interpolation.md``
    - ``docs/design/rotation_interpolation.md``

入力構造::

    multimodal_dialogue_formed/
    └── dataXXX/
        ├── comp.mp4
        ├── comp.wav
        ├── host.mp4
        ├── host.wav
        └── participant.json

出力構造::

    movements/
    └── dataXXX/
        ├── comp/
        │   └── deca_comp_00000_17394.npz
        ├── host/
        │   └── deca_host_00000_17394.npz
        └── participant.json

Example:
    基本的な使用::

        from flare.config import PipelineConfig
        from flare.pipeline.lhg_batch import LHGBatchPipeline

        config = PipelineConfig.from_yaml("configs/lhg_extract_deca.yaml")
        pipeline = LHGBatchPipeline(config=config)
        pipeline.run(
            input_root="./data/multimodal_dialogue_formed",
            output_root="./data/movements",
            num_workers=1,
            redo=False,
        )
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
from loguru import logger

from flare.config import PipelineConfig
from flare.extractors.base import BaseExtractor
from flare.utils.face_detect import FaceDetector
from flare.utils.interp import (
    compute_stats,
    interp_linear,
    normalize,
    split_on_long_gaps,
)
from flare.utils.rotation_interp import interp_rotation

_ROLES: tuple[str, str] = ("comp", "host")
"""tuple[str, str]: 処理対象となる役割の一覧（comp/host の 2 者対話）。"""

_PARAM_VERSION: str = "flare-v2.2"
"""str: 出力 npz に埋め込むパラメータフォーマットバージョン。"""


@dataclass(frozen=True)
class _FeatureExtraction:
    """1 フレーム分の抽出済み特徴量のコンテナ。

    Attributes:
        angle: グローバル回転の軸角表現。形状 ``(3,)``。
        centroid: 頭部位置。形状 ``(3,)``。
        expression: 表情係数。形状 ``(D,)``（DECA: 50 / BFM: 64）。
        shape: 話者形状係数。形状 ``(S,)``。
        jaw_pose: 顎の軸角回転。形状 ``(3,)``。DECA / SMIRK のみ。
        face_size: 顔サイズ相当値（DECA では cam[0]）。スカラ。DECA / SMIRK のみ。
    """

    angle: np.ndarray
    centroid: np.ndarray
    expression: np.ndarray
    shape: np.ndarray
    jaw_pose: Optional[np.ndarray]
    face_size: Optional[np.ndarray]


def _infer_prefix(extractor_type: str, config_prefix: Optional[str]) -> str:
    """Extractor 種別からファイル名プレフィックスを決定する。

    Args:
        extractor_type: Extractor 種別文字列。
        config_prefix: YAML / config で指定された明示プレフィックス。
            ``None`` なら自動決定。

    Returns:
        プレフィックス文字列。

    Raises:
        ValueError: 未知の extractor_type が指定された場合。
    """
    if config_prefix is not None:
        return config_prefix
    mapping = {
        "deca": "deca",
        "deep3d": "bfm",
        "3ddfa": "bfm",
        "tdddfa": "bfm",
        "smirk": "smirk",
    }
    if extractor_type not in mapping:
        raise ValueError(
            f"Unknown extractor type for prefix inference: {extractor_type!r}"
        )
    return mapping[extractor_type]


def _extract_features_from_params(
    params: dict[str, Any],
    extractor_type: str,
) -> _FeatureExtraction:
    """Extractor の出力 dict から LHG 特徴量を抽出する。

    DECA / SMIRK は FLAME 系で ``pose[:, :3]`` = global rotation、
    ``pose[:, 3:6]`` = jaw pose、``cam`` = カメラパラメータ。
    Deep3DFaceRecon は BFM 系で ``pose[:, :3]`` = rotation、
    ``pose[:, 3:6]`` = translation（centroid に対応）。

    Args:
        params: Extractor の ``extract()`` が返す辞書。各値は torch.Tensor か
            numpy.ndarray を想定。バッチ次元は 1 を仮定。
        extractor_type: ``"deca"``, ``"deep3d"``, ``"smirk"``, ``"3ddfa"``,
            ``"tdddfa"`` のいずれか。

    Returns:
        ``_FeatureExtraction`` インスタンス。

    Raises:
        ValueError: サポート外の extractor_type または必要なキーが見つからない場合。
    """
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
        if arr.ndim >= 1 and arr.shape[0] == 1:
            return arr[0]
        return arr

    if extractor_type in ("deca", "smirk"):
        pose = _squeeze_batch(_to_numpy(params["pose"]))
        cam = _squeeze_batch(_to_numpy(params["cam"]))
        exp = _squeeze_batch(_to_numpy(params["exp"]))
        shape = _squeeze_batch(_to_numpy(params["shape"]))
        return _FeatureExtraction(
            angle=pose[:3].astype(np.float32),
            centroid=cam.astype(np.float32),
            expression=exp.astype(np.float32),
            shape=shape.astype(np.float32),
            jaw_pose=pose[3:6].astype(np.float32),
            face_size=np.asarray(cam[0], dtype=np.float32).reshape(()),
        )

    if extractor_type in ("deep3d", "3ddfa", "tdddfa"):
        if "pose" in params:
            pose = _squeeze_batch(_to_numpy(params["pose"]))
            angle = pose[:3].astype(np.float32)
            centroid = pose[3:6].astype(np.float32) if pose.shape[0] >= 6 else np.zeros(3, dtype=np.float32)
        else:
            angle = np.zeros(3, dtype=np.float32)
            centroid = np.zeros(3, dtype=np.float32)

        if "exp" in params:
            exp = _squeeze_batch(_to_numpy(params["exp"]))
        elif "expression" in params:
            exp = _squeeze_batch(_to_numpy(params["expression"]))
        else:
            raise ValueError(
                f"Extractor {extractor_type!r} output does not contain 'exp' key"
            )

        if "id" in params:
            shape = _squeeze_batch(_to_numpy(params["id"]))
        elif "shape" in params:
            shape = _squeeze_batch(_to_numpy(params["shape"]))
        else:
            raise ValueError(
                f"Extractor {extractor_type!r} output does not contain "
                f"'id' or 'shape' key"
            )

        return _FeatureExtraction(
            angle=angle,
            centroid=centroid,
            expression=exp.astype(np.float32),
            shape=shape.astype(np.float32),
            jaw_pose=None,
            face_size=None,
        )

    raise ValueError(f"Unsupported extractor type: {extractor_type!r}")


def _aggregate_shape(
    shapes: np.ndarray,
    mask: np.ndarray,
    method: str,
) -> np.ndarray:
    """シーケンス内の shape 係数を 1 ベクトルに集約する。

    Args:
        shapes: 形状 ``(T, S)`` の shape 配列。欠損位置の値は任意。
        mask: 有効フレームマスク ``(T,)``。
        method: 集約手法。``"median"``, ``"first"``, ``"mean"``。

    Returns:
        集約 shape ベクトル ``(S,)``。
    """
    valid = shapes[mask]
    if valid.shape[0] == 0:
        return np.zeros(shapes.shape[1], dtype=np.float32)
    if method == "median":
        return np.median(valid, axis=0).astype(np.float32)
    if method == "mean":
        return valid.mean(axis=0).astype(np.float32)
    if method == "first":
        return valid[0].astype(np.float32)
    raise ValueError(f"Unknown shape aggregation method: {method!r}")


class LHGBatchPipeline:
    """LHG 頭部特徴量抽出バッチパイプライン。

    ``multimodal_dialogue_formed/dataXXX/{comp,host}.mp4`` 形式のデータセットを
    走査し、各動画から per-frame に 3DMM パラメータを抽出、ギャップ補間・
    シーケンス分割・対話単位正規化を行って ``movements/dataXXX/`` 以下に
    npz として保存する。

    Attributes:
        _config: パイプライン全体設定。
        _extractor: 3DMM Extractor（注入または遅延構築）。
        _face_detector: 顔検出器（注入または遅延構築）。
        _extractor_type: Extractor 種別文字列（key mapping に使用）。
        _prefix: 出力ファイル名プレフィックス（``"deca"`` / ``"bfm"`` / ``"smirk"``）。
    """

    def __init__(
        self,
        config: PipelineConfig,
        extractor: Optional[BaseExtractor] = None,
        face_detector: Optional[FaceDetector] = None,
    ) -> None:
        """LHGBatchPipeline を初期化する。

        Args:
            config: ``PipelineConfig``。``extractor.type`` と ``lhg_extract``
                の各サブ設定を参照する。
            extractor: 使用する Extractor インスタンス。``None`` の場合は
                ``run()`` 実行時に ``config.extractor.type`` から構築を試みる。
                テスト時にはモックを注入できる。
            face_detector: 使用する ``FaceDetector``。``None`` の場合は
                ``run()`` 実行時に構築する。テスト時にはモックを注入できる。
        """
        self._config = config
        self._extractor = extractor
        self._face_detector = face_detector
        self._extractor_type = config.extractor.type.lower()
        self._prefix = _infer_prefix(
            self._extractor_type, config.lhg_extract.output.prefix
        )

    # -------------------------------------------------------------------------
    # 公開 API
    # -------------------------------------------------------------------------

    def run(
        self,
        input_root: Union[str, Path],
        output_root: Union[str, Path],
        num_workers: int = 1,
        redo: bool = False,
    ) -> dict[str, int]:
        """パイプラインを実行する。

        Args:
            input_root: ``multimodal_dialogue_formed`` ルートディレクトリ。
                直下に ``dataXXX/`` 形式のサブディレクトリを含む。
            output_root: ``movements`` 出力ルートディレクトリ。存在しなければ作成。
            num_workers: 並列ワーカ数。現状のテスト用実装は逐次実行 (num_workers=1)
                のみサポート。マルチプロセス版は将来拡張。
            redo: ``True`` の場合、既存出力を上書きする。

        Returns:
            処理統計の辞書:
                - ``"num_data_dirs"``: 処理した dataXXX ディレクトリ数
                - ``"num_sequences"``: 出力した npz ファイル数
                - ``"num_skipped"``: スキップした dataXXX / role 数

        Raises:
            FileNotFoundError: ``input_root`` が存在しない場合。
        """
        input_path = Path(input_root)
        output_path = Path(output_root)

        if not input_path.exists():
            raise FileNotFoundError(f"Input root not found: {input_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        data_dirs = self._enumerate_data_dirs(input_path)
        logger.info("Found {} dataXXX directories in {}", len(data_dirs), input_path)

        if num_workers != 1:
            logger.warning(
                "num_workers={} requested, but current implementation runs sequentially",
                num_workers,
            )

        stats = {"num_data_dirs": 0, "num_sequences": 0, "num_skipped": 0}
        total_start = time.perf_counter()

        for data_dir in data_dirs:
            sub_stats = self._process_data_dir(data_dir, output_path, redo=redo)
            stats["num_data_dirs"] += 1
            stats["num_sequences"] += sub_stats["num_sequences"]
            stats["num_skipped"] += sub_stats["num_skipped"]

        elapsed = time.perf_counter() - total_start
        logger.info(
            "LHG extract complete: {} dirs, {} sequences, {} skipped in {:.1f}s",
            stats["num_data_dirs"],
            stats["num_sequences"],
            stats["num_skipped"],
            elapsed,
        )
        return stats

    # -------------------------------------------------------------------------
    # データセット走査
    # -------------------------------------------------------------------------

    def _enumerate_data_dirs(self, input_root: Path) -> list[Path]:
        """``input_root`` 直下の ``dataXXX/`` サブディレクトリを列挙する。

        サブディレクトリ名が ``data`` で始まり、残りが数字のものを対象とする。

        Args:
            input_root: 走査するルートディレクトリ。

        Returns:
            ソート済みの ``dataXXX`` ディレクトリパスリスト。
        """
        result: list[Path] = []
        for child in sorted(input_root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("data") and name[4:].isdigit():
                result.append(child)
        return result

    def _process_data_dir(
        self,
        data_dir: Path,
        output_root: Path,
        redo: bool,
    ) -> dict[str, int]:
        """単一の ``dataXXX/`` を処理する。

        Args:
            data_dir: 入力 ``dataXXX`` ディレクトリ。
            output_root: 出力ルート。このディレクトリ配下に同名 ``dataXXX/`` を作成。
            redo: ``True`` なら既存出力を上書き。

        Returns:
            統計辞書:
                - ``"num_sequences"``: 出力した npz ファイル数
                - ``"num_skipped"``: スキップした role 数
        """
        stats = {"num_sequences": 0, "num_skipped": 0}
        participant_path = data_dir / "participant.json"
        if not participant_path.exists():
            logger.error(
                "participant.json not found in {}, skipping", data_dir.name
            )
            stats["num_skipped"] += len(_ROLES)
            return stats

        try:
            with participant_path.open("r", encoding="utf-8") as f:
                participant = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse participant.json in {}: {}", data_dir.name, e
            )
            stats["num_skipped"] += len(_ROLES)
            return stats

        output_data_dir = output_root / data_dir.name
        output_data_dir.mkdir(parents=True, exist_ok=True)

        for role in _ROLES:
            video_path = data_dir / f"{role}.mp4"
            role_output = output_data_dir / role

            if not video_path.exists():
                logger.warning(
                    "Video {} not found in {}, skipping role", video_path.name, data_dir.name
                )
                stats["num_skipped"] += 1
                continue

            if role_output.exists() and not redo:
                existing = list(role_output.glob(f"{self._prefix}_{role}_*.npz"))
                if len(existing) > 0:
                    logger.info(
                        "Skipping existing output: {}/{} ({} files)",
                        data_dir.name,
                        role,
                        len(existing),
                    )
                    stats["num_skipped"] += 1
                    continue

            role_output.mkdir(parents=True, exist_ok=True)

            try:
                frames, fps = self._extract_video(video_path)
            except Exception as e:
                logger.error(
                    "Extraction failed for {}/{}: {}", data_dir.name, role, e
                )
                stats["num_skipped"] += 1
                continue

            sequences = self._process_sequence(frames, fps)
            if len(sequences) == 0:
                logger.warning(
                    "No valid sequences extracted from {}/{}", data_dir.name, role
                )
                continue

            num_saved = self._save_sequences(
                sequences=sequences,
                output_dir=role_output,
                role=role,
                participant=participant,
                fps=fps,
            )
            stats["num_sequences"] += num_saved

        # participant.json をコピー
        try:
            shutil.copy2(participant_path, output_data_dir / "participant.json")
        except OSError as e:
            logger.warning(
                "Failed to copy participant.json for {}: {}", data_dir.name, e
            )

        return stats

    # -------------------------------------------------------------------------
    # 動画処理コア
    # -------------------------------------------------------------------------

    def _ensure_components(self) -> None:
        """Extractor と FaceDetector が未構築なら構築する。

        Raises:
            RuntimeError: Extractor が注入されておらず、かつ自動構築に失敗した場合。
        """
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        if self._extractor is None:
            raise RuntimeError(
                "Extractor instance must be injected before running the pipeline. "
                "LHGBatchPipeline does not auto-construct Extractors to avoid "
                "loading large model files when unnecessary."
            )

    def _extract_video(
        self, video_path: Path
    ) -> tuple[list[Optional[_FeatureExtraction]], float]:
        """動画ファイルから per-frame に特徴量を抽出する。

        Args:
            video_path: 入力動画ファイル。

        Returns:
            2 要素のタプル:
                - フレームごとの ``_FeatureExtraction`` または ``None`` のリスト
                - 動画の FPS
        """
        self._ensure_components()
        assert self._face_detector is not None  # noqa: S101
        assert self._extractor is not None  # noqa: S101

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(self._config.pipeline.fps)
        input_size = int(self._config.extractor.input_size)
        device = torch.device(self._config.device_map.extractor)

        frames: list[Optional[_FeatureExtraction]] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                feature = self._process_single_frame(frame, input_size, device)
                frames.append(feature)
        finally:
            cap.release()

        return frames, fps

    def _process_single_frame(
        self,
        frame: np.ndarray,
        input_size: int,
        device: torch.device,
    ) -> Optional[_FeatureExtraction]:
        """1 フレーム分の顔検出 + 抽出を実行する。

        Args:
            frame: BGR 画像 ``(H, W, 3)`` uint8。
            input_size: Extractor 入力サイズ。
            device: 推論デバイス。

        Returns:
            成功時は ``_FeatureExtraction``、失敗時（顔未検出や推論エラー）は ``None``。
        """
        assert self._face_detector is not None  # noqa: S101
        assert self._extractor is not None  # noqa: S101

        try:
            bbox = self._face_detector.detect(frame)
        except Exception as e:
            logger.debug("Face detection error: {}", e)
            return None
        if bbox is None:
            return None

        try:
            margin_scale = 1.25 if self._extractor_type in ("deca", "smirk") else 1.0
            cropped = self._face_detector.crop_and_align(
                frame, bbox, size=input_size, margin_scale=margin_scale
            )
        except Exception as e:
            logger.debug("crop_and_align error: {}", e)
            return None

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        try:
            params = self._extractor.extract(tensor)
        except Exception as e:
            logger.debug("Extractor inference error: {}", e)
            return None

        try:
            return _extract_features_from_params(params, self._extractor_type)
        except (KeyError, ValueError) as e:
            logger.debug("Feature extraction error: {}", e)
            return None

    # -------------------------------------------------------------------------
    # シーケンス加工
    # -------------------------------------------------------------------------

    def _process_sequence(
        self,
        frames: list[Optional[_FeatureExtraction]],
        fps: float,
    ) -> list[dict[str, np.ndarray]]:
        """生フレーム列をギャップ補間・分割・正規化済みシーケンスに変換する。

        Args:
            frames: ``_extract_video`` の出力リスト。
            fps: 動画 FPS。

        Returns:
            各要素が下流 npz として保存できる辞書のリスト。辞書キーは
            ``angle``, ``centroid``, ``expression``, ``jaw_pose``, ``face_size``,
            ``shape``, ``section``, ``*_mean``, ``*_std`` を含む。
        """
        total = len(frames)
        if total == 0:
            return []

        mask = np.array([f is not None for f in frames], dtype=bool)
        if not mask.any():
            return []

        # 特徴量を配列に展開（欠損位置はゼロで仮埋め、後段のマスクで除外）
        reference = next(f for f in frames if f is not None)
        has_jaw = reference.jaw_pose is not None
        has_face_size = reference.face_size is not None
        exp_dim = reference.expression.shape[0]
        shape_dim = reference.shape.shape[0]

        angles = np.zeros((total, 3), dtype=np.float32)
        centroids = np.zeros((total, 3), dtype=np.float32)
        expressions = np.zeros((total, exp_dim), dtype=np.float32)
        shapes = np.zeros((total, shape_dim), dtype=np.float32)
        jaw_poses = np.zeros((total, 3), dtype=np.float32) if has_jaw else None
        face_sizes = np.zeros((total,), dtype=np.float32) if has_face_size else None

        for i, feat in enumerate(frames):
            if feat is None:
                continue
            angles[i] = feat.angle
            centroids[i] = feat.centroid
            expressions[i] = feat.expression
            shapes[i] = feat.shape
            if jaw_poses is not None and feat.jaw_pose is not None:
                jaw_poses[i] = feat.jaw_pose
            if face_sizes is not None and feat.face_size is not None:
                face_sizes[i] = float(feat.face_size)

        lhg_cfg = self._config.lhg_extract
        max_gap = max(1, int(round(lhg_cfg.interpolation.max_gap_sec * fps)))
        lin_order = lhg_cfg.interpolation.linear_order
        rot_order = lhg_cfg.interpolation.rotation_order

        angles, mask_ang = interp_rotation(angles, mask, max_gap, method=rot_order)
        centroids, mask_cen = interp_linear(centroids, mask, max_gap, order=lin_order)
        expressions, mask_exp = interp_linear(expressions, mask, max_gap, order=lin_order)

        if jaw_poses is not None:
            jaw_poses, _ = interp_linear(jaw_poses, mask, max_gap, order=lin_order)
        if face_sizes is not None:
            face_sizes, _ = interp_linear(face_sizes, mask, max_gap, order=lin_order)

        valid_mask = mask_ang & mask_cen & mask_exp

        min_length = lhg_cfg.sequence.min_length
        runs = split_on_long_gaps(valid_mask, min_length=min_length)

        results: list[dict[str, np.ndarray]] = []
        for start, end in runs:
            stop = end + 1
            sub_angle = angles[start:stop]
            sub_centroid = centroids[start:stop]
            sub_expression = expressions[start:stop]

            a_mean, a_std = compute_stats(sub_angle)
            c_mean, c_std = compute_stats(sub_centroid)
            e_mean, e_std = compute_stats(sub_expression)

            normed_angle = normalize(sub_angle, a_mean, a_std)
            normed_centroid = normalize(sub_centroid, c_mean, c_std)
            normed_expression = normalize(sub_expression, e_mean, e_std)

            shape_rep = _aggregate_shape(
                shapes[start:stop],
                mask[start:stop],
                lhg_cfg.output.shape_aggregation,
            )

            entry: dict[str, np.ndarray] = {
                "section": np.array([start, end], dtype=np.int32),
                "angle": normed_angle,
                "centroid": normed_centroid,
                "expression": normed_expression,
                "angle_mean": a_mean,
                "angle_std": a_std,
                "centroid_mean": c_mean,
                "centroid_std": c_std,
                "expression_mean": e_mean,
                "expression_std": e_std,
                "shape": shape_rep,
            }

            if jaw_poses is not None:
                entry["jaw_pose"] = jaw_poses[start:stop].astype(np.float32)
            if face_sizes is not None:
                entry["face_size"] = face_sizes[start:stop].astype(np.float32)

            results.append(entry)

        return results

    # -------------------------------------------------------------------------
    # 出力
    # -------------------------------------------------------------------------

    def _save_sequences(
        self,
        sequences: list[dict[str, np.ndarray]],
        output_dir: Path,
        role: str,
        participant: dict[str, Any],
        fps: float,
    ) -> int:
        """シーケンスを npz として書き出す。

        Args:
            sequences: ``_process_sequence`` の出力。
            output_dir: 出力先ディレクトリ（``.../dataXXX/{role}/``）。
            role: ``"comp"`` または ``"host"``。
            participant: ``participant.json`` の内容。
            fps: 動画 FPS。

        Returns:
            保存に成功したファイル数。
        """
        speaker_id_key = f"{role}_no"
        if speaker_id_key not in participant:
            logger.error(
                "Missing key {!r} in participant.json, cannot assign speaker_id",
                speaker_id_key,
            )
            return 0

        try:
            speaker_id = int(participant[speaker_id_key])
        except (TypeError, ValueError) as e:
            logger.error("Invalid speaker_id value: {}", e)
            return 0

        speaker_name = str(participant.get(role, ""))

        num_saved = 0
        for seq in sequences:
            start_idx = int(seq["section"][0])
            end_idx = int(seq["section"][1])
            filename = f"{self._prefix}_{role}_{start_idx:05d}_{end_idx:05d}.npz"
            output_path = output_dir / filename

            npz_data: dict[str, Any] = {
                "section": seq["section"],
                "speaker_id": np.int64(speaker_id),
                "speaker_name": np.array(speaker_name),
                "fps": np.float32(fps),
                "angle": seq["angle"],
                "centroid": seq["centroid"],
                "expression": seq["expression"],
                "angle_mean": seq["angle_mean"],
                "angle_std": seq["angle_std"],
                "centroid_mean": seq["centroid_mean"],
                "centroid_std": seq["centroid_std"],
                "expression_mean": seq["expression_mean"],
                "expression_std": seq["expression_std"],
                "shape": seq["shape"],
                "extractor_type": np.array(self._extractor_type),
                "param_version": np.array(_PARAM_VERSION),
            }
            if "jaw_pose" in seq:
                npz_data["jaw_pose"] = seq["jaw_pose"]
            if "face_size" in seq:
                npz_data["face_size"] = seq["face_size"]

            np.savez(str(output_path), **npz_data)
            num_saved += 1

        return num_saved
