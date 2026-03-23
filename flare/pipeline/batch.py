"""バッチ処理パイプライン。

仕様書7節「前処理（バッチ）モード設計」に基づき、データセット内の
全動画/フレームを一括処理し、抽出した特徴量をストレージに保存する。

出力構成:
    output_dir/
    ├── metadata.json
    ├── video_001/
    │   ├── params.npz     # shape: (T, param_dim)
    │   ├── landmarks.npy  # shape: (T, 68, 2)
    │   └── crops.npy      # shape: (T, H, W, 3) (オプション)
    ├── video_002/ ...
    ├── summary.csv
    └── checkpoint.json

Example:
    >>> from flare.config import load_config
    >>> from flare.pipeline.batch import BatchPipeline
    >>> config = load_config("config.yaml")
    >>> pipeline = BatchPipeline(config)
    >>> pipeline.run("/data/videos/", "/data/features/", resume=True)
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from flare.config import FLAREConfig
from flare.utils.errors import FaceNotDetectedError, PipelineError
from flare.utils.face_detect import FaceDetector
from flare.utils.metrics import FPSCounter, PipelineMetrics
from flare.utils.video import VideoReader

#: チェックポイント保存間隔（フレーム数）
_DEFAULT_CHECKPOINT_INTERVAL: int = 1000

#: サポートする動画拡張子
_VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


class BatchPipeline:
    """バッチ特徴量抽出パイプライン。

    データセット内の全動画からフレーム単位で3DMMパラメータを抽出し、
    npz/npy形式でストレージに保存する。チェックポイントによる
    中断・再開機能を備える。

    Attributes:
        _config: FLARE統合設定。
        _face_detector: 顔検出器。
        _metrics: パイプラインメトリクス。
        _fps_counter: FPSカウンタ。
        _checkpoint_interval: チェックポイント保存間隔。

    Example:
        >>> pipeline = BatchPipeline(config)
        >>> pipeline.run("/data/videos/", "/data/features/")
    """

    def __init__(self, config: FLAREConfig) -> None:
        """BatchPipelineを初期化する。

        Args:
            config: FLARE統合設定。
        """
        self._config: FLAREConfig = config
        self._face_detector: FaceDetector = FaceDetector(
            device="cpu", fallback_to_prev=True
        )
        self._metrics: PipelineMetrics = PipelineMetrics()
        self._fps_counter: FPSCounter = FPSCounter(window_size=60)
        self._checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL

    def run(
        self,
        input_dir: str,
        output_dir: str,
        *,
        resume: bool = True,
    ) -> None:
        """バッチ処理を実行する。

        input_dir内の全動画ファイルを処理し、抽出結果をoutput_dirに保存する。

        Args:
            input_dir: 入力動画ディレクトリパス。
            output_dir: 出力ディレクトリパス。
            resume: Trueの場合、既存チェックポイントから処理を再開する。

        Raises:
            PipelineError: 入力ディレクトリが存在しない場合。
        """
        input_path: Path = Path(input_dir)
        output_path: Path = Path(output_dir)

        if not input_path.exists():
            raise PipelineError(f"入力ディレクトリが存在しません: {input_dir}")

        output_path.mkdir(parents=True, exist_ok=True)

        # 動画ファイルの列挙
        video_files: List[Path] = sorted(
            p
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
        )

        if not video_files:
            logger.warning("動画ファイルが見つかりません: {}", input_dir)
            return

        logger.info(
            "バッチ処理開始: {} 動画 | input={} | output={}",
            len(video_files),
            input_dir,
            output_dir,
        )

        # メタデータ保存
        metadata: Dict[str, Any] = {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_config": self._config.pipeline.name,
            "extractor_type": self._config.extractor.type,
            "num_videos": len(video_files),
            "input_dir": str(input_path.resolve()),
        }
        self._write_json(output_path / "metadata.json", metadata)

        all_stats: List[Dict[str, Any]] = []

        for video_idx, video_path in enumerate(video_files):
            video_name: str = f"video_{video_idx:03d}"
            video_out_dir: Path = output_path / video_name
            video_out_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "処理中 [{}/{}]: {} → {}",
                video_idx + 1,
                len(video_files),
                video_path.name,
                video_name,
            )

            stats: Dict[str, Any] = self._process_video(
                video_path=video_path,
                output_dir=video_out_dir,
                video_name=video_name,
                resume=resume,
            )
            stats["source_file"] = video_path.name
            all_stats.append(stats)

        # サマリーCSV出力
        self._save_summary_csv(str(output_path), all_stats)

        logger.info(
            "バッチ処理完了: {} 動画処理済み | output={}",
            len(video_files),
            output_dir,
        )

    def _process_video(
        self,
        video_path: Path,
        output_dir: Path,
        video_name: str,
        resume: bool,
    ) -> Dict[str, Any]:
        """単一動画を処理する。

        Args:
            video_path: 入力動画ファイルパス。
            output_dir: 出力ディレクトリ。
            video_name: 動画の識別名。
            resume: チェックポイントからの再開を試みるか。

        Returns:
            処理統計情報のDict。
        """
        self._face_detector.reset_state()
        self._metrics.reset()
        self._fps_counter.reset()

        # チェックポイント確認
        start_frame: int = 0
        if resume:
            checkpoint: Optional[Dict[str, Any]] = self._load_checkpoint(
                str(output_dir)
            )
            if checkpoint is not None:
                start_frame = checkpoint.get("last_frame_index", -1) + 1
                logger.info(
                    "チェックポイントから再開: frame {} | {}",
                    start_frame,
                    video_name,
                )

        all_params: List[Dict[str, np.ndarray]] = []
        all_landmarks: List[Optional[np.ndarray]] = []
        all_crops: List[np.ndarray] = []
        errors: int = 0

        with VideoReader(str(video_path)) as reader:
            total_frames: int = reader.get_total_frames()
            fps: float = reader.get_fps()

            pbar = tqdm(
                enumerate(reader),
                total=total_frames if total_frames < 2**30 else None,
                desc=video_name,
                unit="frame",
            )

            for frame_idx, frame in pbar:
                if frame_idx < start_frame:
                    continue

                start_t: float = time.perf_counter()

                try:
                    # 顔検出・クロップ
                    bbox = self._face_detector.detect(frame)
                    cropped: np.ndarray = self._face_detector.crop_and_align(
                        frame, bbox, size=self._config.extractor.input_size
                    )

                    # ランドマーク検出
                    landmarks: Optional[np.ndarray] = (
                        self._face_detector.detect_landmarks(frame)
                    )

                    # パラメータ抽出（Phase 2で具象Extractorを接続）
                    # 現時点ではクロップ済み画像とランドマークを保存
                    params: Dict[str, np.ndarray] = {
                        "crop_shape": np.array(cropped.shape),
                    }

                    all_params.append(params)
                    all_landmarks.append(landmarks)
                    all_crops.append(cropped)

                except FaceNotDetectedError:
                    errors += 1
                    self._metrics.record_drop()
                    all_params.append({})
                    all_landmarks.append(None)
                    all_crops.append(
                        np.zeros(
                            (
                                self._config.extractor.input_size,
                                self._config.extractor.input_size,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                    )
                except Exception as exc:
                    errors += 1
                    self._metrics.record_drop()
                    logger.warning("フレーム {} 処理失敗: {}", frame_idx, exc)
                    all_params.append({})
                    all_landmarks.append(None)
                    all_crops.append(
                        np.zeros(
                            (
                                self._config.extractor.input_size,
                                self._config.extractor.input_size,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                    )

                elapsed_ms: float = (time.perf_counter() - start_t) * 1000.0
                self._metrics.record_frame(latency_ms=elapsed_ms)
                self._fps_counter.update()

                pbar.set_postfix(
                    fps=f"{self._fps_counter.get_fps():.1f}",
                    errors=errors,
                )

                # チェックポイント保存
                if (
                    self._config.checkpoint.enabled
                    and (frame_idx + 1) % self._checkpoint_interval == 0
                ):
                    self._save_checkpoint(
                        state={
                            "input_source": str(video_path),
                            "total_frames": total_frames,
                            "processed_frames": frame_idx + 1 - start_frame,
                            "last_frame_index": frame_idx,
                        },
                        output_dir=str(output_dir),
                    )

        # 結果保存
        self._save_results(output_dir, all_params, all_landmarks, all_crops)

        summary: Dict[str, float] = self._metrics.summary()
        stats: Dict[str, Any] = {
            "video_name": video_name,
            "total_frames": len(all_crops),
            "errors": errors,
            "avg_fps": summary.get("avg_latency_ms", 0.0),
            "drop_rate": summary.get("drop_rate", 0.0),
        }
        return stats

    def _save_results(
        self,
        output_dir: Path,
        all_params: List[Dict[str, np.ndarray]],
        all_landmarks: List[Optional[np.ndarray]],
        all_crops: List[np.ndarray],
    ) -> None:
        """抽出結果をファイルに保存する。

        Args:
            output_dir: 出力ディレクトリ。
            all_params: 全フレームのパラメータDictリスト。
            all_landmarks: 全フレームのランドマーク（Noneあり）。
            all_crops: 全フレームのクロップ済み画像。
        """
        # params.npz
        params_dict: Dict[str, np.ndarray] = {}
        if all_params and all_params[0]:
            for key in all_params[0]:
                arrays: List[np.ndarray] = [
                    p.get(key, np.array([])) for p in all_params
                ]
                valid: List[np.ndarray] = [a for a in arrays if a.size > 0]
                if valid:
                    params_dict[key] = np.stack(valid)

        if params_dict:
            np.savez_compressed(str(output_dir / "params.npz"), **params_dict)

        # landmarks.npy
        lm_list: List[np.ndarray] = []
        for lm in all_landmarks:
            if lm is not None:
                lm_list.append(lm)
            else:
                lm_list.append(np.zeros((68, 2), dtype=np.float32))

        if lm_list:
            np.save(str(output_dir / "landmarks.npy"), np.stack(lm_list))

        # crops.npy
        if all_crops:
            np.save(str(output_dir / "crops.npy"), np.stack(all_crops))

        logger.debug("結果保存完了: {}", output_dir)

    def _save_checkpoint(
        self, state: Dict[str, Any], output_dir: str
    ) -> None:
        """チェックポイントを保存する。

        仕様書7.3節のJSONスキーマに従いcheckpoint.jsonを保存する。

        Args:
            state: チェックポイント状態データ。
            output_dir: チェックポイント保存先ディレクトリ。
        """
        summary: Dict[str, float] = self._metrics.summary()

        checkpoint: Dict[str, Any] = {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_config": self._config.pipeline.name,
            "input_source": state.get("input_source", ""),
            "total_frames": state.get("total_frames", 0),
            "processed_frames": state.get("processed_frames", 0),
            "last_frame_index": state.get("last_frame_index", 0),
            "output_dir": output_dir,
            "extractor_state": state.get("extractor_state", {}),
            "metrics": {
                "avg_fps": summary.get("avg_latency_ms", 0.0),
                "dropped_frames": int(summary.get("dropped_frames", 0)),
                "errors": state.get("errors", 0),
            },
        }

        path: Path = Path(output_dir) / "checkpoint.json"
        self._write_json(path, checkpoint)
        logger.debug(
            "チェックポイント保存: frame {} | {}",
            state.get("last_frame_index", 0),
            path,
        )

    def _load_checkpoint(
        self, output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """チェックポイントを読み込む。

        checkpoint.jsonを読み込み、pipeline_configの一致を検証して返す。

        Args:
            output_dir: チェックポイントファイルのディレクトリ。

        Returns:
            チェックポイントデータ。ファイル未存在または設定不一致時はNone。
        """
        path: Path = Path(output_dir) / "checkpoint.json"

        if not path.exists():
            return None

        try:
            data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("チェックポイント読み込み失敗: {}", exc)
            return None

        # pipeline_configの一致検証
        saved_config: str = data.get("pipeline_config", "")
        current_config: str = self._config.pipeline.name

        if saved_config != current_config:
            logger.warning(
                "チェックポイントの設定が不一致: saved={} vs current={}",
                saved_config,
                current_config,
            )
            return None

        return data

    def _save_summary_csv(
        self, output_dir: str, stats: List[Dict[str, Any]]
    ) -> None:
        """全動画の統計情報をsummary.csvに書き出す。

        Args:
            output_dir: 出力ディレクトリ。
            stats: 各動画の統計情報リスト。
        """
        if not stats:
            return

        path: Path = Path(output_dir) / "summary.csv"
        fieldnames: List[str] = list(stats[0].keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer: csv.DictWriter = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats)

        logger.info("サマリーCSV保存: {}", path)

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        """JSONファイルを書き出す。

        Args:
            path: 出力ファイルパス。
            data: 書き出すデータ。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
