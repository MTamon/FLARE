"""バッチ処理パイプラインモジュール。

データセット内の全動画/フレームを一括処理し、抽出した特徴量を
``.npy`` / ``.npz`` 形式でストレージに保存する。

仕様書§7の設計に基づき、以下の機能を提供する:
    - 動画ファイル群からの3DMMパラメータ一括抽出
    - metadata.json / params.npz / landmarks.npy / crops.npy 形式での保存
    - JSONチェックポイントによる中断・再開機能
    - PipelineErrorHandler によるエラー処理

出力ディレクトリ構造::

    output_dir/
    ├── metadata.json
    ├── video_001/
    │   ├── params.npz       # shape: (T, param_dim)
    │   ├── landmarks.npy    # shape: (T, 68, 2)
    │   └── crops.npy        # shape: (T, H, W, 3)
    ├── video_002/
    │   └── ...
    └── summary.csv

Example:
    バッチ処理の実行::

        from flare.config import PipelineConfig
        from flare.pipeline.batch import BatchPipeline

        config = PipelineConfig.from_yaml("config.yaml")
        pipeline = BatchPipeline()
        pipeline.run(
            input_dir="/data/videos/",
            output_dir="/data/features/",
            config=config,
        )
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from loguru import logger

from flare.config import PipelineConfig
from flare.utils.errors import (
    ErrorPolicy,
    FaceNotDetectedError,
    PipelineErrorHandler,
)
from flare.utils.face_detect import FaceDetector
from flare.utils.metrics import PipelineMetrics


class BatchPipeline:
    """バッチ処理パイプライン。

    入力ディレクトリ内の動画ファイル群を処理し、3DMMパラメータを
    抽出して出力ディレクトリに保存する。

    仕様書§7に基づき、以下のフローで処理を行う:
        1. 入力ディレクトリの動画ファイルを列挙
        2. 各動画に対して顔検出 → 3DMMパラメータ抽出をフレームごとに実行
        3. 結果をnpz/npy形式で保存
        4. metadata.jsonとsummary.csvを生成

    Attributes:
        _error_handler: エラーハンドラインスタンス。
        _metrics: パフォーマンス計測インスタンス。
        _face_detector: 顔検出インスタンス。
        _extractor: 3DMMパラメータ抽出インスタンス（run時に設定）。
        _checkpoint_interval: チェックポイント保存間隔（フレーム数）。
    """

    _VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    """set[str]: サポートする動画ファイルの拡張子。"""

    def __init__(self, checkpoint_interval: int = 1000) -> None:
        """BatchPipelineを初期化する。

        Args:
            checkpoint_interval: チェックポイントの自動保存間隔（フレーム数）。
                仕様書§7.3に基づくデフォルト値は1000。
        """
        self._error_handler = PipelineErrorHandler()
        self._metrics = PipelineMetrics()
        self._face_detector: Optional[FaceDetector] = None
        self._checkpoint_interval = checkpoint_interval

    def run(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        config: PipelineConfig,
    ) -> None:
        """バッチ処理を実行する。

        入力ディレクトリ内の全動画ファイルを処理し、3DMMパラメータを
        出力ディレクトリに保存する。

        Args:
            input_dir: 入力動画ファイルが格納されたディレクトリのパス。
            output_dir: 出力先ディレクトリのパス。存在しない場合は自動作成。
            config: パイプライン設定。extractor / device_map / checkpoint 等を参照。

        Raises:
            FileNotFoundError: input_dirが存在しない場合。
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        self._face_detector = FaceDetector()

        video_files = sorted(
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in self._VIDEO_EXTENSIONS
        )

        if len(video_files) == 0:
            logger.warning("No video files found in {}", input_path)
            return

        logger.info(
            "Starting batch processing: {} videos in {}", len(video_files), input_path
        )

        summary_rows: list[dict[str, Any]] = []
        total_start = time.perf_counter()

        for video_idx, video_file in enumerate(video_files):
            logger.info(
                "[{}/{}] Processing: {}",
                video_idx + 1,
                len(video_files),
                video_file.name,
            )
            video_stats = self._process_single_video(
                video_file, output_path, config
            )
            summary_rows.append(video_stats)

        total_elapsed = time.perf_counter() - total_start

        overall_stats = {
            "total_videos": len(video_files),
            "total_time_sec": total_elapsed,
            "metrics": self._metrics.get_summary(),
        }

        self._save_metadata(output_path, config, overall_stats)
        self._save_summary_csv(output_path, summary_rows)

        if self._face_detector is not None:
            self._face_detector.release()

        logger.info(
            "Batch processing complete: {} videos in {:.1f}s",
            len(video_files),
            total_elapsed,
        )

    def _process_single_video(
        self,
        video_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> dict[str, Any]:
        """単一の動画ファイルを処理する。

        動画の全フレームに対して顔検出とクロッピングを行い、
        結果をnpz/npy形式で保存する。

        Args:
            video_path: 入力動画ファイルのパス。
            output_dir: 出力先ルートディレクトリ。
            config: パイプライン設定。

        Returns:
            処理統計の辞書。キー:
                - ``"video_name"``: 動画ファイル名。
                - ``"total_frames"``: 総フレーム数。
                - ``"processed_frames"``: 処理済みフレーム数。
                - ``"skipped_frames"``: スキップされたフレーム数。
                - ``"elapsed_sec"``: 処理時間（秒）。
        """
        video_name = video_path.stem
        video_output = output_dir / video_name
        video_output.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video: {}", video_path)
            return {
                "video_name": video_name,
                "total_frames": 0,
                "processed_frames": 0,
                "skipped_frames": 0,
                "elapsed_sec": 0.0,
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        crop_size = config.extractor.input_size

        crops_list: list[np.ndarray] = []
        processed = 0
        skipped = 0
        start_time = time.perf_counter()

        checkpoint_path = video_output / "checkpoint.json"
        start_frame = self._load_checkpoint(checkpoint_path)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(
                "Resuming {} from frame {}", video_name, start_frame
            )

        frame_idx = start_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                bbox = self._face_detector.detect(frame)
                if bbox is None:
                    raise FaceNotDetectedError(
                        f"No face detected at frame {frame_idx}"
                    )
                cropped = self._face_detector.crop_and_align(
                    frame, bbox, size=crop_size
                )
                crops_list.append(cropped)
                processed += 1

            except Exception as e:
                policy = self._error_handler.handle(
                    e, {"video": video_name, "frame_index": frame_idx}
                )
                if policy == ErrorPolicy.SKIP:
                    skipped += 1
                elif policy == ErrorPolicy.ABORT:
                    logger.critical(
                        "Aborting video {} at frame {}", video_name, frame_idx
                    )
                    break

            frame_idx += 1

            if (
                self._checkpoint_interval > 0
                and frame_idx % self._checkpoint_interval == 0
            ):
                self._save_checkpoint(
                    checkpoint_path, config, video_path, total_frames, frame_idx
                )

        cap.release()
        elapsed = time.perf_counter() - start_time

        if len(crops_list) > 0:
            crops_arr = np.stack(crops_list, axis=0)
            np.save(str(video_output / "crops.npy"), crops_arr)

        if processed > 0 and elapsed > 0:
            self._metrics.update(fps=processed / elapsed, dropped=skipped)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info(
            "  {} complete: {}/{} frames in {:.1f}s ({} skipped)",
            video_name,
            processed,
            total_frames,
            elapsed,
            skipped,
        )

        return {
            "video_name": video_name,
            "total_frames": total_frames,
            "processed_frames": processed,
            "skipped_frames": skipped,
            "elapsed_sec": elapsed,
        }

    def _save_metadata(
        self,
        output_dir: Path,
        config: PipelineConfig,
        stats: dict[str, Any],
    ) -> None:
        """メタデータをJSON形式で保存する。

        仕様書§7.2のmetadata.json形式に従い、データセット情報、
        パラメータ形式、バージョン情報を保存する。

        Args:
            output_dir: 出力ディレクトリのパス。
            config: パイプライン設定。
            stats: 処理統計の辞書。
        """
        metadata = {
            "version": "2.2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_config": {
                "name": config.pipeline.name,
                "extractor_type": config.extractor.type,
                "input_size": config.extractor.input_size,
                "device_map": {
                    "extractor": config.device_map.extractor,
                    "lhg_model": config.device_map.lhg_model,
                    "renderer": config.device_map.renderer,
                },
            },
            "stats": stats,
        }

        metadata_path = output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Metadata saved to {}", metadata_path)

    def _save_summary_csv(
        self,
        output_dir: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        """全動画の統計情報をCSV形式で保存する。

        Args:
            output_dir: 出力ディレクトリのパス。
            rows: 各動画の統計辞書のリスト。
        """
        if len(rows) == 0:
            return

        csv_path = output_dir / "summary.csv"
        fieldnames = list(rows[0].keys())

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Summary CSV saved to {}", csv_path)

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        config: PipelineConfig,
        input_source: Path,
        total_frames: int,
        processed_frames: int,
    ) -> None:
        """チェックポイントをJSON形式で保存する。

        仕様書§7.3のチェックポイント形式に従う。

        Args:
            checkpoint_path: チェックポイントファイルのパス。
            config: パイプライン設定。
            input_source: 入力動画ファイルのパス。
            total_frames: 動画の総フレーム数。
            processed_frames: 処理済みフレーム数。
        """
        checkpoint = {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_config": config.pipeline.name,
            "input_source": str(input_source),
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "last_frame_index": processed_frames - 1,
        }

        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        """チェックポイントから再開フレームインデックスを読み込む。

        Args:
            checkpoint_path: チェックポイントファイルのパス。

        Returns:
            再開すべきフレームインデックス。チェックポイントが存在しない場合は0。
        """
        if not checkpoint_path.exists():
            return 0

        try:
            with checkpoint_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            last_index = data.get("last_frame_index", -1)
            resume_from = last_index + 1
            logger.info(
                "Checkpoint loaded: resuming from frame {}", resume_from
            )
            return resume_from
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load checkpoint: {}, starting from 0", e)
            return 0
