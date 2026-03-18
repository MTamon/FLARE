"""CLI エントリポイント (Section 7.4)

サブコマンド:
  extract   — バッチ特徴量抽出 (BatchPipeline.run_extract)
  render    — バッチレンダリング (BatchPipeline.run_render)
  realtime  — リアルタイムパイプライン (RealtimePipeline.start)

使用例 (Section 7.4):
  python -m lhg_toolkit extract \\
      --input-dir /data/videos/ --output-dir /data/features/ \\
      --config config.yaml --batch-size 32

  python -m lhg_toolkit render \\
      --input-dir /data/features/ --output-dir /data/rendered/ \\
      --config config.yaml --avatar-model /models/avatar_001/ --resolution 512

  python -m lhg_toolkit realtime \\
      --config config.yaml --camera 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lhg_toolkit",
        description="LHG リアルタイム特徴量抽出・レンダリングツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- extract -----
    p_ext = subparsers.add_parser(
        "extract",
        help="バッチ特徴量抽出 (Section 7.4)",
    )
    p_ext.add_argument("--input-dir", type=Path, required=True,
                       help="入力動画ディレクトリ")
    p_ext.add_argument("--output-dir", type=Path, required=True,
                       help="出力ディレクトリ")
    p_ext.add_argument("--config", type=Path, required=True,
                       help="config.yaml のパス")
    p_ext.add_argument("--batch-size", type=int, default=32,
                       help="バッチサイズ (default: 32)")
    p_ext.add_argument("--gpu", type=int, default=0,
                       help="GPU デバイス番号 (default: 0)")
    p_ext.add_argument("--no-resume", action="store_true",
                       help="チェックポイントからの再開を無効化")

    # ----- render -----
    p_rnd = subparsers.add_parser(
        "render",
        help="バッチレンダリング (Section 7.4)",
    )
    p_rnd.add_argument("--input-dir", type=Path, required=True,
                       help="抽出済み特徴量ディレクトリ")
    p_rnd.add_argument("--output-dir", type=Path, required=True,
                       help="レンダリング出力ディレクトリ")
    p_rnd.add_argument("--config", type=Path, required=True,
                       help="config.yaml のパス")
    p_rnd.add_argument("--avatar-model", type=Path, default=None,
                       help="アバターモデルのパス (FlashAvatar 等)")
    p_rnd.add_argument("--resolution", type=int, default=512,
                       help="出力画像解像度 (default: 512)")
    p_rnd.add_argument("--gpu", type=int, default=0,
                       help="GPU デバイス番号 (default: 0)")

    # ----- realtime -----
    p_rt = subparsers.add_parser(
        "realtime",
        help="リアルタイムパイプライン (Section 6)",
    )
    p_rt.add_argument("--config", type=Path, required=True,
                      help="config.yaml のパス")
    p_rt.add_argument("--camera", type=int, default=0,
                      help="Webcam デバイス番号 (default: 0)")

    return parser


def _cmd_extract(args: argparse.Namespace) -> None:
    """extract サブコマンドの実行。"""
    from lhg_toolkit.config import load_config
    from lhg_toolkit.pipeline.batch import BatchPipeline
    from lhg_toolkit.utils.logging import setup_logging_from_config

    config = load_config(args.config)

    # GPU 指定を device_map に反映
    device = f"cuda:{args.gpu}"
    config.device_map.extractor = device
    config.device_map.lhg_model = device
    config.device_map.renderer = device

    setup_logging_from_config(config.logging)
    logger.info(f"extract command: input={args.input_dir}, output={args.output_dir}")

    pipeline = BatchPipeline(config)
    pipeline.run_extract(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )


def _cmd_render(args: argparse.Namespace) -> None:
    """render サブコマンドの実行。"""
    from lhg_toolkit.config import load_config
    from lhg_toolkit.pipeline.batch import BatchPipeline
    from lhg_toolkit.utils.logging import setup_logging_from_config

    config = load_config(args.config)

    device = f"cuda:{args.gpu}"
    config.device_map.extractor = device
    config.device_map.lhg_model = device
    config.device_map.renderer = device

    setup_logging_from_config(config.logging)
    logger.info(f"render command: input={args.input_dir}, output={args.output_dir}")

    pipeline = BatchPipeline(config)
    pipeline.run_render(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        avatar_model=args.avatar_model,
        resolution=args.resolution,
    )


def _cmd_realtime(args: argparse.Namespace) -> None:
    """realtime サブコマンドの実行。"""
    from lhg_toolkit.config import load_config
    from lhg_toolkit.pipeline.realtime import RealtimePipeline
    from lhg_toolkit.utils.logging import setup_logging_from_config

    config = load_config(args.config)
    setup_logging_from_config(config.logging)
    logger.info(f"realtime command: camera={args.camera}")

    pipeline = RealtimePipeline(config)
    try:
        pipeline.start(camera_id=args.camera)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        pipeline.stop()


def main() -> None:
    """CLI メインエントリポイント。"""
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "extract": _cmd_extract,
        "render": _cmd_render,
        "realtime": _cmd_realtime,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()