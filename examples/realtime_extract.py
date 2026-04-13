#!/usr/bin/env python
"""リアルタイム特徴抽出パイプライン起動スクリプト。

Webカメラまたは動画ファイルから、リアルタイムに 3DMM 特徴量を抽出し、
LHG モデル推論 + レンダリング結果を表示する。

使用方法:

    # Webカメラ (デバイス 0) + FLAME ルート
    python examples/realtime_extract.py \\
        --config configs/realtime_flame.yaml \\
        --source 0

    # 動画ファイル入力
    python examples/realtime_extract.py \\
        --config configs/realtime_flame.yaml \\
        --source ./data/test_video.mp4

    # BFM ルート + PyQt6 GUI
    python examples/realtime_extract.py \\
        --config configs/realtime_bfm.yaml \\
        --source 0 \\
        --display pyqt

    # 設定なしでデフォルト (DECA + FlashAvatar)
    python examples/realtime_extract.py --source 0

表示ウィンドウで 'q' キーを押すと停止します。

詳細は docs/guide_realtime.md を参照。
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLARE リアルタイム特徴抽出パイプライン"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 設定ファイルパス。省略時はデフォルト設定",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="映像ソース。整数=Webカメラ ID、文字列=動画ファイルパス (既定: 0)",
    )
    parser.add_argument(
        "--display",
        default="opencv",
        choices=["opencv", "pyqt"],
        help="表示バックエンド (既定: opencv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from flare.config import PipelineConfig
    from flare.pipeline.realtime import RealtimePipeline

    # 設定読み込み
    if args.config is not None:
        print(f"Loading config: {args.config}")
        config = PipelineConfig.from_yaml(args.config)
    else:
        print("Using default config (DECA + FlashAvatar)")
        config = PipelineConfig()

    # ソースの解釈: 数字ならカメラ ID、それ以外はファイルパス
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    print(f"Source: {source}")
    print(f"Display: {args.display}")
    print(f"Extractor: {config.extractor.type}")
    print(f"Device: {config.device_map.extractor}")
    print("Press 'q' to stop")
    print("---")

    pipeline = RealtimePipeline(source=source, display_backend=args.display)
    pipeline.run(config)


if __name__ == "__main__":
    main()
