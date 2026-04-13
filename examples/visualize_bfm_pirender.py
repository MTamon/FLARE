#!/usr/bin/env python
"""BFM パラメータの PIRender 可視化スクリプト。

lhg-extract (BFM ルート) で抽出した npz ファイルから、
PIRender を使ってフォトリアルな顔画像動画を生成する。

使用方法:

    # 単一 npz → 動画
    python examples/visualize_bfm_pirender.py \\
        --npz data/movements/data001/comp/bfm_comp_00000_04499.npz \\
        --source-image ./data/source_portrait.png \\
        --pirender-model ./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt \\
        --pirender-dir ./PIRender \\
        --output demo_bfm.mp4

    # ディレクトリ内の全 npz を連結
    python examples/visualize_bfm_pirender.py \\
        --npz-dir data/movements/data001/comp/ \\
        --source-image ./data/source_portrait.png \\
        --pirender-model ./checkpoints/pirender/epoch_00190_*.pt \\
        --output demo_bfm_all.mp4

    # 最初の 300 フレームのみ
    python examples/visualize_bfm_pirender.py \\
        --npz ... --source-image ... --pirender-model ... \\
        --max-frames 300 --output short_demo.mp4

前提条件:
    - PIRender リポジトリがクローン済み (--pirender-dir で指定)
    - PIRender チェックポイントがダウンロード済み
    - 対象人物のソース画像 (正面顔) を 1 枚用意

詳細は docs/guide_bfm_visualization.md を参照。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


def load_npz_denormalized(npz_path: str) -> dict[str, np.ndarray]:
    """npz を読み込みデノーマライズして返す。"""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "expression": data["expression"] * data["expression_std"] + data["expression_mean"],
        "angle": data["angle"] * data["angle_std"] + data["angle_mean"],
        "centroid": data["centroid"] * data["centroid_std"] + data["centroid_mean"],
        "section": data["section"],
        "fps": float(data["fps"]) if "fps" in data else 30.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BFM パラメータの PIRender 可視化"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", help="単一 npz ファイルパス")
    group.add_argument("--npz-dir", help="npz ディレクトリパス")
    parser.add_argument(
        "--source-image", required=True, help="ソース画像 (対象人物の正面顔)"
    )
    parser.add_argument(
        "--pirender-model", required=True, help="PIRender チェックポイントパス"
    )
    parser.add_argument(
        "--pirender-dir", default="./PIRender", help="PIRender リポジトリパス"
    )
    parser.add_argument(
        "--output", "-o", default="output_bfm.mp4", help="出力動画パス"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    dev = torch.device(args.device)

    # npz 読み込み
    if args.npz is not None:
        segments = [load_npz_denormalized(args.npz)]
    else:
        npz_files = sorted(Path(args.npz_dir).glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No npz files in {args.npz_dir}")
        segments = [load_npz_denormalized(str(f)) for f in npz_files]
        segments.sort(key=lambda s: int(s["section"][0]))

    total_frames = sum(s["expression"].shape[0] for s in segments)
    fps = segments[0]["fps"]
    print(f"Loaded {len(segments)} segment(s), {total_frames} total frames, {fps} FPS")

    # PIRender 初期化
    from flare.renderers.pirender import PIRenderRenderer

    print(f"Loading PIRender: {args.pirender_model}")
    renderer = PIRenderRenderer(
        model_path=args.pirender_model,
        device=args.device,
        pirender_dir=args.pirender_dir,
    )

    # ソース画像ロード + setup
    print(f"Loading source image: {args.source_image}")
    src_bgr = cv2.imread(args.source_image)
    if src_bgr is None:
        raise FileNotFoundError(f"Source image not found: {args.source_image}")
    src_resized = cv2.resize(src_bgr, (256, 256))
    src_rgb = cv2.cvtColor(src_resized, cv2.COLOR_BGR2RGB)
    src_tensor = (
        torch.from_numpy(src_rgb).permute(2, 0, 1).float() / 255.0
    ).unsqueeze(0).to(dev)

    renderer.setup(source_image=src_tensor)
    print("PIRender ready")

    # 動画出力準備
    output_size = 256
    # source | rendered の横並び
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_size * 2, output_size),
    )

    rendered_count = 0
    with torch.no_grad():
        for seg in segments:
            T = seg["expression"].shape[0]
            for t in range(T):
                if args.max_frames is not None and rendered_count >= args.max_frames:
                    break

                exp = seg["expression"][t:t+1]       # (1, 64)
                angle = seg["angle"][t:t+1]           # (1, 3)
                centroid = seg["centroid"][t:t+1]      # (1, 3)

                # BFM pose = [rotation(3), translation(3)]
                pose = np.concatenate([angle, centroid[:, :3]], axis=-1)  # (1, 6)

                exp_t = torch.from_numpy(exp).float().to(dev)
                pose_t = torch.from_numpy(pose).float().to(dev)
                trans_t = torch.from_numpy(centroid).float().to(dev)

                output = renderer.render({
                    "exp": exp_t,
                    "pose": pose_t,
                    "trans": trans_t,
                })

                # テンソル → numpy BGR
                out_np = (
                    output[0].cpu().permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

                # 横並び: source | rendered
                combined = np.concatenate([src_resized, out_bgr], axis=1)
                cv2.putText(
                    combined,
                    f"Frame {rendered_count}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
                writer.write(combined)

                rendered_count += 1
                if rendered_count % 100 == 0:
                    print(f"  Rendered {rendered_count}/{total_frames} frames")

    writer.release()
    print(f"Saved: {args.output} ({rendered_count} frames)")


if __name__ == "__main__":
    main()
