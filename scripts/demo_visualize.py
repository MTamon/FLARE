#!/usr/bin/env python
"""抽出済み 3DMM 特徴量の可視化デモスクリプト。

LHG パイプライン (``lhg-extract``) で抽出した npz ファイルから
顔画像 / メッシュを再構成し、動画として保存する。

2 つの可視化モードをサポート:

1. **mesh** モード (学習不要):
   FLAME パラメトリックモデルでメッシュ頂点を計算し、ワイヤフレーム描画。
   パラメータの定性的なサニティチェックに最適。

2. **neural** モード (学習済みデコーダ必要):
   ``train_face_decoder.py`` で学習した FaceDecoderNet を使って
   フォトリアルな顔画像を生成。

使用方法
--------

**メッシュ可視化 (学習不要)**

.. code-block:: bash

    python scripts/demo_visualize.py \\
        --npz data/movements/data001/comp/deca_comp_00000_04499.npz \\
        --mode mesh \\
        --flame-model ./checkpoints/generic_model.pkl \\
        --output demo_mesh.mp4

**ニューラルデコーダ可視化**

.. code-block:: bash

    python scripts/demo_visualize.py \\
        --npz data/movements/data001/comp/deca_comp_00000_04499.npz \\
        --mode neural \\
        --decoder-path ./checkpoints/face_decoder/person01/face_decoder.pth \\
        --source-image ./checkpoints/face_decoder/person01/source_image.png \\
        --output demo_neural.mp4

**複数 npz の連結可視化**

.. code-block:: bash

    python scripts/demo_visualize.py \\
        --npz-dir data/movements/data001/comp/ \\
        --mode mesh \\
        --flame-model ./checkpoints/generic_model.pkl \\
        --output demo_all.mp4

**画像として保存 (動画ではなく)**

.. code-block:: bash

    python scripts/demo_visualize.py \\
        --npz data/movements/data001/comp/deca_comp_00000_04499.npz \\
        --mode mesh \\
        --flame-model ./checkpoints/generic_model.pkl \\
        --output-dir demo_frames/ \\
        --save-frames
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


def load_npz(npz_path: str) -> dict[str, np.ndarray]:
    """npz ファイルを読み込み、デノーマライズ済みパラメータを返す。

    Args:
        npz_path: npz ファイルパス。

    Returns:
        デノーマライズ済みの辞書。キーは ``angle``, ``expression``,
        ``centroid``, ``shape``, ``jaw_pose`` (optional), ``fps`` 等。
    """
    data = np.load(npz_path, allow_pickle=True)
    result: dict[str, np.ndarray] = {}

    # デノーマライズ
    result["angle"] = data["angle"] * data["angle_std"] + data["angle_mean"]
    result["expression"] = (
        data["expression"] * data["expression_std"] + data["expression_mean"]
    )
    result["centroid"] = (
        data["centroid"] * data["centroid_std"] + data["centroid_mean"]
    )
    result["shape"] = data["shape"]
    result["section"] = data["section"]
    result["fps"] = float(data["fps"]) if "fps" in data else 30.0

    if "jaw_pose" in data:
        result["jaw_pose"] = data["jaw_pose"]
    if "face_size" in data:
        result["face_size"] = data["face_size"]
    if "speaker_name" in data:
        result["speaker_name"] = str(data["speaker_name"])
    if "extractor_type" in data:
        result["extractor_type"] = str(data["extractor_type"])

    return result


def load_multiple_npz(npz_dir: str) -> list[dict[str, np.ndarray]]:
    """ディレクトリ内の全 npz をロードし、section 順にソートして返す。"""
    path = Path(npz_dir)
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files in {npz_dir}")

    segments = [load_npz(str(f)) for f in files]
    segments.sort(key=lambda s: int(s["section"][0]))
    return segments


# ---------------------------------------------------------------------------
# Mesh mode
# ---------------------------------------------------------------------------


def render_mesh_sequence(
    segments: list[dict[str, np.ndarray]],
    flame_model_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_frames: bool = False,
    image_size: int = 512,
    max_frames: Optional[int] = None,
) -> None:
    """FLAME メッシュモードで可視化する。

    Args:
        segments: load_npz / load_multiple_npz の出力。
        flame_model_path: ``generic_model.pkl`` のパス。
        output_path: 出力動画パス (mp4)。
        output_dir: フレーム画像の出力ディレクトリ。
        save_frames: True なら個別フレーム画像も保存。
        image_size: 出力画像サイズ。
        max_frames: 最大レンダリングフレーム数。
    """
    from flare.decoders.flame_mesh_renderer import FLAMEMeshRenderer

    print(f"[Mesh] Loading FLAME model: {flame_model_path}")
    renderer = FLAMEMeshRenderer(flame_model_path, device="cpu")

    writer: Optional[cv2.VideoWriter] = None
    fps = segments[0]["fps"]

    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (image_size, image_size))

    if save_frames and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for seg in segments:
        T = seg["angle"].shape[0]
        shape = seg["shape"]
        has_jaw = "jaw_pose" in seg

        for t in range(T):
            if max_frames is not None and total_frames >= max_frames:
                break

            img = renderer.render(
                shape=shape,
                expression=seg["expression"][t],
                global_pose=seg["angle"][t],
                jaw_pose=seg["jaw_pose"][t] if has_jaw else None,
                image_size=image_size,
            )

            # フレーム情報テキスト
            info = f"Frame {total_frames}"
            if "speaker_name" in seg:
                info += f"  [{seg['speaker_name']}]"
            cv2.putText(
                img, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )

            if writer is not None:
                writer.write(img)
            if save_frames and output_dir is not None:
                cv2.imwrite(
                    str(Path(output_dir) / f"frame_{total_frames:06d}.png"), img
                )

            total_frames += 1

            if total_frames % 100 == 0:
                print(f"  Rendered {total_frames} frames...")

    if writer is not None:
        writer.release()
        print(f"[Mesh] Saved video: {output_path} ({total_frames} frames)")


# ---------------------------------------------------------------------------
# Neural decoder mode
# ---------------------------------------------------------------------------


def render_neural_sequence(
    segments: list[dict[str, np.ndarray]],
    decoder_path: str,
    source_image_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_frames: bool = False,
    device: str = "cuda:0",
    image_size: int = 256,
    max_frames: Optional[int] = None,
) -> None:
    """学習済みニューラルデコーダで可視化する。

    Args:
        segments: load_npz / load_multiple_npz の出力。
        decoder_path: 学習済みデコーダの ``face_decoder.pth`` パス。
        source_image_path: ソース画像パス (学習時に保存されたもの)。
        output_path: 出力動画パス。
        output_dir: フレーム画像出力ディレクトリ。
        save_frames: 個別フレーム画像保存の有無。
        device: 推論デバイス。
        image_size: 出力画像サイズ。
        max_frames: 最大フレーム数。
    """
    from flare.decoders.face_decoder_net import FaceDecoderNet

    dev = torch.device(device)

    # モデルロード
    print(f"[Neural] Loading decoder: {decoder_path}")
    ckpt = torch.load(decoder_path, map_location=dev, weights_only=True)
    cond_dim = ckpt["cond_dim"]

    model = FaceDecoderNet(cond_dim=cond_dim, pretrained_encoder=False).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ソース画像ロード
    print(f"[Neural] Loading source image: {source_image_path}")
    src_bgr = cv2.imread(source_image_path)
    if src_bgr is None:
        raise FileNotFoundError(f"Source image not found: {source_image_path}")
    src_rgb = cv2.cvtColor(
        cv2.resize(src_bgr, (image_size, image_size)), cv2.COLOR_BGR2RGB
    )
    src_tensor = (
        torch.from_numpy(src_rgb).float().permute(2, 0, 1) / 255.0
    ).unsqueeze(0).to(dev)  # (1, 3, 256, 256)

    writer: Optional[cv2.VideoWriter] = None
    fps = segments[0]["fps"]

    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 出力: source | generated を横に並べる
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (image_size * 2, image_size)
        )

    if save_frames and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_frames = 0

    with torch.no_grad():
        for seg in segments:
            T = seg["angle"].shape[0]
            has_jaw = "jaw_pose" in seg

            for t in range(T):
                if max_frames is not None and total_frames >= max_frames:
                    break

                # 条件ベクトル構築
                parts = [seg["expression"][t], seg["angle"][t]]
                if has_jaw:
                    parts.append(seg["jaw_pose"][t])
                cond = np.concatenate(parts).astype(np.float32)

                # cond_dim に合わせてパディング / トランケート
                if len(cond) < cond_dim:
                    cond = np.pad(cond, (0, cond_dim - len(cond)))
                elif len(cond) > cond_dim:
                    cond = cond[:cond_dim]

                cond_tensor = torch.from_numpy(cond).unsqueeze(0).to(dev)

                pred = model(src_tensor, cond_tensor)  # (1, 3, 256, 256)
                pred_np = (
                    pred[0].cpu().permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

                # 横並び画像
                src_display = cv2.resize(src_bgr, (image_size, image_size))
                combined = np.concatenate([src_display, pred_bgr], axis=1)

                cv2.putText(
                    combined, f"Frame {total_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                )

                if writer is not None:
                    writer.write(combined)
                if save_frames and output_dir is not None:
                    cv2.imwrite(
                        str(Path(output_dir) / f"frame_{total_frames:06d}.png"),
                        combined,
                    )

                total_frames += 1
                if total_frames % 100 == 0:
                    print(f"  Rendered {total_frames} frames...")

    if writer is not None:
        writer.release()
        print(f"[Neural] Saved video: {output_path} ({total_frames} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抽出済み 3DMM 特徴量からの顔可視化デモ"
    )

    # 入力
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", help="単一 npz ファイルパス")
    group.add_argument("--npz-dir", help="npz ディレクトリパス")

    # モード
    parser.add_argument(
        "--mode",
        choices=["mesh", "neural"],
        default="mesh",
        help="可視化モード: mesh (FLAME メッシュ) / neural (学習済みデコーダ)",
    )

    # Mesh モード用
    parser.add_argument(
        "--flame-model",
        default="./checkpoints/generic_model.pkl",
        help="FLAME generic_model.pkl のパス (mesh モード用)",
    )

    # Neural モード用
    parser.add_argument(
        "--decoder-path",
        default=None,
        help="学習済み face_decoder.pth のパス (neural モード用)",
    )
    parser.add_argument(
        "--source-image",
        default=None,
        help="ソース画像パス (neural モード用)",
    )

    # 出力
    parser.add_argument("--output", "-o", default=None, help="出力動画パス (.mp4)")
    parser.add_argument("--output-dir", default=None, help="フレーム画像出力ディレクトリ")
    parser.add_argument(
        "--save-frames", action="store_true", help="個別フレーム画像を保存"
    )

    # その他
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # npz ロード
    if args.npz is not None:
        segments = [load_npz(args.npz)]
        print(f"Loaded 1 npz: {args.npz}")
    else:
        segments = load_multiple_npz(args.npz_dir)
        print(f"Loaded {len(segments)} npz from {args.npz_dir}")

    total_frames = sum(s["angle"].shape[0] for s in segments)
    print(f"Total frames: {total_frames}")
    print(f"FPS: {segments[0]['fps']}")
    if "extractor_type" in segments[0]:
        print(f"Extractor: {segments[0]['extractor_type']}")

    if args.mode == "mesh":
        render_mesh_sequence(
            segments=segments,
            flame_model_path=args.flame_model,
            output_path=args.output,
            output_dir=args.output_dir,
            save_frames=args.save_frames,
            image_size=args.image_size,
            max_frames=args.max_frames,
        )

    elif args.mode == "neural":
        if args.decoder_path is None:
            raise ValueError("--decoder-path is required for neural mode")
        if args.source_image is None:
            raise ValueError("--source-image is required for neural mode")

        render_neural_sequence(
            segments=segments,
            decoder_path=args.decoder_path,
            source_image_path=args.source_image,
            output_path=args.output,
            output_dir=args.output_dir,
            save_frames=args.save_frames,
            device=args.device,
            image_size=min(args.image_size, 256),
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
