#!/usr/bin/env python
"""動画ファイルを使った FlashAvatar レンダリングテストデモ。

指定の動画ファイルから SMIRK (または DECA) で特徴抽出を行い、
学習済み FlashAvatar で各フレームをレンダリングして動画ファイルに保存する。

出力動画: 入力フレーム (左) と FlashAvatar レンダリング (右) を横並びにした
サイド・バイ・サイド動画。

前提:
    - FlashAvatar が third_party/FlashAvatar に初期化されていること
    - 対象人物の FlashAvatar モデルが学習済みであること
    - SMIRK チェックポイント (または DECA チェックポイント) が配置されていること

Usage:
    # SMIRK で特徴抽出 → FlashAvatar でレンダリング
    python demos/demo_video_render.py \\
        --input_video data/raw/sample.mp4 \\
        --output_video output/rendered.mp4 \\
        --checkpoint_dir checkpoints/flashavatar/person01

    # DECA ルート
    python demos/demo_video_render.py \\
        --input_video data/raw/sample.mp4 \\
        --extractor deca \\
        --checkpoint_dir checkpoints/flashavatar/person01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# FLARE ルートを sys.path に追加
_DEMO_DIR = Path(__file__).resolve().parent
_FLARE_ROOT = _DEMO_DIR.parent
if str(_FLARE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLARE_ROOT))

import cv2
import numpy as np
import torch
from loguru import logger

from flare.converters.deca_to_flame import DECAToFlameAdapter
from flare.converters.smirk_to_flashavatar import SmirkToFlashAvatarAdapter
from flare.utils.face_detect import FaceDetector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="動画ファイルでの FlashAvatar レンダリングテストデモ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_video", required=True, help="入力動画ファイルパス")
    p.add_argument(
        "--output_video",
        default=None,
        help="出力動画ファイルパス (省略時は input_video と同ディレクトリに _rendered.mp4 を追記)",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="./checkpoints/flashavatar/",
        help="FlashAvatar チェックポイントディレクトリ",
    )
    p.add_argument(
        "--extractor",
        choices=["smirk", "deca"],
        default="smirk",
        help="特徴抽出器 (default: smirk)",
    )
    p.add_argument(
        "--smirk_model",
        default="./checkpoints/smirk/SMIRK_em1.pt",
        help="SMIRK チェックポイントパス",
    )
    p.add_argument(
        "--deca_model",
        default="./checkpoints/deca/deca_model.tar",
        help="DECA チェックポイントパス",
    )
    p.add_argument("--device", default="cuda:0", help="推論デバイス")
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="最大処理フレーム数 (省略時は全フレーム)",
    )
    p.add_argument(
        "--display_width",
        type=int,
        default=512,
        help="出力動画の 1 ペイン幅 (px)。横幅は 2 倍になる",
    )
    p.add_argument(
        "--no_eye_supplement",
        action="store_true",
        help="DECA 使用時に MediaPipe による eyes_pose / eyelids 補完を無効化",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# ローダ
# ---------------------------------------------------------------------------

def _load_smirk(model_path: str, device: str):
    from flare.extractors.smirk import SMIRKExtractor
    logger.info("SMIRK ロード中: {}", model_path)
    return SMIRKExtractor(
        model_path=model_path,
        device=device,
        smirk_dir=str(_FLARE_ROOT / "third_party" / "smirk"),
    )


def _load_deca(model_path: str, device: str):
    from flare.extractors.deca import DECAExtractor
    logger.info("DECA ロード中: {}", model_path)
    return DECAExtractor(
        model_path=model_path,
        device=device,
        deca_dir=str(_FLARE_ROOT / "third_party" / "DECA"),
    )


def _load_flashavatar(checkpoint_dir: str, device: str):
    from flare.renderers.flashavatar import FlashAvatarRenderer
    logger.info("FlashAvatar ロード中: {}", checkpoint_dir)
    renderer = FlashAvatarRenderer(model_path=checkpoint_dir, device=device)
    renderer.setup()
    return renderer


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = args.device
    input_video = Path(args.input_video)

    if not input_video.exists():
        logger.error(
            "入力動画が見つかりません: {}\n"
            "  --input_video に正しいパスを指定してください。",
            input_video,
        )
        sys.exit(1)

    if args.output_video:
        output_video = Path(args.output_video)
    else:
        output_video = input_video.with_name(input_video.stem + "_rendered.mp4")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    # --- モデルロード ---
    if args.extractor == "smirk":
        if not Path(args.smirk_model).exists():
            logger.error(
                "SMIRK チェックポイントが見つかりません: {}\n"
                "  bash install/setup_smirk.sh を実行してください。",
                args.smirk_model,
            )
            sys.exit(1)
        extractor = _load_smirk(args.smirk_model, device)
        adapter = SmirkToFlashAvatarAdapter()
    else:
        if not Path(args.deca_model).exists():
            logger.error(
                "DECA チェックポイントが見つかりません: {}\n"
                "  bash install/setup_deca.sh を実行してください。",
                args.deca_model,
            )
            sys.exit(1)
        extractor = _load_deca(args.deca_model, device)
        # DECA は eyes_pose / eyelids を出力しないため、MediaPipe で補完する
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=not args.no_eye_supplement)

    if not Path(args.checkpoint_dir).exists():
        logger.error(
            "FlashAvatar チェックポイントが見つかりません: {}\n"
            "  scripts/train_flashavatar.py で対象人物を学習してから実行してください。",
            args.checkpoint_dir,
        )
        sys.exit(1)

    renderer = _load_flashavatar(args.checkpoint_dir, device)
    face_detector = FaceDetector()

    # --- 動画オープン ---
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        logger.error("動画を開けません: {}", input_video)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_f = args.max_frames or total
    pw = args.display_width  # 1 ペインの幅

    logger.info("入力動画: {} ({} frames @ {:.1f} fps)", input_video.name, total, fps)

    out_writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (pw * 2, pw),
    )
    if not out_writer.isOpened():
        logger.error("出力動画を作成できません: {}", output_video)
        sys.exit(1)

    last_render: Optional[np.ndarray] = None
    n_ok = 0
    n_fail = 0
    t0 = time.perf_counter()

    for frame_idx in range(max_f):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        try:
            bbox = face_detector.detect(frame_bgr)
            if bbox is None:
                raise RuntimeError("顔を検出できませんでした")

            cropped = face_detector.crop_and_align(
                frame_bgr, bbox, size=224, margin_scale=1.25
            )
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

            with torch.no_grad():
                params = extractor.extract(tensor)
                if args.extractor == "deca" and not args.no_eye_supplement:
                    eyes_pose, eyelids = face_detector.detect_eye_pose(frame_bgr, bbox)
                    params = dict(params)
                    params["eyes_pose"] = eyes_pose.to(device)
                    params["eyelids"] = eyelids.to(device)
                flash_params = adapter.convert(params)
                rendered = renderer.render(flash_params)

            rendered_rgb = (
                rendered[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)

            last_render = cv2.resize(rendered_bgr, (pw, pw))
            n_ok += 1

        except Exception as e:
            logger.debug("frame {:05d}: 失敗 ({})", frame_idx, e)
            n_fail += 1

        left = cv2.resize(frame_bgr, (pw, pw))
        right = last_render if last_render is not None else np.zeros((pw, pw, 3), np.uint8)
        out_writer.write(np.hstack([left, right]))

        if (frame_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                "進捗: {:5d}/{} ({:.1f} fps, ok={}, fail={})",
                frame_idx + 1, max_f, (frame_idx + 1) / elapsed, n_ok, n_fail,
            )

    cap.release()
    out_writer.release()
    face_detector.release()

    elapsed = time.perf_counter() - t0
    logger.info(
        "=== 完了: {} フレーム処理 ({:.1f} fps), ok={}, fail={} ===",
        frame_idx + 1, (frame_idx + 1) / elapsed, n_ok, n_fail,
    )
    logger.info("出力動画: {}", output_video)


if __name__ == "__main__":
    main()
