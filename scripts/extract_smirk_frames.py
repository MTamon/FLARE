#!/usr/bin/env python
"""Video → per-frame SMIRK extraction for FlashAvatar training.

入力動画から全フレームを抽出し、SMIRK で FLAME パラメータを抽出して
per-frame の .pt ファイルとして保存する。出力ファイルは FlashAvatar の
FlameConverter (``--tracker smirk``) が直接読み込める形式。

ディレクトリ構成 (出力):
    <out_dir>/imgs/00001.jpg          # 1-indexed フレーム画像 (固定クロップ)
    <out_dir>/imgs/00002.jpg
    ...
    <out_dir>/smirk_outputs/00000.pt  # 0-indexed SMIRK 出力 (.pt)
    <out_dir>/smirk_outputs/00001.pt
    ...
    <out_dir>/crop_region.json        # 固定クロップ領域 (デバッグ用)
    <out_dir>/extract_log.json        # 処理ログ (スキップフレーム等)

インデックスのずれ (frame_delta=1):
    FlashAvatar の Scene_mica は frame_delta=1 を前提とする。
    .frame 00000 は imgs/00001.jpg に対応する。
    本スクリプトは imgs を 1-indexed、smirk_outputs を 0-indexed で保存し
    この対応関係を維持する (extract_deca_frames.py と同じ規約)。

クロップ戦略 (2-pass, extract_deca_frames.py と共通):
    Phase 0: 先頭 ``--bbox_sample_frames`` 枚 (デフォルト 200) から bbox を
        収集 → 中央値 + ``--fixed_margin`` (デフォルト 2.0) で正方形領域を
        決定し ``crop_region.json`` に保存。
    Phase 1: imgs/ には固定領域を切り抜いて保存 (FlashAvatar の cam が
        フレーム間で安定する)。SMIRK に渡す画像は毎フレーム個別検出 +
        ``--smirk_margin`` (デフォルト 1.25) で再クロップ → 224×224。

DECA との違い:
    - Extractor: SMIRKExtractor (MTamon/smirk@release/cuda128)
    - 出力キー: shape(300) / exp(50) / pose(6) / cam(3) / eyelid(2)
      ※ DECA は eyelid を持たないが、SMIRK はネイティブで持つ。
    - 出力サブディレクトリ名: ``smirk_outputs/`` (DECA は ``deca_outputs/``)

Usage:
    python scripts/extract_smirk_frames.py \\
        --video path/to/input.mp4 \\
        --out_dir data/flashavatar_training/person01 \\
        [--model_path checkpoints/smirk/SMIRK_em1.pt] \\
        [--smirk_dir third_party/smirk] \\
        [--device cuda:0] \\
        [--img_size 512] \\
        [--bbox_sample_frames 200] \\
        [--fixed_margin 2.0] \\
        [--smirk_margin 1.25] \\
        [--skip_existing]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video → per-frame SMIRK extraction for FlashAvatar training"
    )
    p.add_argument("--video", required=True, help="入力動画ファイルのパス")
    p.add_argument(
        "--out_dir",
        required=True,
        help="出力ディレクトリ (imgs/, smirk_outputs/ が作成される)",
    )
    p.add_argument(
        "--model_path",
        default="./checkpoints/smirk/SMIRK_em1.pt",
        help="SMIRK チェックポイントパス (SMIRK_em1.pt)",
    )
    p.add_argument(
        "--smirk_dir",
        default="./third_party/smirk",
        help="SMIRK リポジトリのルートパス (sys.path 追加用)",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="推論デバイス",
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="保存する imgs フレームの一辺サイズ (px)",
    )
    p.add_argument(
        "--bbox_sample_frames",
        type=int,
        default=200,
        help="Phase 0 で bbox 収集に使うフレーム数 (動画長より小さい場合は先頭から)",
    )
    p.add_argument(
        "--fixed_margin",
        type=float,
        default=2.0,
        help="imgs/ 用固定クロップの長辺マージン倍率 (頭+首が入る程度)",
    )
    p.add_argument(
        "--smirk_margin",
        type=float,
        default=1.25,
        help="SMIRK 入力 (224) 用の per-frame 再クロップマージン倍率",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="既存の smirk_outputs をスキップして再開",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="最大処理フレーム数 (デバッグ用)",
    )
    return p.parse_args()


def _load_extractor_and_detector(model_path: str, smirk_dir: str, device: str):
    """SMIRKExtractor と FaceDetector を初期化する。"""
    from flare.extractors.smirk import SMIRKExtractor
    from flare.utils.face_detect import FaceDetector

    logger.info("SMIRK モデルをロード中: {}", model_path)
    extractor = SMIRKExtractor(
        model_path=model_path,
        device=device,
        smirk_dir=smirk_dir,
    )
    face_detector = FaceDetector()
    logger.info("SMIRK + FaceDetector 初期化完了")
    return extractor, face_detector


def _frame_to_tensor(
    frame_bgr: np.ndarray, device: str = "cpu"
) -> torch.Tensor:
    """BGR numpy フレーム → SMIRK 用テンソル (1, 3, 224, 224)。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.shape[:2] != (224, 224):
        frame_rgb = cv2.resize(frame_rgb, (224, 224))
    t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def _crop_with_bbox(
    frame: np.ndarray, bbox: tuple[int, int, int, int], size: int
) -> np.ndarray:
    """固定 bbox でクロップ + ゼロパディング + 指定サイズへリサイズ。"""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = bbox

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(w, x2)
    cy2 = min(h, y2)

    cropped = frame[cy1:cy2, cx1:cx2]
    if pad_left or pad_top or pad_right or pad_bottom:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)


def _collect_fixed_bbox(
    video_path: str,
    face_detector,
    sample_frames: int,
    fixed_margin: float,
) -> tuple[tuple[int, int, int, int], dict]:
    """Phase 0: 動画先頭から bbox を収集し、固定クロップ領域を算出する。

    Returns:
        (fixed_bbox, info_dict)。info_dict は crop_region.json に書く内容。
    """
    from flare.utils.face_detect import compute_fixed_bbox

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    bboxes: list[tuple[int, int, int, int]] = []
    frame_shape: tuple[int, int] | None = None
    scanned = 0
    detected = 0

    while scanned < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_shape is None:
            frame_shape = (frame.shape[0], frame.shape[1])
        scanned += 1
        bbox = face_detector.detect(frame)
        if bbox is not None:
            bboxes.append(bbox)
            detected += 1

    cap.release()

    if not bboxes:
        raise RuntimeError(
            f"Phase 0: 先頭 {scanned} フレームで顔を 1 度も検出できませんでした"
        )

    fixed_bbox = compute_fixed_bbox(
        bboxes, margin_scale=fixed_margin, frame_shape=None
    )
    logger.info(
        "Phase 0: scanned={}, detected={}, fixed_bbox={} (margin={})",
        scanned, detected, fixed_bbox, fixed_margin,
    )

    info = {
        "video": str(video_path),
        "phase0_scanned_frames": scanned,
        "phase0_detected_frames": detected,
        "fixed_margin_scale": fixed_margin,
        "frame_shape_hw": list(frame_shape) if frame_shape else None,
        "fixed_bbox_xyxy": list(fixed_bbox),
    }
    return fixed_bbox, info


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    imgs_dir = out_dir / "imgs"
    smirk_dir_out = out_dir / "smirk_outputs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    smirk_dir_out.mkdir(parents=True, exist_ok=True)

    logger.add(str(out_dir / "extract_smirk.log"), rotation="10 MB")
    logger.info("=== extract_smirk_frames.py ===")
    logger.info("video        : {}", args.video)
    logger.info("out_dir      : {}", out_dir)
    logger.info("device       : {}", args.device)
    logger.info("img_size     : {}", args.img_size)
    logger.info("fixed_margin : {}  (Phase 0 imgs/ クロップ)", args.fixed_margin)
    logger.info("smirk_margin : {}  (per-frame SMIRK 入力)", args.smirk_margin)

    # モデルのロード
    extractor, face_detector = _load_extractor_and_detector(
        args.model_path, args.smirk_dir, args.device
    )

    # ---- Phase 0: 固定クロップ領域の決定 ----
    fixed_bbox, crop_info = _collect_fixed_bbox(
        args.video, face_detector,
        sample_frames=args.bbox_sample_frames,
        fixed_margin=args.fixed_margin,
    )
    crop_info["smirk_margin_scale"] = args.smirk_margin
    crop_info["img_size"] = args.img_size
    with open(out_dir / "crop_region.json", "w") as f:
        json.dump(crop_info, f, indent=2, ensure_ascii=False)

    # ---- Phase 1: 抽出 ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("動画を開けません: {}", args.video)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("動画フレーム数: {}, FPS: {:.1f}", total_frames, fps)

    # 頭部位置考慮モード用のメタデータ (すべての .pt で共通)
    orig_img_size_t = torch.tensor([orig_w, orig_h], dtype=torch.float32)
    crop_bbox_t = torch.tensor(list(fixed_bbox), dtype=torch.float32)
    crop_img_size_t = torch.tensor([args.img_size, args.img_size], dtype=torch.float32)

    last_good_codedict: dict[str, torch.Tensor] | None = None
    skipped_frames: list[int] = []
    carry_forward_frames: list[int] = []

    frame_idx = 0
    processed = 0
    max_frames = args.max_frames or total_frames

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        # ---- imgs/ への保存 (固定クロップ + img_size リサイズ) ----
        img_name = f"{frame_idx + 1:05d}.jpg"
        img_path = imgs_dir / img_name
        fixed_crop = _crop_with_bbox(frame_bgr, fixed_bbox, args.img_size)
        cv2.imwrite(str(img_path), fixed_crop)

        # ---- 既存スキップ ----
        pt_path = smirk_dir_out / f"{frame_idx:05d}.pt"
        if args.skip_existing and pt_path.exists():
            frame_idx += 1
            continue

        # ---- 顔検出 + SMIRK 抽出 (per-frame, smirk_margin で再クロップ) ----
        try:
            bbox = face_detector.detect(frame_bgr)
            if bbox is None:
                raise RuntimeError("顔を検出できませんでした")

            cropped = face_detector.crop_and_align(
                frame_bgr, bbox, size=224, margin_scale=args.smirk_margin
            )
            image_tensor = _frame_to_tensor(cropped, args.device)

            with torch.no_grad():
                codedict = extractor.extract(image_tensor)

            codedict_cpu = {k: v.cpu() for k, v in codedict.items()}

            # 頭部位置考慮モード用の per-frame メタデータ
            bx1, by1, bx2, by2 = bbox
            bbox_center = torch.tensor(
                [(bx1 + bx2) / 2.0, (by1 + by2) / 2.0], dtype=torch.float32
            )
            bbox_scale = torch.tensor(
                [float(min(bx2 - bx1, by2 - by1))], dtype=torch.float32
            )
            codedict_cpu["bbox_center"] = bbox_center
            codedict_cpu["bbox_scale"] = bbox_scale
            codedict_cpu["img_size"] = orig_img_size_t.clone()
            codedict_cpu["crop_bbox"] = crop_bbox_t.clone()
            codedict_cpu["crop_img_size"] = crop_img_size_t.clone()

            torch.save(codedict_cpu, str(pt_path))
            last_good_codedict = codedict_cpu

        except Exception as e:
            if last_good_codedict is not None:
                torch.save(last_good_codedict, str(pt_path))
                carry_forward_frames.append(frame_idx)
                logger.warning(
                    "frame {:05d}: 抽出失敗 ({}) → 直前フレームで補完",
                    frame_idx, e,
                )
            else:
                skipped_frames.append(frame_idx)
                logger.error(
                    "frame {:05d}: 抽出失敗・補完不可 ({})", frame_idx, e
                )

        frame_idx += 1
        processed += 1

        if processed % 100 == 0:
            logger.info("進捗: {}/{} フレーム処理済み", processed, max_frames)

    cap.release()
    face_detector.release()

    # ---- スキップ埋め (先頭スキップは後続の最初の成功フレームで埋める) ----
    if skipped_frames:
        fallback_pt = None
        for i in range(frame_idx):
            pt = smirk_dir_out / f"{i:05d}.pt"
            if pt.exists():
                fallback_pt = pt
                break
        if fallback_pt is not None:
            fb_data = torch.load(str(fallback_pt), map_location="cpu", weights_only=False)
            for idx in skipped_frames:
                dst = smirk_dir_out / f"{idx:05d}.pt"
                if not dst.exists():
                    torch.save(fb_data, str(dst))
                    logger.warning(
                        "frame {:05d}: 先頭スキップを後続フレームで埋めました", idx
                    )

    # ---- ログ出力 ----
    log = {
        "video": str(args.video),
        "extractor": "smirk",
        "total_extracted": frame_idx,
        "skipped_frames": skipped_frames,
        "carry_forward_frames": carry_forward_frames,
        "img_size": args.img_size,
        "fixed_margin_scale": args.fixed_margin,
        "smirk_margin_scale": args.smirk_margin,
        "fixed_bbox_xyxy": list(fixed_bbox),
        "imgs_dir": str(imgs_dir),
        "smirk_outputs_dir": str(smirk_dir_out),
    }
    with open(out_dir / "extract_log.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    logger.info(
        "=== 完了: {} フレーム抽出, {} フレームスキップ, {} フレームキャリーフォワード ===",
        frame_idx,
        len(skipped_frames),
        len(carry_forward_frames),
    )
    logger.info("imgs/         : {}", imgs_dir)
    logger.info("smirk_outputs/: {}", smirk_dir_out)
    logger.info("crop_region   : {}", out_dir / "crop_region.json")

    if skipped_frames:
        logger.warning(
            "抽出に完全失敗したフレーム ({} 件): {}",
            len(skipped_frames),
            skipped_frames[:10],
        )


if __name__ == "__main__":
    main()
