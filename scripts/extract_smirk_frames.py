#!/usr/bin/env python
"""Video → per-frame SMIRK extraction for FlashAvatar training.

入力動画から全フレームを抽出し、SMIRK で FLAME パラメータを抽出して
per-frame の .pt ファイルとして保存する。出力ファイルは FlashAvatar の
FlameConverter (``--tracker smirk``) が直接読み込める形式。

ディレクトリ構成 (出力):
    <out_dir>/imgs/00001.jpg          # 1-indexed フレーム画像
    <out_dir>/imgs/00002.jpg
    ...
    <out_dir>/smirk_outputs/00000.pt  # 0-indexed SMIRK 出力 (.pt)
    <out_dir>/smirk_outputs/00001.pt
    ...
    <out_dir>/extract_log.json        # 処理ログ (スキップフレーム等)

インデックスのずれ (frame_delta=1):
    FlashAvatar の Scene_mica は frame_delta=1 を前提とする。
    .frame 00000 は imgs/00001.jpg に対応する。
    本スクリプトは imgs を 1-indexed、smirk_outputs を 0-indexed で保存し
    この対応関係を維持する (extract_deca_frames.py と同じ規約)。

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
    frame_224 = cv2.resize(frame_rgb, (224, 224))
    t = torch.from_numpy(frame_224).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    imgs_dir = out_dir / "imgs"
    smirk_dir_out = out_dir / "smirk_outputs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    smirk_dir_out.mkdir(parents=True, exist_ok=True)

    logger.add(str(out_dir / "extract_smirk.log"), rotation="10 MB")
    logger.info("=== extract_smirk_frames.py ===")
    logger.info("video   : {}", args.video)
    logger.info("out_dir : {}", out_dir)
    logger.info("device  : {}", args.device)
    logger.info("img_size: {}", args.img_size)

    extractor, face_detector = _load_extractor_and_detector(
        args.model_path, args.smirk_dir, args.device
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("動画を開けません: {}", args.video)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("動画フレーム数: {}, FPS: {:.1f}", total_frames, fps)

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

        img_name = f"{frame_idx + 1:05d}.jpg"
        img_path = imgs_dir / img_name
        frame_resized = cv2.resize(frame_bgr, (args.img_size, args.img_size))
        cv2.imwrite(str(img_path), frame_resized)

        pt_path = smirk_dir_out / f"{frame_idx:05d}.pt"
        if args.skip_existing and pt_path.exists():
            frame_idx += 1
            continue

        try:
            bbox = face_detector.detect(frame_bgr)
            if bbox is None:
                raise RuntimeError("顔を検出できませんでした")

            cropped = face_detector.crop_and_align(frame_bgr, bbox, size=224)
            image_tensor = _frame_to_tensor(cropped, args.device)

            with torch.no_grad():
                codedict = extractor.extract(image_tensor)

            codedict_cpu = {k: v.cpu() for k, v in codedict.items()}
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

    log = {
        "video": str(args.video),
        "extractor": "smirk",
        "total_extracted": frame_idx,
        "skipped_frames": skipped_frames,
        "carry_forward_frames": carry_forward_frames,
        "img_size": args.img_size,
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

    if skipped_frames:
        logger.warning(
            "抽出に完全失敗したフレーム ({} 件): {}",
            len(skipped_frames),
            skipped_frames[:10],
        )


if __name__ == "__main__":
    main()
