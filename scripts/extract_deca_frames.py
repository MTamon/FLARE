#!/usr/bin/env python
"""Video → per-frame DECA extraction for FlashAvatar training.

入力動画から全フレームを抽出し、DECA で FLAME パラメータを抽出して
per-frame の .pt ファイルとして保存する。出力ファイルは FlashAvatar の
FlameConverter が直接読み込める形式 (torch.save された dict) 。

ディレクトリ構成 (出力):
    <out_dir>/imgs/00001.jpg        # 1-indexed フレーム画像
    <out_dir>/imgs/00002.jpg
    ...
    <out_dir>/deca_outputs/00000.pt # 0-indexed DECA 出力 (.pt)
    <out_dir>/deca_outputs/00001.pt
    ...
    <out_dir>/extract_log.json      # 処理ログ (スキップフレーム等)

インデックスのずれ (frame_delta=1):
    FlashAvatar の Scene_mica は frame_delta=1 を前提とする。
    .frame 00000 は imgs/00001.jpg に対応する。
    本スクリプトは imgs を 1-indexed、deca_outputs を 0-indexed で保存し
    この対応関係を維持する。

Usage:
    python scripts/extract_deca_frames.py \\
        --video path/to/input.mp4 \\
        --out_dir data/flashavatar_training/person01 \\
        [--model_path checkpoints/deca/deca_model.tar] \\
        [--deca_dir third_party/DECA] \\
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
        description="Video → per-frame DECA extraction for FlashAvatar training"
    )
    p.add_argument("--video", required=True, help="入力動画ファイルのパス")
    p.add_argument(
        "--out_dir",
        required=True,
        help="出力ディレクトリ (imgs/, deca_outputs/ が作成される)",
    )
    p.add_argument(
        "--model_path",
        default="./checkpoints/deca/deca_model.tar",
        help="DECA チェックポイントパス",
    )
    p.add_argument(
        "--deca_dir",
        default="./third_party/DECA",
        help="DECA リポジトリのルートパス (sys.path 追加用)",
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
        help="既存の deca_outputs をスキップして再開",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="最大処理フレーム数 (デバッグ用)",
    )
    p.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help=(
            "出力フレームレート (fps)。元動画より低い値を指定するとフレームを間引く。"
            "省略時は全フレームを処理。FlashAvatar 論文設定に合わせるなら 25 を推奨。"
        ),
    )
    return p.parse_args()


def _load_extractor_and_detector(model_path: str, deca_dir: str, device: str):
    """DECAExtractor と FaceDetector を初期化する。"""
    from flare.extractors.deca import DECAExtractor
    from flare.utils.face_detect import FaceDetector

    logger.info("DECA モデルをロード中: {}", model_path)
    extractor = DECAExtractor(
        model_path=model_path,
        device=device,
        deca_dir=deca_dir,
    )
    face_detector = FaceDetector()
    logger.info("DECA + FaceDetector 初期化完了")
    return extractor, face_detector


def _frame_to_tensor(
    frame_bgr: np.ndarray, device: str = "cpu"
) -> torch.Tensor:
    """BGR numpy フレーム → DECA 用テンソル (1, 3, 224, 224)。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_224 = cv2.resize(frame_rgb, (224, 224))
    t = torch.from_numpy(frame_224).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    imgs_dir = out_dir / "imgs"
    deca_dir_out = out_dir / "deca_outputs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    deca_dir_out.mkdir(parents=True, exist_ok=True)

    logger.add(str(out_dir / "extract_deca.log"), rotation="10 MB")
    logger.info("=== extract_deca_frames.py ===")
    logger.info("video      : {}", args.video)
    logger.info("out_dir    : {}", out_dir)
    logger.info("device     : {}", args.device)
    logger.info("img_size   : {}", args.img_size)
    logger.info("target_fps : {}", args.target_fps if args.target_fps else "全フレーム (間引きなし)")

    # モデルのロード
    extractor, face_detector = _load_extractor_and_detector(
        args.model_path, args.deca_dir, args.device
    )

    # 動画のオープン
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("動画を開けません: {}", args.video)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    logger.info("動画フレーム数: {}, 元 FPS: {:.1f}", total_frames, source_fps)

    # FPS サブサンプリング設定
    # floor(raw_idx * target_fps / source_fps) が前フレームから増加したときだけ採用する。
    target_fps = args.target_fps
    if target_fps is not None and target_fps >= source_fps:
        logger.warning(
            "target_fps ({:.1f}) が元 FPS ({:.1f}) 以上のため間引きをスキップします",
            target_fps, source_fps,
        )
        target_fps = None

    # 前フレームのキャリーフォワード用
    last_good_codedict: dict[str, torch.Tensor] | None = None
    skipped_frames: list[int] = []
    carry_forward_frames: list[int] = []

    raw_idx = 0       # 元動画のフレーム通し番号 (読み込み順)
    frame_idx = 0     # 出力インデックス (deca_outputs / .frame のインデックス, 0-indexed)
    processed = 0
    max_frames = args.max_frames or total_frames

    import math

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # FPS サブサンプリング: このフレームを採用するか判定
        if target_fps is not None:
            keep = math.floor(raw_idx * target_fps / source_fps) > math.floor(
                (raw_idx - 1) * target_fps / source_fps
            )
            raw_idx += 1
            if not keep:
                continue
        else:
            raw_idx += 1

        if frame_idx >= max_frames:
            break

        # ---- imgs/ への保存 (1-indexed) ----
        img_name = f"{frame_idx + 1:05d}.jpg"
        img_path = imgs_dir / img_name
        frame_resized = cv2.resize(frame_bgr, (args.img_size, args.img_size))
        cv2.imwrite(str(img_path), frame_resized)

        # ---- 既存スキップ ----
        pt_path = deca_dir_out / f"{frame_idx:05d}.pt"
        if args.skip_existing and pt_path.exists():
            frame_idx += 1
            continue

        # ---- 顔検出 + DECA 抽出 ----
        try:
            bbox = face_detector.detect(frame_bgr)
            if bbox is None:
                raise RuntimeError("顔を検出できませんでした")

            cropped = face_detector.crop_and_align(frame_bgr, bbox, size=224)
            image_tensor = _frame_to_tensor(cropped, args.device)

            with torch.no_grad():
                codedict = extractor.extract(image_tensor)

            # CPU へ移動して保存
            codedict_cpu = {k: v.cpu() for k, v in codedict.items()}
            torch.save(codedict_cpu, str(pt_path))
            last_good_codedict = codedict_cpu

        except Exception as e:
            # 顔未検出・推論エラー → キャリーフォワード
            if last_good_codedict is not None:
                torch.save(last_good_codedict, str(pt_path))
                carry_forward_frames.append(frame_idx)
                logger.warning(
                    "frame {:05d}: 抽出失敗 ({}) → 直前フレームで補完",
                    frame_idx, e,
                )
            else:
                # 先頭フレームで失敗した場合はスキップ記録のみ
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
        # 最初の .pt ファイルを先頭スキップ分にコピー
        fallback_pt = None
        for i in range(frame_idx):
            pt = deca_dir_out / f"{i:05d}.pt"
            if pt.exists():
                fallback_pt = pt
                break
        if fallback_pt is not None:
            fb_data = torch.load(str(fallback_pt), map_location="cpu", weights_only=False)
            for idx in skipped_frames:
                dst = deca_dir_out / f"{idx:05d}.pt"
                if not dst.exists():
                    torch.save(fb_data, str(dst))
                    logger.warning(
                        "frame {:05d}: 先頭スキップを後続フレームで埋めました", idx
                    )

    # ---- ログ出力 ----
    log = {
        "video": str(args.video),
        "total_extracted": frame_idx,
        "source_fps": source_fps,
        "target_fps": args.target_fps,
        "total_raw_frames_scanned": raw_idx,
        "skipped_frames": skipped_frames,
        "carry_forward_frames": carry_forward_frames,
        "img_size": args.img_size,
        "imgs_dir": str(imgs_dir),
        "deca_outputs_dir": str(deca_dir_out),
    }
    with open(out_dir / "extract_log.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    logger.info(
        "=== 完了: {} フレーム抽出, {} フレームスキップ, {} フレームキャリーフォワード ===",
        frame_idx,
        len(skipped_frames),
        len(carry_forward_frames),
    )
    logger.info("imgs/        : {}", imgs_dir)
    logger.info("deca_outputs/: {}", deca_dir_out)

    if skipped_frames:
        logger.warning(
            "抽出に完全失敗したフレーム ({} 件): {}",
            len(skipped_frames),
            skipped_frames[:10],
        )


if __name__ == "__main__":
    main()
