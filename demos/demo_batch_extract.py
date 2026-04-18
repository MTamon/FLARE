#!/usr/bin/env python
"""バッチ特徴抽出デモ (SMIRK / DECA)

data/multimodal_dialogue_formed/ 以下の動画ファイルを一括処理し、
per-frame の FLAME パラメータを .npz 形式で data/movements/ に保存する。

出力パス規約:
    data/movements/data<NNN>/{comp,host}/smirk_<role>_<start>_<end>.npz
                                        deca_<role>_<start>_<end>.npz
    既存の deca_*.npz 命名規約に準拠。

.npz の配列キー (SMIRK ルート):
    shape   (T, 300)  FLAME 形状パラメータ
    exp     (T,  50)  表情パラメータ
    pose    (T,   6)  姿勢 [global_rot(3), jaw(3)]
    cam     (T,   3)  カメラパラメータ
    eyelid  (T,   2)  瞼パラメータ (SMIRK ネイティブ)

.npz の配列キー (DECA ルート):
    shape   (T, 300)
    exp     (T,  50)
    pose    (T,   6)
    cam     (T,   3)

Usage:
    # SMIRK (デフォルト)
    python demos/demo_batch_extract.py \\
        --input_dir data/multimodal_dialogue_formed \\
        --output_dir data/movements \\
        --extractor smirk

    # DECA
    python demos/demo_batch_extract.py \\
        --input_dir data/multimodal_dialogue_formed \\
        --output_dir data/movements \\
        --extractor deca

    # GPU delegate + 上書き許可
    python demos/demo_batch_extract.py \\
        --mp_delegate gpu --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
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

from flare.utils.face_detect import FaceDetector


_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="バッチ特徴抽出デモ (SMIRK / DECA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        default="data/multimodal_dialogue_formed",
        help="入力動画ルートディレクトリ",
    )
    p.add_argument(
        "--output_dir",
        default="data/movements",
        help="出力 .npz ルートディレクトリ",
    )
    p.add_argument(
        "--extractor",
        choices=["smirk", "deca"],
        default="smirk",
        help="特徴量抽出器 (default: smirk)",
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
    p.add_argument(
        "--device",
        default="cuda:0",
        help="推論デバイス",
    )
    p.add_argument(
        "--mp_delegate",
        choices=["cpu", "gpu"],
        default="cpu",
        help="MediaPipe 顔検出の推論デバイス (default: cpu)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="既存 .npz を上書きする (省略時はスキップ)",
    )
    p.add_argument(
        "--log_path",
        default=None,
        help="失敗フレームのログファイルパス (省略時は output_dir/batch_extract.log)",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="1 動画あたりの最大処理フレーム数 (デバッグ用)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Extractor ローダ
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


# ---------------------------------------------------------------------------
# フレームテンソル変換
# ---------------------------------------------------------------------------

def _bgr_to_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# 1 動画の処理
# ---------------------------------------------------------------------------

def _process_video(
    video_path: Path,
    extractor,
    face_detector: FaceDetector,
    device: str,
    max_frames: Optional[int],
    fail_log: list[dict],
) -> Optional[dict[str, np.ndarray]]:
    """1 本の動画から全フレームの特徴量を抽出して ndarray の dict を返す。

    戻り値の各 ndarray は shape (T, dim)。抽出失敗フレームは直前値で補完。
    先頭フレームが失敗した場合は最初の成功フレームで埋める (後処理)。
    全フレームが失敗した場合は None を返す。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("動画を開けません: {}", video_path)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_f = max_frames or total
    logger.info("  フレーム数: {}, 上限: {}", total, max_f)

    accum: dict[str, list[np.ndarray]] = {}
    last_good: dict[str, np.ndarray] | None = None
    pending_head: list[int] = []

    frame_idx = 0
    t_start = time.perf_counter()

    while frame_idx < max_f:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        try:
            bbox = face_detector.detect(frame_bgr)
            if bbox is None:
                raise RuntimeError("顔を検出できませんでした")
            cropped = face_detector.crop_and_align(frame_bgr, bbox, size=224)
            tensor = _bgr_to_tensor(cropped, device)
            with torch.no_grad():
                params = extractor.extract(tensor)
            row = {k: v.squeeze(0).cpu().numpy() for k, v in params.items()}
            last_good = row
            if pending_head:
                for _ in pending_head:
                    for k, v in row.items():
                        accum.setdefault(k, []).append(v.copy())
                pending_head.clear()
            for k, v in row.items():
                accum.setdefault(k, []).append(v)

        except Exception as e:
            err_info = {"video": str(video_path), "frame": frame_idx, "reason": str(e)}
            fail_log.append(err_info)

            if last_good is not None:
                for k, v in last_good.items():
                    accum.setdefault(k, []).append(v.copy())
                logger.debug("frame {:05d}: 失敗→補完", frame_idx)
            else:
                pending_head.append(frame_idx)
                logger.debug("frame {:05d}: 失敗・先頭待機", frame_idx)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.perf_counter() - t_start
            fps = frame_idx / elapsed
            logger.info("  進捗: {:5d}/{} ({:.1f} fps)", frame_idx, max_f, fps)

    cap.release()

    if not accum:
        logger.warning("  全フレーム失敗: {}", video_path.name)
        return None

    return {k: np.stack(v, axis=0) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# 出力パス解決
# ---------------------------------------------------------------------------

def _resolve_output_path(
    video_path: Path,
    input_root: Path,
    output_root: Path,
    extractor_name: str,
) -> Path:
    """動画パスから出力 .npz パスを決定する。

    入力: data/multimodal_dialogue_formed/data001/comp/comp_host_0_1000.mp4
    出力: data/movements/data001/comp/smirk_host_0_1000.npz

    ベスト エフォートで input_root からの相対パスを使い、
    ファイル名先頭の役割部 (comp/host etc.) をそのまま残す。
    extractor_name (smirk / deca) をプレフィックスとして付与。
    """
    try:
        rel = video_path.relative_to(input_root)
    except ValueError:
        rel = Path(video_path.name)

    stem = rel.stem
    # 例: "comp_host_0_1000" → "smirk_host_0_1000" (先頭 role 部を extractor 名に置き換え)
    parts = stem.split("_", 1)
    new_stem = f"{extractor_name}_{parts[1]}" if len(parts) == 2 else f"{extractor_name}_{stem}"

    out_path = output_root / rel.parent / (new_stem + ".npz")
    return out_path


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_root = (_FLARE_ROOT / args.input_dir).resolve()
    output_root = (_FLARE_ROOT / args.output_dir).resolve()
    log_path = Path(args.log_path) if args.log_path else output_root / "batch_extract.log"

    logger.add(str(log_path), rotation="10 MB")
    logger.info("=== demo_batch_extract.py ===")
    logger.info("input_dir  : {}", input_root)
    logger.info("output_dir : {}", output_root)
    logger.info("extractor  : {}", args.extractor)
    logger.info("device     : {}", args.device)
    logger.info("mp_delegate: {}", args.mp_delegate)
    logger.info("overwrite  : {}", args.overwrite)

    if not input_root.exists():
        logger.error(
            "入力ディレクトリが見つかりません: {}\n"
            "  data/multimodal_dialogue_formed/ に動画ファイルを配置してください。",
            input_root,
        )
        sys.exit(1)

    # Extractor のロード
    if args.extractor == "smirk":
        if not Path(args.smirk_model).exists():
            logger.error(
                "SMIRK チェックポイントが見つかりません: {}\n"
                "  bash install/setup_smirk.sh を実行してください。",
                args.smirk_model,
            )
            sys.exit(1)
        extractor = _load_smirk(args.smirk_model, args.device)
    else:
        if not Path(args.deca_model).exists():
            logger.error(
                "DECA チェックポイントが見つかりません: {}\n"
                "  bash install/setup_deca.sh を実行してください。",
                args.deca_model,
            )
            sys.exit(1)
        extractor = _load_deca(args.deca_model, args.device)

    face_detector = FaceDetector()

    # 動画ファイルを列挙
    video_files = sorted(
        f for f in input_root.rglob("*") if f.suffix.lower() in _VIDEO_EXTENSIONS
    )
    if not video_files:
        logger.error("動画ファイルが見つかりません: {}", input_root)
        sys.exit(1)
    logger.info("動画ファイル数: {}", len(video_files))

    fail_log: list[dict] = []
    n_done = 0
    n_skip = 0
    n_fail = 0

    for i, video_path in enumerate(video_files, 1):
        out_path = _resolve_output_path(video_path, input_root, output_root, args.extractor)

        if out_path.exists() and not args.overwrite:
            logger.info("[{}/{}] スキップ (既存): {}", i, len(video_files), out_path.name)
            n_skip += 1
            continue

        logger.info("[{}/{}] 処理中: {}", i, len(video_files), video_path.name)
        result = _process_video(
            video_path, extractor, face_detector, args.device, args.max_frames, fail_log
        )

        if result is None:
            n_fail += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), **result)
        logger.info("  保存: {}", out_path)
        n_done += 1

    face_detector.release()

    # 失敗フレームログの保存
    if fail_log:
        fail_log_path = log_path.with_suffix(".fail.jsonl")
        with open(fail_log_path, "w", encoding="utf-8") as f:
            for rec in fail_log:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.warning(
            "失敗フレームを記録しました ({} 件): {}", len(fail_log), fail_log_path
        )

    logger.info("=== 完了: done={}, skip={}, fail={} ===", n_done, n_skip, n_fail)


if __name__ == "__main__":
    main()
