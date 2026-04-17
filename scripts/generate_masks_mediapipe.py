#!/usr/bin/env python
"""MediaPipe を使った FlashAvatar 学習用マスク生成スクリプト。

FlashAvatar の学習には以下の 3 種類のマスクが必要:

    parsing/<frame>_neckhead.png  二値マスク: 頭部・頸部領域
    parsing/<frame>_mouth.png     二値マスク: 口の内部領域
    alpha/<frame>.jpg             グレースケール: 前景アルファマスク

本スクリプトは MediaPipe のみを使用して上記マスクをすべて生成する
(外部の face-parsing モデルは不要)。

  頭・頸部マスク: MediaPipe FaceMesh の顔輪郭 + 下方向へ膨張
  口マスク      : MediaPipe FaceMesh の口唇ランドマーク内部
  アルファマスク : MediaPipe SelfieSegmentation

Usage:
    python scripts/generate_masks_mediapipe.py \\
        --imgs_dir  data/flashavatar_training/person01/imgs \\
        --out_dir   data/flashavatar_training/person01 \\
        [--img_size 512] \\
        [--skip_existing]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe が見つかりません。pip install mediapipe を実行してください。")
    sys.exit(1)


# ---------------------------------------------------------------------------
# MediaPipe 定数
# ---------------------------------------------------------------------------

# 顔の輪郭ランドマーク (FaceMesh FACEMESH_CONTOURS の外縁)
_FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# 口の外唇ランドマーク (口唇輪郭)
_LIPS_OUTER_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# 口の内唇ランドマーク (口腔内マスク)
_LIPS_INNER_IDX = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95,
]


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _landmarks_to_points(
    landmarks, w: int, h: int, indices: list[int]
) -> np.ndarray:
    """FaceMesh landmarks → (N, 2) int32 ピクセル座標。"""
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)


def _generate_neckhead_mask(
    landmarks, w: int, h: int, neck_extend_ratio: float = 0.45
) -> np.ndarray:
    """顔輪郭から頭部・頸部マスクを生成する。

    顔輪郭 (Face Oval) を取得し、下方向に neck_extend_ratio * 高さ分
    延長した多角形で塗りつぶす。楕円で包んだ後に膨張処理で頸部を含む
    自然なマスクを生成する。

    Returns:
        (h, w) uint8 binary mask [0 or 255]
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = _landmarks_to_points(landmarks, w, h, _FACE_OVAL_IDX)

    # 顔の下端 y 座標を基準に首方向へ延長
    y_max = pts[:, 1].max()
    neck_y = min(h - 1, int(y_max + neck_extend_ratio * h))
    x_center = int(pts[:, 0].mean())
    x_half_width = int((pts[:, 0].max() - pts[:, 0].min()) / 2 * 1.1)

    # 下端を延長した追加点
    extra = np.array([
        [x_center - x_half_width, y_max],
        [x_center - x_half_width, neck_y],
        [x_center + x_half_width, neck_y],
        [x_center + x_half_width, y_max],
    ], dtype=np.int32)
    full_pts = np.vstack([pts, extra])

    hull = cv2.convexHull(full_pts)
    cv2.fillPoly(mask, [hull], 255)

    # 小さなノイズ除去と膨張
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _generate_mouth_mask(
    landmarks, w: int, h: int, dilate_px: int = 4
) -> np.ndarray:
    """口唇ランドマークから口の内部マスクを生成する。

    Returns:
        (h, w) uint8 binary mask [0 or 255]
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    inner_pts = _landmarks_to_points(landmarks, w, h, _LIPS_INNER_IDX)
    outer_pts = _landmarks_to_points(landmarks, w, h, _LIPS_OUTER_IDX)

    hull_inner = cv2.convexHull(inner_pts)
    hull_outer = cv2.convexHull(outer_pts)
    all_pts = np.vstack([inner_pts, outer_pts])
    hull_all = cv2.convexHull(all_pts)

    cv2.fillPoly(mask, [hull_all], 255)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def process_single_frame(
    frame_bgr: np.ndarray,
    face_mesh: "mp.solutions.face_mesh.FaceMesh",
    selfie_seg: "mp.solutions.selfie_segmentation.SelfieSegmentation",
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """1 フレームから 3 種類のマスクを生成する。

    Returns:
        (neckhead_mask, mouth_mask, alpha_mask) それぞれ (H, W) uint8、
        または顔検出失敗時は None。
    """
    frame_resized = cv2.resize(frame_bgr, (img_size, img_size))
    h, w = img_size, img_size
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # ---- アルファマスク: SelfieSegmentation ----
    seg_result = selfie_seg.process(frame_rgb)
    if seg_result.segmentation_mask is None:
        alpha = np.zeros((h, w), dtype=np.uint8)
    else:
        seg = seg_result.segmentation_mask  # float [0, 1]
        alpha = (seg * 255).clip(0, 255).astype(np.uint8)
        # 二値化してぼかし (エッジをなめらかに)
        _, alpha_bin = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        alpha = cv2.GaussianBlur(alpha_bin, (7, 7), 0)

    # ---- FaceMesh: 頭部マスク + 口マスク ----
    mesh_result = face_mesh.process(frame_rgb)
    if not mesh_result.multi_face_landmarks:
        return None  # 顔未検出

    face_lm = mesh_result.multi_face_landmarks[0]
    neckhead = _generate_neckhead_mask(face_lm, w, h)
    mouth = _generate_mouth_mask(face_lm, w, h)

    return neckhead, mouth, alpha


def main() -> None:
    p = argparse.ArgumentParser(
        description="MediaPipe を使った FlashAvatar 学習用マスク生成"
    )
    p.add_argument(
        "--imgs_dir",
        required=True,
        help="フレーム画像ディレクトリ (imgs/)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="出力ルートディレクトリ (parsing/, alpha/ が作成される)",
    )
    p.add_argument("--img_size", type=int, default=512, help="マスクの解像度")
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="既存マスクをスキップして再開",
    )
    args = p.parse_args()

    imgs_dir = Path(args.imgs_dir)
    out_dir = Path(args.out_dir)
    parsing_dir = out_dir / "parsing"
    alpha_dir = out_dir / "alpha"
    parsing_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    logger.add(str(out_dir / "mask_gen.log"), rotation="10 MB")
    logger.info("=== generate_masks_mediapipe.py ===")
    logger.info("imgs_dir : {}", imgs_dir)
    logger.info("out_dir  : {}", out_dir)

    # フレームファイルのソート
    frame_files = sorted(imgs_dir.glob("*.jpg")) + sorted(imgs_dir.glob("*.png"))
    if not frame_files:
        logger.error("フレームが見つかりません: {}", imgs_dir)
        sys.exit(1)
    logger.info("対象フレーム数: {}", len(frame_files))

    # MediaPipe の初期化
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=1  # landscape model (より精度高)
    )

    failed: list[str] = []
    carry_prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for idx, fpath in enumerate(frame_files):
        stem = fpath.stem  # e.g. "00001"

        # スキップ確認
        neckhead_path = parsing_dir / f"{stem}_neckhead.png"
        mouth_path = parsing_dir / f"{stem}_mouth.png"
        alpha_path = alpha_dir / f"{stem}.jpg"
        if args.skip_existing and neckhead_path.exists() and mouth_path.exists() and alpha_path.exists():
            continue

        frame_bgr = cv2.imread(str(fpath))
        if frame_bgr is None:
            logger.warning("読み込み失敗: {}", fpath)
            failed.append(stem)
            continue

        result = process_single_frame(frame_bgr, face_mesh, selfie_seg, args.img_size)

        if result is None:
            # 顔未検出 → キャリーフォワード
            if carry_prev is not None:
                result = carry_prev
                logger.warning("frame {}: 顔未検出 → 直前マスクで補完", stem)
            else:
                # 最初のフレームで失敗: 全白マスクをフォールバック
                empty = np.zeros((args.img_size, args.img_size), dtype=np.uint8)
                result = (empty.copy(), empty.copy(), empty.copy())
                logger.error("frame {}: 顔未検出・フォールバックとして空マスクを使用", stem)
                failed.append(stem)

        neckhead, mouth, alpha = result
        carry_prev = result

        # 保存: parsing は PNG (二値), alpha は JPG (グレースケール)
        cv2.imwrite(str(neckhead_path), neckhead)
        cv2.imwrite(str(mouth_path), mouth)
        cv2.imwrite(str(alpha_path), alpha)

        if (idx + 1) % 100 == 0:
            logger.info("進捗: {}/{}", idx + 1, len(frame_files))

    face_mesh.close()
    selfie_seg.close()

    logger.info(
        "=== 完了: {} フレーム処理, {} フレーム失敗 ===",
        len(frame_files),
        len(failed),
    )
    if failed:
        logger.warning("失敗フレーム (最大 10 件表示): {}", failed[:10])
    logger.info("parsing/ : {}", parsing_dir)
    logger.info("alpha/   : {}", alpha_dir)


if __name__ == "__main__":
    main()
