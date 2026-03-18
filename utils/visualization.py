"""可視化ユーティリティ

3DMM パラメータやパイプラインの中間結果を可視化する。
バウンディングボックスの描画、パラメータのプロット等。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """フレームにバウンディングボックスを描画する。

    Args:
        frame: BGR 画像 (H, W, 3)。コピーして描画する。
        bbox: (x1, y1, x2, y2)。
        color: BGR 色。
        thickness: 線の太さ。
        label: ボックス上部に表示するテキスト。

    Returns:
        描画済み画像。
    """
    vis = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            vis, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
        )
    return vis


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 2,
) -> np.ndarray:
    """2D ランドマークを描画する。

    Args:
        landmarks: (N, 2) 座標配列。
    """
    vis = frame.copy()
    for x, y in landmarks.astype(int):
        cv2.circle(vis, (x, y), radius, color, -1)
    return vis


def side_by_side(
    images: List[np.ndarray],
    target_height: int = 256,
) -> np.ndarray:
    """複数画像を横に並べて 1 枚にする。"""
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_height)))
    return np.hstack(resized)


def overlay_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """FPS テキストをフレームに重畳する。"""
    vis = frame.copy()
    cv2.putText(
        vis, f"FPS: {fps:.1f}", position,
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
    )
    return vis