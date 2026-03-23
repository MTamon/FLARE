"""可視化ユーティリティ。

パイプラインの処理結果やデバッグ情報を画像上にオーバーレイする
描画関数群を提供する。全関数は元の画像を変更せずコピーを返す。

Example:
    >>> from flare.utils.visualization import draw_bbox, draw_fps_overlay, side_by_side
    >>> vis = draw_bbox(frame, bbox=(100, 50, 300, 250))
    >>> vis = draw_fps_overlay(vis, fps=28.5)
    >>> combined = side_by_side(original, rendered)
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """バウンディングボックスを矩形描画する。

    元の画像は変更せず、描画済みのコピーを返す。

    Args:
        image: 入力画像（BGR形式、shape: (H, W, 3)）。
        bbox: バウンディングボックス ``(x1, y1, x2, y2)``。
        color: 矩形の色（BGR）。デフォルト緑。
        thickness: 線の太さ（ピクセル）。

    Returns:
        矩形描画済みの画像コピー。
    """
    canvas: np.ndarray = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 2,
) -> np.ndarray:
    """ランドマーク点を円で描画する。

    元の画像は変更せず、描画済みのコピーを返す。

    Args:
        image: 入力画像（BGR形式、shape: (H, W, 3)）。
        landmarks: ランドマーク座標（shape: (N, 2)）。各行は ``(x, y)``。
        color: 点の色（BGR）。デフォルト赤。
        radius: 点の半径（ピクセル）。

    Returns:
        ランドマーク描画済みの画像コピー。
    """
    canvas: np.ndarray = image.copy()

    for i in range(landmarks.shape[0]):
        x: int = int(round(landmarks[i, 0]))
        y: int = int(round(landmarks[i, 1]))
        cv2.circle(canvas, (x, y), radius, color, -1)

    return canvas


def draw_fps_overlay(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """FPS値をテキストオーバーレイする。

    元の画像は変更せず、描画済みのコピーを返す。

    Args:
        image: 入力画像（BGR形式、shape: (H, W, 3)）。
        fps: 表示するFPS値。
        position: テキスト描画位置 ``(x, y)``。デフォルト左上。

    Returns:
        FPSオーバーレイ済みの画像コピー。
    """
    canvas: np.ndarray = image.copy()
    text: str = f"FPS: {fps:.1f}"
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_metrics_overlay(
    image: np.ndarray,
    metrics: Dict[str, float],
) -> np.ndarray:
    """メトリクス辞書を複数行テキストでオーバーレイする。

    ``PipelineMetrics.summary()`` の返り値をそのまま渡すことを想定。
    元の画像は変更せず、描画済みのコピーを返す。

    Args:
        image: 入力画像（BGR形式、shape: (H, W, 3)）。
        metrics: メトリクスキーと数値のDict。

    Returns:
        メトリクスオーバーレイ済みの画像コピー。
    """
    canvas: np.ndarray = image.copy()
    y_offset: int = 25
    line_height: int = 22

    for key, value in metrics.items():
        text: str = f"{key}: {value:.2f}"
        cv2.putText(
            canvas,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += line_height

    return canvas


def side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    gap: int = 4,
    gap_color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """2枚の画像を横並びに結合する。

    高さが異なる場合は短い方の下部にパディングを追加して揃える。

    Args:
        img1: 左側の画像（BGR形式、shape: (H1, W1, 3)）。
        img2: 右側の画像（BGR形式、shape: (H2, W2, 3)）。
        gap: 画像間のギャップ幅（ピクセル）。
        gap_color: ギャップの色（BGR）。

    Returns:
        横並びに結合された画像（shape: (max_H, W1+gap+W2, 3)）。
    """
    h1: int = img1.shape[0]
    h2: int = img2.shape[0]
    w1: int = img1.shape[1]
    w2: int = img2.shape[1]
    max_h: int = max(h1, h2)

    # 高さを揃えるためにパディング
    padded1: np.ndarray = _pad_height(img1, max_h, gap_color)
    padded2: np.ndarray = _pad_height(img2, max_h, gap_color)

    # ギャップ
    gap_strip: np.ndarray = np.full(
        (max_h, gap, 3), gap_color, dtype=np.uint8
    )

    combined: np.ndarray = np.concatenate(
        [padded1, gap_strip, padded2], axis=1
    )
    return combined


def tensor_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """PyTorchテンソルをBGR numpy配列に変換する。

    値域 ``[0, 1]`` のfloatテンソルを ``[0, 255]`` のuint8に変換し、
    RGB→BGR変換を行う。

    Args:
        image_tensor: 画像テンソル。shape ``(3, H, W)`` または ``(1, 3, H, W)``。
            値域 ``[0, 1]``。

    Returns:
        BGR形式のnumpy配列（shape: (H, W, 3)、dtype: uint8）。

    Raises:
        ValueError: テンソルの次元数が3でも4でもない場合。
    """
    import torch

    tensor: torch.Tensor = image_tensor

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    if tensor.ndim != 3:
        raise ValueError(
            f"テンソルは3次元 (3, H, W) または4次元 (1, 3, H, W) である"
            f"必要があります。受け取った次元数: {image_tensor.ndim}"
        )

    # (3, H, W) → (H, W, 3)、GPU→CPU、clamp→uint8
    rgb: np.ndarray = (
        tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).astype(np.uint8)

    bgr: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def _pad_height(
    image: np.ndarray,
    target_height: int,
    pad_color: Tuple[int, int, int],
) -> np.ndarray:
    """画像の下部にパディングを追加して指定の高さに揃える。

    Args:
        image: 入力画像（shape: (H, W, 3)）。
        target_height: 目標の高さ。
        pad_color: パディング色（BGR）。

    Returns:
        パディング済み画像。既に目標高さ以上の場合はそのまま返す。
    """
    h: int = image.shape[0]
    if h >= target_height:
        return image

    w: int = image.shape[1]
    pad_h: int = target_height - h
    padding: np.ndarray = np.full((pad_h, w, 3), pad_color, dtype=np.uint8)
    return np.concatenate([image, padding], axis=0)
