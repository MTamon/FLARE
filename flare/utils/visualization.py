"""可視化ユーティリティモジュール。

パイプラインの中間結果やデバッグ情報の可視化機能を提供する。
バウンディングボックスの描画、パラメータのオーバーレイ表示等に使用する。

Example:
    フレームへのバウンディングボックス描画::

        from flare.utils.visualization import draw_bbox, show_frame

        annotated = draw_bbox(frame, bbox=(100, 80, 300, 320))
        show_frame(annotated, window_name="Detection")
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def draw_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """フレームにバウンディングボックスを描画する。

    入力フレームのコピーに描画し、元のフレームは変更しない。

    Args:
        frame: BGR形式の入力画像。形状は ``(H, W, 3)``、dtype は ``uint8``。
        bbox: バウンディングボックス ``(x1, y1, x2, y2)``。ピクセル座標。
        color: 描画色 ``(B, G, R)``。デフォルトは緑。
        thickness: 線の太さ（ピクセル）。
        label: ボックスの左上に表示するラベル文字列。Noneの場合は表示しない。

    Returns:
        バウンディングボックスが描画されたフレームのコピー。
        形状とdtypeは入力と同一。
    """
    vis = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(
            vis,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            vis,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )
    return vis


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: tuple[int, int] = (10, 30),
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """フレームにFPS値を描画する。

    入力フレームのコピーに描画し、元のフレームは変更しない。

    Args:
        frame: BGR形式の入力画像。形状は ``(H, W, 3)``、dtype は ``uint8``。
        fps: 表示するFPS値。
        position: テキストの表示位置 ``(x, y)``。ピクセル座標。
        color: テキストの色 ``(B, G, R)``。デフォルトは緑。

    Returns:
        FPS値が描画されたフレームのコピー。
    """
    vis = frame.copy()
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        vis,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return vis


def show_frame(
    frame: np.ndarray,
    window_name: str = "FLARE",
    wait_ms: int = 1,
) -> int:
    """フレームをOpenCVウィンドウに表示する。

    Args:
        frame: BGR形式の表示画像。形状は ``(H, W, 3)``、dtype は ``uint8``。
        window_name: ウィンドウ名。
        wait_ms: cv2.waitKeyのタイムアウト（ミリ秒）。
            0を指定するとキー入力まで待機する。

    Returns:
        押されたキーのASCIIコード。タイムアウト時は ``-1``。
    """
    cv2.imshow(window_name, frame)
    key: int = cv2.waitKey(wait_ms)
    return key


def create_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    border_width: int = 2,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """2枚の画像を横に並べた比較画像を生成する。

    右画像を左画像と同じ高さにリサイズしてから結合する。

    Args:
        left: 左側の画像。形状は ``(H, W, 3)``。
        right: 右側の画像。形状は ``(H2, W2, 3)``。
        border_width: 画像間の境界線の幅（ピクセル）。
        border_color: 境界線の色 ``(B, G, R)``。

    Returns:
        横結合された画像。形状は ``(H, W_left + border_width + W_right_scaled, 3)``。
    """
    h_left = left.shape[0]
    h_right, w_right = right.shape[:2]
    scale = h_left / h_right
    new_w = int(w_right * scale)
    right_resized: np.ndarray = cv2.resize(
        right, (new_w, h_left), interpolation=cv2.INTER_LINEAR
    )

    border = np.full(
        (h_left, border_width, 3), border_color, dtype=np.uint8
    )

    combined: np.ndarray = np.concatenate(
        [left, border, right_resized], axis=1
    )
    return combined
