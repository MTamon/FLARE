"""動画 I/O ユーティリティ

Webcam キャプチャ（リアルタイムモード）と動画ファイル読み書き（バッチモード）
の基本操作を提供する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from loguru import logger


def open_video(
    source: Union[str, Path, int],
) -> cv2.VideoCapture:
    """動画ファイルまたはカメラデバイスを開く。

    Args:
        source: ファイルパス or カメラデバイス番号 (0, 1, …)。

    Returns:
        オープン済みの cv2.VideoCapture。

    Raises:
        IOError: オープンに失敗した場合。
    """
    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")
    return cap


def read_frames(
    source: Union[str, Path, int],
    max_frames: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """動画からフレームを順に yield する。

    Args:
        source: ファイルパスまたはカメラデバイス番号。
        max_frames: 読み取りフレーム数上限。None で全フレーム。

    Yields:
        BGR 画像 (H, W, 3) as np.ndarray (uint8)。
    """
    cap = open_video(source)
    count = 0
    try:
        while True:
            if max_frames is not None and count >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            count += 1
    finally:
        cap.release()
    logger.debug(f"Read {count} frames from {source}")


def get_video_info(
    source: Union[str, Path, int],
) -> dict:
    """動画のメタ情報を返す。"""
    cap = open_video(source)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def frame_to_tensor(
    frame: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """BGR numpy 画像を (1, C, H, W) float32 Tensor に変換する。

    Args:
        frame: (H, W, 3) uint8 BGR。
        size: (H, W) にリサイズ。None ならそのまま。
        device: 配置先デバイス。

    Returns:
        (1, 3, H, W) float32 Tensor, 値域 [0, 1]。
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]))  # cv2.resize は (W, H)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)


def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) float Tensor → BGR numpy uint8 画像。"""
    img = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class VideoWriter:
    """cv2.VideoWriter のラッパー。コンテキストマネージャ対応。"""

    def __init__(
        self,
        path: Union[str, Path],
        fps: float = 30.0,
        size: Tuple[int, int] = (512, 512),
        codec: str = "mp4v",
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # cv2.VideoWriter は (W, H)
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (size[1], size[0]))
        self._count = 0

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._count += 1

    def close(self) -> None:
        self._writer.release()
        logger.info(f"Wrote {self._count} frames to {self._path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()