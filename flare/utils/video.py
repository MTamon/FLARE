"""動画I/Oユーティリティ。

cv2.VideoCapture / cv2.VideoWriter をラップし、コンテキストマネージャと
イテレータプロトコルを提供する。リアルタイムモード（Webcam入力）と
バッチモード（動画ファイル入力/出力）の両方に対応する。

Example:
    >>> with VideoReader("input.mp4") as reader:
    ...     for frame in reader:
    ...         process(frame)
    >>>
    >>> with VideoWriter("output.mp4", fps=30.0, frame_size=(512, 512)) as writer:
    ...     writer.write_frame(rendered_frame)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger


class VideoReader:
    """cv2.VideoCaptureのラッパークラス。

    動画ファイルまたはWebカメラからフレームを読み込む。
    コンテキストマネージャおよびイテレータプロトコルをサポートする。

    Attributes:
        _cap: OpenCVのVideoCaptureインスタンス。
        _source: 入力ソース（ファイルパスまたはカメラID）。

    Example:
        >>> # ファイルから読み込み
        >>> with VideoReader("video.mp4") as reader:
        ...     print(f"FPS: {reader.get_fps()}, Frames: {reader.get_total_frames()}")
        ...     for frame in reader:
        ...         cv2.imshow("frame", frame)
        >>>
        >>> # Webcamから読み込み
        >>> with VideoReader(0, width=640, height=480) as reader:
        ...     frame = reader.read_frame()
    """

    def __init__(
        self,
        source: Union[str, int],
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """VideoReaderを初期化する。

        Args:
            source: 動画ファイルパス（str）またはカメラデバイスID（int）。
            width: フレームの横幅を指定。Noneの場合は元の解像度を使用。
            height: フレームの縦幅を指定。Noneの場合は元の解像度を使用。

        Raises:
            FileNotFoundError: ファイルパスが指定されたが存在しない場合。
            RuntimeError: VideoCaptureのオープンに失敗した場合。
        """
        self._source: Union[str, int] = source

        if isinstance(source, str) and not Path(source).exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {source}")

        self._cap: cv2.VideoCapture = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"動画ソースのオープンに失敗しました: {source}"
            )

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

        logger.debug(
            "VideoReader初期化: source={} | size={}",
            source,
            self.get_frame_size(),
        )

    def read_frame(self) -> Optional[np.ndarray]:
        """1フレームを読み込む。

        Returns:
            BGR形式の画像ndarray（shape: (H, W, 3), dtype: uint8）。
            動画終端または読み込み失敗時は ``None``。
        """
        ret: bool
        frame: np.ndarray
        ret, frame = self._cap.read()

        if not ret:
            return None
        return frame

    def get_fps(self) -> float:
        """動画のFPSを返す。

        Returns:
            フレームレート（fps）。Webcamの場合はデバイス報告値。
        """
        fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0.0 else 30.0

    def get_total_frames(self) -> int:
        """動画の総フレーム数を返す。

        Returns:
            ファイルの場合は総フレーム数。
            Webcam（int source）の場合は ``sys.maxsize``。
        """
        if isinstance(self._source, int):
            return sys.maxsize

        total: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total if total > 0 else sys.maxsize

    def get_frame_size(self) -> Tuple[int, int]:
        """フレームサイズを返す。

        Returns:
            ``(width, height)`` のタプル。
        """
        width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def release(self) -> None:
        """VideoCaptureリソースを解放する。"""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.debug("VideoReader解放: source={}", self._source)

    def __enter__(self) -> VideoReader:
        """コンテキストマネージャのエントリ。

        Returns:
            自身のインスタンス。
        """
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """コンテキストマネージャの終了処理。リソースを解放する。

        Args:
            exc_type: 例外型（あれば）。
            exc_val: 例外値（あれば）。
            exc_tb: トレースバック（あれば）。
        """
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        """フレームを順次返すイテレータ。

        Yields:
            BGR形式の画像ndarray。動画終端で停止。
        """
        while True:
            frame: Optional[np.ndarray] = self.read_frame()
            if frame is None:
                break
            yield frame


class VideoWriter:
    """cv2.VideoWriterのラッパークラス。

    レンダリング結果を動画ファイルに書き出す。
    コンテキストマネージャをサポートする。

    Example:
        >>> with VideoWriter("output.mp4", fps=30.0, frame_size=(512, 512)) as writer:
        ...     writer.write_frame(frame)
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        *,
        codec: str = "mp4v",
    ) -> None:
        """VideoWriterを初期化する。

        Args:
            output_path: 出力動画ファイルパス。
            fps: 出力フレームレート。
            frame_size: フレームサイズ ``(width, height)``。
            codec: FourCCコーデック文字列。デフォルトは ``"mp4v"``。

        Raises:
            RuntimeError: VideoWriterのオープンに失敗した場合。
        """
        self._output_path: str = output_path
        self._fps: float = fps
        self._frame_size: Tuple[int, int] = frame_size
        self._frame_count: int = 0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc: int = cv2.VideoWriter_fourcc(*codec)
        self._writer: cv2.VideoWriter = cv2.VideoWriter(
            output_path, fourcc, fps, frame_size
        )

        if not self._writer.isOpened():
            raise RuntimeError(
                f"VideoWriterのオープンに失敗しました: {output_path}"
            )

        logger.debug(
            "VideoWriter初期化: path={} | fps={} | size={} | codec={}",
            output_path,
            fps,
            frame_size,
            codec,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """1フレームを書き込む。

        Args:
            frame: BGR形式の画像ndarray（shape: (H, W, 3), dtype: uint8）。
        """
        self._writer.write(frame)
        self._frame_count += 1

    def release(self) -> None:
        """VideoWriterリソースを解放する。"""
        if self._writer is not None and self._writer.isOpened():
            self._writer.release()
            logger.debug(
                "VideoWriter解放: path={} | frames={}",
                self._output_path,
                self._frame_count,
            )

    def __enter__(self) -> VideoWriter:
        """コンテキストマネージャのエントリ。

        Returns:
            自身のインスタンス。
        """
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """コンテキストマネージャの終了処理。リソースを解放する。

        Args:
            exc_type: 例外型（あれば）。
            exc_val: 例外値（あれば）。
            exc_tb: トレースバック（あれば）。
        """
        self.release()
