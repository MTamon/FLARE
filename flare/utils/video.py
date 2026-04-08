"""動画・フレームの入出力ユーティリティモジュール。

OpenCVベースの動画読み書きと、NumPy形式でのフレーム保存・読み込みを提供する。
リアルタイムモード（Webカメラ入力）とバッチモード（ファイル入力）の両方で使用される。

Example:
    動画ファイルからのフレーム読み込み::

        with VideoReader("input.mp4") as reader:
            print(f"FPS: {reader.get_fps()}, Total: {reader.get_total_frames()}")
            while True:
                frame = reader.read_frame()
                if frame is None:
                    break
                process(frame)

    動画ファイルへのフレーム書き込み::

        with VideoWriter("output.mp4", fps=30.0, width=512, height=512) as writer:
            for frame in frames:
                writer.write_frame(frame)

    フレームのnpz保存・読み込み::

        save_frames_to_npz(frame_list, "frames.npz")
        loaded = load_frames_from_npz("frames.npz")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


class VideoReader:
    """OpenCV VideoCapture のラッパークラス。

    動画ファイルまたはWebカメラデバイスからフレームを読み込む。
    コンテキストマネージャプロトコルに対応し、withブロックで安全にリソースを
    解放できる。

    Attributes:
        _cap: OpenCVのVideoCaptureオブジェクト。
        _source: 動画ファイルパスまたはデバイスインデックス。
    """

    def __init__(self, source: Union[str, int]) -> None:
        """VideoReaderを初期化する。

        Args:
            source: 動画ファイルのパス（文字列）またはWebカメラのデバイス
                インデックス（整数、例: 0）。

        Raises:
            FileNotFoundError: 文字列パスが指定され、ファイルが存在しない場合。
            RuntimeError: VideoCaptureのオープンに失敗した場合。
        """
        self._source = source
        if isinstance(source, str) and not Path(source).exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

    def read_frame(self) -> Optional[np.ndarray]:
        """次のフレームを読み込む。

        Returns:
            BGR形式のフレーム画像。形状は ``(H, W, 3)``、dtype は ``uint8``。
            動画の末尾に達した場合、またはフレーム取得に失敗した場合は ``None``。
        """
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def get_fps(self) -> float:
        """動画のフレームレートを返す。

        Returns:
            フレームレート（FPS）。Webカメラの場合はデバイス報告値。
        """
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def get_total_frames(self) -> int:
        """動画の総フレーム数を返す。

        Returns:
            総フレーム数。Webカメラの場合は0または不正確な値の可能性がある。
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self) -> None:
        """VideoCaptureリソースを解放する。

        複数回呼び出しても安全。
        """
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> VideoReader:
        """コンテキストマネージャのエントリ。

        Returns:
            自身のVideoReaderインスタンス。
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """コンテキストマネージャのイグジット。リソースを解放する。

        Args:
            exc_type: 例外の型。例外が発生していない場合はNone。
            exc_val: 例外インスタンス。例外が発生していない場合はNone。
            exc_tb: トレースバック。例外が発生していない場合はNone。
        """
        self.release()


class VideoWriter:
    """OpenCV VideoWriter のラッパークラス。

    動画ファイルへフレームを書き込む。コンテキストマネージャプロトコルに対応し、
    withブロックで安全にリソースを解放できる。

    Attributes:
        _writer: OpenCVのVideoWriterオブジェクト。
        _path: 出力動画ファイルのパス。
    """

    def __init__(
        self,
        path: Union[str, Path],
        fps: float = 30.0,
        width: int = 512,
        height: int = 512,
        codec: str = "mp4v",
    ) -> None:
        """VideoWriterを初期化する。

        Args:
            path: 出力動画ファイルのパス。
            fps: 出力動画のフレームレート。
            width: 出力フレームの幅（ピクセル）。
            height: 出力フレームの高さ（ピクセル）。
            codec: FourCC コーデック文字列。デフォルトは ``"mp4v"``。

        Raises:
            RuntimeError: VideoWriterの初期化に失敗した場合。
        """
        self._path = str(path)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(self._path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self._path}")

    def write_frame(self, frame: np.ndarray) -> None:
        """フレームを書き込む。

        Args:
            frame: BGR形式のフレーム画像。形状は ``(H, W, 3)``、
                dtype は ``uint8``。

        Raises:
            RuntimeError: VideoWriterが既に解放されている場合。
        """
        self._writer.write(frame)

    def release(self) -> None:
        """VideoWriterリソースを解放し、ファイルをフラッシュする。

        複数回呼び出しても安全。
        """
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> VideoWriter:
        """コンテキストマネージャのエントリ。

        Returns:
            自身のVideoWriterインスタンス。
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """コンテキストマネージャのイグジット。リソースを解放する。

        Args:
            exc_type: 例外の型。例外が発生していない場合はNone。
            exc_val: 例外インスタンス。例外が発生していない場合はNone。
            exc_tb: トレースバック。例外が発生していない場合はNone。
        """
        self.release()


def save_frames_to_npz(
    frames: list[np.ndarray], path: Union[str, Path]
) -> None:
    """フレームリストをnpz形式で保存する。

    フレームリストを1つのNumPy配列にスタックし、圧縮npz形式で保存する。

    Args:
        frames: フレーム画像のリスト。各要素は ``(H, W, 3)`` のndarray。
            全フレームが同一形状であること。
        path: 保存先ファイルパス。拡張子 ``.npz`` を推奨。
            親ディレクトリが存在しない場合は自動作成される。

    Raises:
        ValueError: framesが空リストの場合。
    """
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(frames, axis=0)
    np.savez_compressed(str(save_path), frames=stacked)


def load_frames_from_npz(path: Union[str, Path]) -> np.ndarray:
    """npzファイルからフレーム配列を読み込む。

    ``save_frames_to_npz`` で保存されたファイルを読み込み、フレーム配列を返す。

    Args:
        path: npzファイルのパス。

    Returns:
        フレーム配列。形状は ``(T, H, W, 3)``、dtype は保存時と同一。
        Tはフレーム数。

    Raises:
        FileNotFoundError: 指定パスにファイルが存在しない場合。
        KeyError: npzファイルに ``"frames"`` キーが存在しない場合。
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {load_path}")
    data = np.load(str(load_path))
    if "frames" not in data:
        raise KeyError(
            f"Key 'frames' not found in {load_path}. Available keys: {list(data.keys())}"
        )
    return data["frames"]
