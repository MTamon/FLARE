"""Converter chain ランナー。

抽出済みパラメータ (extractor 出力) を converter_chain で順に変換するための
共通ヘルパー。``use_mediapipe_supplement=True`` を持つ Adapter が含まれていれば、
MediaPipe Face Landmarker から推定した ``eyes_pose`` / ``eyelids`` を
``source_params`` に注入してから ``adapter.convert()`` を呼び出す。

このモジュールが解決する問題:
    - DECA / SMIRK は ``eyes_pose`` をネイティブ出力しない。
    - DECA / SMIRK は ``eyelids`` をネイティブ出力しない (SMIRK はネイティブで持つ)。
    - これらを MediaPipe で外部推定して FlashAvatar に渡したい場合、
      Adapter 側のフラグだけでは不十分で、呼び出し側が実際に MediaPipe を実行して
      ``source_params`` に値を入れてあげる必要がある。本モジュールがその橋渡しを担う。

設計方針:
    - 顔検出は呼び出し側で実施済み (``bbox`` を渡す) を前提とする。
      Extractor がクロップ済み画像を期待するのと同じ理由で、検出結果は使い回したい。
    - MediaPipe の起動コストは無視できないため、フラグが ``False`` のときは
      Face Landmarker を一切呼び出さない。
    - 単一の Adapter 用に書かず、チェーンの先頭 Adapter のフラグだけを見る
      設計とする (チェーン途中で source_params が形を変えるため、注入は先頭でしか
      意味を持たない)。

Example:
    converter_chain を使った変換::

        from flare.converters.registry import AdapterRegistry
        from flare.converters.deca_to_flame import DECAToFlameAdapter
        from flare.utils.face_detect import FaceDetector
        from flare.pipeline.converter_runner import run_converter_chain

        registry = AdapterRegistry()
        registry.register(DECAToFlameAdapter())
        chain = registry.build_chain([
            {"type": "deca_to_flash_avatar", "use_mediapipe_supplement": True},
        ])

        face_detector = FaceDetector()
        bbox = face_detector.detect(frame)
        deca_params = extractor.extract(cropped_tensor)

        flash_params = run_converter_chain(
            chain,
            deca_params,
            frame=frame,
            bbox=bbox,
            face_detector=face_detector,
        )
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from flare.converters.base import BaseAdapter


def run_converter_chain(
    chain: list[BaseAdapter],
    source_params: dict[str, torch.Tensor],
    *,
    frame: Optional[np.ndarray] = None,
    bbox: Optional[tuple[int, int, int, int]] = None,
    face_detector: Optional[Any] = None,
) -> dict[str, torch.Tensor]:
    """converter_chain を順に適用してパラメータを変換する。

    チェーンの先頭 Adapter が ``use_mediapipe_supplement=True`` を持ち、かつ
    ``frame`` / ``bbox`` / ``face_detector`` が全て与えられている場合、
    MediaPipe Face Landmarker から推定した ``eyes_pose`` / ``eyelids`` を
    ``source_params`` に注入してから変換を実行する。

    Args:
        chain: ``AdapterRegistry.build_chain()`` から得た Adapter のリスト。
            空リストの場合は ``source_params`` をそのまま返す。
        source_params: Extractor の出力パラメータ辞書。in-place 改変はせず、
            必要な場合のみ shallow copy を作って注入する。
        frame: BGR 画像 ``(H, W, 3)`` uint8。MediaPipe 補完を行う場合のみ必須。
        bbox: ``face_detector.detect()`` から得た顔バウンディングボックス
            ``(x, y, w, h)``。MediaPipe 補完を行う場合のみ必須。
        face_detector: ``flare.utils.face_detect.FaceDetector`` インスタンス。
            ``detect_eye_pose(frame, bbox)`` メソッドを持つ必要がある。
            MediaPipe 補完を行う場合のみ必須。

    Returns:
        チェーン適用後のパラメータ辞書。

    Raises:
        ValueError: ``use_mediapipe_supplement=True`` の Adapter が含まれているのに
            ``frame`` / ``bbox`` / ``face_detector`` のいずれかが ``None`` の場合。
    """
    if not chain:
        return source_params

    head = chain[0]
    needs_supplement = bool(getattr(head, "use_mediapipe_supplement", False))

    if needs_supplement:
        if frame is None or bbox is None or face_detector is None:
            raise ValueError(
                "use_mediapipe_supplement=True requires frame, bbox, and "
                "face_detector to be provided to run_converter_chain()."
            )
        eyes_pose, eyelids = face_detector.detect_eye_pose(frame, bbox)
        source_params = dict(source_params)
        source_params.setdefault("eyes_pose", eyes_pose)
        source_params.setdefault("eyelids", eyelids)

    params = source_params
    for adapter in chain:
        params = adapter.convert(params)
    return params
