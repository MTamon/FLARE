"""FLARE renderers package.

3DMMパラメータからフォトリアルな顔画像を生成するレンダラーモジュール群。
ルートA（BFM: PIRender）およびルートB（FLAME: FlashAvatar / HeadGaS）を提供する。

全Rendererは :class:`BaseRenderer` ABCを継承し、setup/render分離パターンで
統一的なパイプライン制御を実現する。
"""

from flare.renderers.base import BaseRenderer as BaseRenderer

__all__: list[str] = ["BaseRenderer"]
