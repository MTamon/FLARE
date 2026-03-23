"""FLARE extractors package.

3DMMパラメータ抽出モジュール群。ルートA（BFM: Deep3DFaceRecon）および
ルートB（FLAME: DECA / SMIRK）の特徴量抽出器を提供する。

全Extractorは :class:`BaseExtractor` ABCを継承し、統一インターフェースで
画像からの3DMMパラメータ抽出を行う。
"""

from flare.extractors.base import BaseExtractor as BaseExtractor

__all__: list[str] = ["BaseExtractor"]
