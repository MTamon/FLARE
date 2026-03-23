"""FLARE model interface package.

LHGモデル（Listening Head Generation）の抽象インターフェースを提供する。
音声特徴量と話者動作からリスナーの頭部動作を予測するモデルは、
全て :class:`BaseLHGModel` ABCを継承して統一インターフェースを実装する。
"""

from flare.model_interface.base import BaseLHGModel as BaseLHGModel

__all__: list[str] = ["BaseLHGModel"]
