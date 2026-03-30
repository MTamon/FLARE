"""FLARE converters package.

パラメータ空間の変換を担うモジュール群。Extractor出力形式とRenderer入力形式の
間の変換（例: DECA→FLAME）、およびLHGモデル出力からRenderer入力への変換を行う。

AdapterRegistryパターンにより、config.yamlのextractor.typeとrenderer.typeの
組み合わせから適切なAdapterを自動選択する。

モジュール構成:
    - ``base.py``: BaseAdapter ABC
    - ``registry.py``: AdapterRegistry（自動選択）
    - ``deca_to_flame.py``: DECA→FLAME変換（ゼロパディング）
    - ``flame_to_pirender.py``: FLAME→PIRender変換
    - ``identity.py``: IdentityAdapter（変換不要時のパススルー）
"""

from __future__ import annotations

from typing import List

__all__: List[str] = ["BaseAdapter", "AdapterRegistry", "DECAToFlameAdapter", "FlameToPIRenderAdapter", "IdentityAdapter",]
