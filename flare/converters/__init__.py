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

__all__: List[str] = []

# --- BaseAdapter ---
try:
    from flare.converters.base import BaseAdapter as BaseAdapter

    __all__.append("BaseAdapter")
except ImportError:
    pass

# --- AdapterRegistry ---
try:
    from flare.converters.registry import AdapterRegistry as AdapterRegistry

    __all__.append("AdapterRegistry")
except ImportError:
    pass

# --- Concrete adapters ---
try:
    from flare.converters.deca_to_flame import DECAToFlameAdapter as DECAToFlameAdapter

    __all__.append("DECAToFlameAdapter")
except ImportError:
    pass

try:
    from flare.converters.flame_to_pirender import FlameToPIRenderAdapter as FlameToPIRenderAdapter

    __all__.append("FlameToPIRenderAdapter")
except ImportError:
    pass

try:
    from flare.converters.identity import IdentityAdapter as IdentityAdapter

    __all__.append("IdentityAdapter")
except ImportError:
    pass
