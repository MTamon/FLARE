"""FLAME→PIRender パラメータ変換アダプタ。

FLAMEパラメータ形式をPIRender入力形式（BFM motion descriptor）に変換する。

重要な注意事項:
    FLAMEのexp(50D)とBFMのexp(64D)は**異なるパラメータ空間**である。
    FLAMEはgeneric_model.pklのshapedirs PCA基底を使用し、
    BFMはBasel Face ModelのPCA基底を使用する。
    両者のPCA基底は異なるトポロジー・学習データから生成されており、
    **理論的に正確な変換は不可能**である。

    本アダプタでは実用的な近似変換として、FLAMEのexp(50D)を
    BFM exp(64D)の先頭50次元にコピーし、残り14次元をゼロパディングする
    方式を採用する。この変換は以下の前提に基づく:
    - 主要な表情成分は先頭の主成分に集中する
    - 高次の主成分（51-64）はゼロでも視覚的影響は小さい
    - 正確な変換が必要な場合はルートBを使用すべき

    仕様書の推奨: ルートAとルートBを混在させず、どちらかに統一するのが最も実用的。

Example:
    >>> adapter = FlameToPIRenderAdapter()
    >>> pirender_params = adapter.convert({"exp": flame_exp_50d, "pose": pose_6d})
    >>> pirender_params["exp"].shape  # (B, 64)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from loguru import logger

from flare.converters.base import BaseAdapter
from flare.converters.registry import AdapterRegistry

_registry: AdapterRegistry = AdapterRegistry.get_instance()


@_registry.register
class FlameToPIRenderAdapter(BaseAdapter):
    """FLAME→PIRender（BFM motion descriptor）変換アダプタ。

    FLAMEのexp(50D)をBFM exp(64D)に近似変換し、poseはそのまま渡す。

    変換内容:
        - exp: FLAME 50D → BFM 64D（先頭50Dコピー + 残り14Dゼロパディング）
        - pose: FLAME 6D → PIRender 6D（そのまま渡す）

    警告:
        FLAMEとBFMのexpression空間は厳密には互換性がない。
        本変換は近似であり、正確なレンダリングが必要な場合は
        ルートBの使用を推奨する。

    Example:
        >>> adapter = FlameToPIRenderAdapter()
        >>> result = adapter.convert({"exp": flame_exp, "pose": pose})
        >>> result["exp"].shape  # (B, 64)
    """

    #: BFM expression パラメータの次元数
    _BFM_EXP_DIM: int = 64

    #: FLAME expression パラメータの次元数
    _FLAME_EXP_DIM: int = 50

    def __init__(self) -> None:
        """FlameToPIRenderAdapterを初期化する。"""
        logger.debug(
            "FlameToPIRenderAdapter初期化: "
            "FLAME exp({}D) → BFM exp({}D) 近似変換",
            self._FLAME_EXP_DIM,
            self._BFM_EXP_DIM,
        )

    @property
    def source_format(self) -> str:
        """変換元フォーマット名。

        Returns:
            ``"flame"``。
        """
        return "flame"

    @property
    def target_format(self) -> str:
        """変換先フォーマット名。

        Returns:
            ``"pirender"``。
        """
        return "pirender"

    def convert(
        self, source_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """FLAMEパラメータをPIRender入力形式に変換する。

        FLAMEのexp(50D)をBFM exp(64D)に近似変換する。先頭50次元に
        FLAMEのexp値をコピーし、残り14次元をゼロパディングする。
        poseは形式が同一（回転3D + 平行移動/jaw 3D）のためそのまま渡す。

        注意:
            この変換は**近似**であり、FLAMEとBFMのPCA基底の違いにより
            レンダリング品質が低下する可能性がある。仕様書では
            「両者のパラメータ空間は直接互換性がないため、混在させないこと」
            と注記されている（3.3節）。

        Args:
            source_params: FLAME出力Dict。
                必須キー: ``"exp"`` (B, 50D), ``"pose"`` (B, 6D)。
                オプション: その他のFLAMEパラメータ（無視される）。

        Returns:
            PIRender入力Dict:
                - ``"exp"``: (B, 64D) — BFM近似expression
                - ``"pose"``: (B, 6D) — 頭部姿勢
        """
        flame_exp: torch.Tensor = source_params["exp"]  # (B, 50)
        pose: torch.Tensor = source_params["pose"]  # (B, 6)

        # FLAME exp(50D) → BFM exp(64D): ゼロパディング近似
        pad_size: int = self._BFM_EXP_DIM - flame_exp.shape[-1]

        if pad_size > 0:
            bfm_exp: torch.Tensor = F.pad(
                flame_exp, (0, pad_size), value=0.0
            )  # (B, 64)
        elif pad_size == 0:
            bfm_exp = flame_exp
        else:
            # FLAME expがBFM expより大きい場合（通常は発生しない）
            bfm_exp = flame_exp[:, : self._BFM_EXP_DIM]

        return {
            "exp": bfm_exp,  # (B, 64)
            "pose": pose,    # (B, 6)
        }
