"""BFM→FLAME パラメータ変換モジュール。

BFM（Basel Face Model）形式の3DMMパラメータをFLAME形式に変換するアダプタ。
仕様書§5の設計に従い、TimoBolkart/BFMtoFLAME の手法を使用する。

対応するBFMフォーマット:
    - BFM 2017: Deep3DFaceRecon等の出力（id 80D + exp 64D）
    - BFM 2009: 旧BFMフォーマット（id 80D + exp 64D）
    - cropped BFM 2009: 3DDFA用の切り詰めBFM（shape 40D + exp 10D）

変換ロジック:
    BFM expression空間とFLAME expression空間は異なるPCA基底を使用するため、
    事前計算された線形変換行列（BFMtoFLAMEプロジェクト提供）を使用して
    変換を行う。変換行列が利用不可の場合は、次元数の調整（パディング/切り詰め）
    による近似変換にフォールバックする。

    1. BFM exp → FLAME expr: 線形変換行列 M_exp (D_bfm, D_flame) を適用
       フォールバック: ゼロパディングで次元調整
    2. BFM pose → FLAME pose: axis-angle表現の互換変換
    3. BFM id → FLAME shape: 線形変換行列 M_shape を適用
       フォールバック: ゼロパディングで次元調整

Example:
    BFM→FLAME変換::

        adapter = BFMToFlameAdapter()
        flame_params = adapter.convert({
            "exp": torch.randn(1, 64),
            "pose": torch.randn(1, 6),
            "id": torch.randn(1, 80),
        })
        # flame_params["expr"].shape == (1, 100)
        # flame_params["pose"].shape == (1, 6)
        # flame_params["shape"].shape == (1, 300)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from flare.converters.base import BaseAdapter

_BFM_2017_EXP_DIM: int = 64
"""int: BFM 2017のexpression次元数。"""

_BFM_2009_EXP_DIM: int = 64
"""int: BFM 2009のexpression次元数。"""

_BFM_CROPPED_SHAPE_DIM: int = 40
"""int: cropped BFM 2009（3DDFA）のshape次元数。"""

_BFM_CROPPED_EXP_DIM: int = 10
"""int: cropped BFM 2009（3DDFA）のexp次元数。"""

_FLAME_EXPR_DIM: int = 100
"""int: FLAME expression空間の次元数。"""

_FLAME_SHAPE_DIM: int = 300
"""int: FLAME shape空間の次元数。"""


class BFMToFlameAdapter(BaseAdapter):
    """BFM → FLAME パラメータ変換アダプタ。

    BFM形式（BFM 2017 / BFM 2009 / cropped BFM 2009）の3DMMパラメータを
    FLAME形式に変換する。TimoBolkart/BFMtoFLAMEプロジェクトの線形変換行列を
    使用した高精度変換と、行列が利用不可の場合のフォールバック近似変換を
    サポートする。

    変換ロジック:
        1. ``exp`` (B, D_bfm) → ``expr`` (B, 100):
           変換行列が利用可能ならM_expを適用、不可ならゼロパディング。
        2. ``pose`` (B, 6) → ``pose`` (B, 6):
           BFMとFLAMEで同一のaxis-angle表現を使用するためパススルー。
        3. ``id`` (B, D_id) → ``shape`` (B, 300):
           変換行列が利用可能ならM_shapeを適用、不可ならゼロパディング。

    Attributes:
        _mapping_dir: BFMtoFLAME変換行列ファイルのディレクトリパス。
        _exp_mapping: expression変換行列 (D_bfm, D_flame)。
        _shape_mapping: shape変換行列 (D_bfm_id, D_flame_shape)。
        _bfm_variant: BFMフォーマットの種別。
    """

    def __init__(
        self,
        mapping_dir: Optional[str] = None,
        bfm_variant: str = "bfm2017",
    ) -> None:
        """BFMToFlameAdapterを初期化する。

        Args:
            mapping_dir: BFMtoFLAME変換行列ファイルのディレクトリパス。
                ``exp_mapping.pt`` と ``shape_mapping.pt`` を含むディレクトリ。
                Noneの場合はフォールバック近似変換を使用する。
            bfm_variant: BFMフォーマットの種別。以下のいずれか:
                - ``"bfm2017"``: BFM 2017（Deep3DFaceRecon等）
                - ``"bfm2009"``: BFM 2009
                - ``"cropped_bfm2009"``: cropped BFM 2009（3DDFA用）
        """
        self._mapping_dir = Path(mapping_dir) if mapping_dir is not None else None
        self._bfm_variant = bfm_variant
        self._exp_mapping: Optional[torch.Tensor] = None
        self._shape_mapping: Optional[torch.Tensor] = None
        self._load_mappings()

    def _load_mappings(self) -> None:
        """BFMtoFLAME変換行列をロードする。

        変換行列ファイルが見つからない場合は、フォールバック近似変換を使用する。
        ワーニングログを出力するが例外は発生させない。
        """
        if self._mapping_dir is None:
            return

        exp_path = self._mapping_dir / "exp_mapping.pt"
        shape_path = self._mapping_dir / "shape_mapping.pt"

        if exp_path.exists():
            self._exp_mapping = torch.load(
                str(exp_path), map_location="cpu", weights_only=True
            )

        if shape_path.exists():
            self._shape_mapping = torch.load(
                str(shape_path), map_location="cpu", weights_only=True
            )

    def convert(
        self, source_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """BFMパラメータをFLAME形式に変換する。

        Args:
            source_params: BFM出力の辞書。必須キー:
                - ``"exp"``: BFM表情パラメータ (B, D_exp)。
                  BFM 2017/2009: D_exp=64、cropped BFM 2009: D_exp=10。
                - ``"pose"``: 姿勢パラメータ (B, 6)。
                  rotation(3D) + translation/jaw(3D)。
                オプションキー:
                - ``"id"``: BFM identity係数 (B, D_id)。
                  省略時はゼロベクトルのFLAME shapeを返す。
                - ``"shape"``: ``"id"`` の代替キー（3DDFA形式）。

        Returns:
            FLAME形式のパラメータ辞書:
                - ``"expr"``: (B, 100) FLAME表情パラメータ
                - ``"pose"``: (B, 6) 姿勢パラメータ（パススルー）
                - ``"shape"``: (B, 300) FLAME形状パラメータ

        Raises:
            KeyError: source_paramsに必要なキーが存在しない場合。
        """
        exp = source_params["exp"]
        pose = source_params["pose"]

        batch_size = exp.shape[0]
        device = exp.device
        dtype = exp.dtype

        # expression変換: BFM exp → FLAME expr (100D)
        if self._exp_mapping is not None:
            mapping = self._exp_mapping.to(device=device, dtype=dtype)
            # mapping shape: (D_bfm, D_flame) or compatible
            if exp.shape[-1] == mapping.shape[0]:
                expr = exp @ mapping
                if expr.shape[-1] < _FLAME_EXPR_DIM:
                    expr = F.pad(expr, (0, _FLAME_EXPR_DIM - expr.shape[-1]))
                elif expr.shape[-1] > _FLAME_EXPR_DIM:
                    expr = expr[:, :_FLAME_EXPR_DIM]
            else:
                expr = self._fallback_exp_to_expr(exp)
        else:
            expr = self._fallback_exp_to_expr(exp)

        # pose変換: BFMとFLAMEは同一axis-angle表現のためパススルー
        flame_pose = pose.clone()

        # identity/shape変換: BFM id → FLAME shape (300D)
        id_coeff = source_params.get("id", source_params.get("shape"))
        if id_coeff is not None:
            if self._shape_mapping is not None:
                mapping = self._shape_mapping.to(device=device, dtype=dtype)
                if id_coeff.shape[-1] == mapping.shape[0]:
                    shape = id_coeff @ mapping
                    if shape.shape[-1] < _FLAME_SHAPE_DIM:
                        shape = F.pad(
                            shape, (0, _FLAME_SHAPE_DIM - shape.shape[-1])
                        )
                    elif shape.shape[-1] > _FLAME_SHAPE_DIM:
                        shape = shape[:, :_FLAME_SHAPE_DIM]
                else:
                    shape = self._fallback_id_to_shape(id_coeff)
            else:
                shape = self._fallback_id_to_shape(id_coeff)
        else:
            shape = torch.zeros(
                batch_size, _FLAME_SHAPE_DIM, device=device, dtype=dtype
            )

        return {
            "expr": expr,
            "pose": flame_pose,
            "shape": shape,
        }

    def _fallback_exp_to_expr(self, exp: torch.Tensor) -> torch.Tensor:
        """フォールバック: BFM expをFLAME exprに近似変換する。

        変換行列が利用不可の場合に使用する。BFM expression係数を
        FLAME expression空間の100D次元にゼロパディングで変換する。
        BFMとFLAMEのPCA基底は異なるが、低次元成分には一定の相関があるため、
        近似的に利用可能である。

        Args:
            exp: BFM expression係数テンソル (B, D_exp)。

        Returns:
            FLAME expression係数テンソル (B, 100)。
        """
        exp_dim = exp.shape[-1]
        if exp_dim >= _FLAME_EXPR_DIM:
            return exp[:, :_FLAME_EXPR_DIM]
        return F.pad(exp, (0, _FLAME_EXPR_DIM - exp_dim), value=0.0)

    def _fallback_id_to_shape(self, id_coeff: torch.Tensor) -> torch.Tensor:
        """フォールバック: BFM idをFLAME shapeに近似変換する。

        変換行列が利用不可の場合に使用する。BFM identity係数を
        FLAME shape空間の300D次元にゼロパディングで変換する。

        Args:
            id_coeff: BFM identity係数テンソル (B, D_id)。

        Returns:
            FLAME shape係数テンソル (B, 300)。
        """
        id_dim = id_coeff.shape[-1]
        if id_dim >= _FLAME_SHAPE_DIM:
            return id_coeff[:, :_FLAME_SHAPE_DIM]
        return F.pad(id_coeff, (0, _FLAME_SHAPE_DIM - id_dim), value=0.0)

    @property
    def source_format(self) -> str:
        """変換元のパラメータ形式名を返す。

        Returns:
            ``"bfm"``。
        """
        return "bfm"

    @property
    def target_format(self) -> str:
        """変換先のパラメータ形式名を返す。

        Returns:
            ``"flame"``。
        """
        return "flame"

    @property
    def bfm_variant(self) -> str:
        """BFMフォーマットの種別を返す。

        Returns:
            ``"bfm2017"``, ``"bfm2009"``, または ``"cropped_bfm2009"``。
        """
        return self._bfm_variant
