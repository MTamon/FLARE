"""Deep3DFaceRecon 特徴量抽出モジュール。

Deep3DFaceRecon（Deep 3D Face Reconstruction）はBFMベースの3DMMパラメータを
高精度に抽出するExtractorである。BFMmodelfront.matを使用してBFM空間のパラメータを
出力する。Route Aの標準Extractorとして使用される。

仕様書§3.1に基づく性能特性:
    - 推論速度: GPU推論で50 FPS目標（§6.5）
    - 出力: BFM id 80D + exp 64D + tex 80D + pose 6D + lighting 27D
    - BFMmodelfront.matの事前データに基づくパラメータ空間

仕様書§8.3の設計に従い、顔検出はface_detect.pyが担当し、
Deep3DFaceReconのextract()は検出済み画像のみを受け取る。

Example:
    Deep3DFaceReconExtractorの使用::

        extractor = Deep3DFaceReconExtractor(
            model_path="./checkpoints/deep3d/deep3d_epoch20.pth",
            device="cuda:0",
        )
        params = extractor.extract(image_tensor)
        print(params["exp"].shape)  # (1, 64)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from flare.extractors.base import BaseExtractor
from flare.utils.errors import ModelLoadError

_DEEP3D_PARAM_KEYS: list[str] = [
    "id",
    "exp",
    "tex",
    "pose",
    "lighting",
]
"""list[str]: Deep3DFaceReconが出力するパラメータキーのリスト。"""

_DEEP3D_PARAM_DIMS: dict[str, int] = {
    "id": 80,
    "exp": 64,
    "tex": 80,
    "pose": 6,
    "lighting": 27,
}
"""dict[str, int]: 各パラメータキーの次元数マッピング。"""

_DEEP3D_TOTAL_DIM: int = sum(_DEEP3D_PARAM_DIMS.values())
"""int: パラメータ総次元数 (80+64+80+6+27=257)。"""


class Deep3DFaceReconExtractor(BaseExtractor):
    """Deep3DFaceRecon特徴量抽出器。

    Deep3DFaceReconモデルをラップし、BaseExtractorインターフェースを提供する。
    BFMmodelfront.matを基盤としたBFMパラメータ空間で出力する。
    入力画像はface_detect.pyでクロッピング済みであることを前提とする。

    Deep3DFaceReconリポジトリ: sicxu/Deep3DFaceRecon_pytorch

    出力パラメータ:
        - ``id``: (B, 80) BFM identity係数
        - ``exp``: (B, 64) BFM expression係数
        - ``tex``: (B, 80) BFM texture係数
        - ``pose``: (B, 6) 姿勢（rotation 3D + translation 3D）
        - ``lighting``: (B, 27) 照明（9x3 SH係数）

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: モデルチェックポイントのパス。
        _bfm_path: BFMmodelfront.matファイルのパス。
        _model: ロード済みDeep3DFaceReconモデルインスタンス。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/deep3d/deep3d_epoch20.pth",
        device: str = "cuda:0",
        bfm_path: str = "./checkpoints/deep3d/BFM/BFMmodelfront.mat",
        deep3d_dir: Optional[str] = None,
    ) -> None:
        """Deep3DFaceReconExtractorを初期化する。

        Args:
            model_path: Deep3DFaceReconモデルチェックポイントファイルのパス。
            device: 推論デバイス。例: ``"cuda:0"``, ``"cpu"``。
            bfm_path: BFMmodelfront.matファイルのパス。
                BFM基底ベクトルを含むMATLABファイル。
            deep3d_dir: Deep3DFaceReconリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
                Noneの場合はインポート済みと仮定する。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._bfm_path = Path(bfm_path)
        self._deep3d_dir = deep3d_dir
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Deep3DFaceReconモデルをロードする。

        Deep3DFaceReconリポジトリからモデルをインポートし、
        BFMmodelfront.matとチェックポイントをロードする。

        Raises:
            ModelLoadError: モジュールのインポートまたはモデルロードに失敗した場合。
        """
        try:
            if self._deep3d_dir is not None:
                deep3d_path = str(Path(self._deep3d_dir).resolve())
                if deep3d_path not in sys.path:
                    sys.path.insert(0, deep3d_path)

            from models.networks import ReconNetWrapper  # type: ignore[import-untyped]

            self._model = ReconNetWrapper(
                net_recon="resnet50",
                use_last_fc=False,
            ).to(self._device)

            if self._model_path.exists():
                checkpoint = torch.load(
                    str(self._model_path),
                    map_location=self._device,
                    weights_only=False,
                )
                if "net_recon" in checkpoint:
                    self._model.load_state_dict(checkpoint["net_recon"])
                elif "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                else:
                    self._model.load_state_dict(checkpoint)

            self._model.eval()

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import Deep3DFaceRecon modules. Ensure the "
                f"Deep3DFaceRecon_pytorch repository is available. Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Deep3DFaceRecon model from "
                f"{self._model_path}: {e}"
            ) from e

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """1フレームの画像からBFMパラメータを抽出する。

        Deep3DFaceReconのネットワークは257D（id 80 + exp 64 + tex 80 +
        pose 6 + lighting 27）の係数ベクトルを出力する。
        本メソッドはそれをパラメータキーごとに分割して返す。

        Args:
            image: 入力画像テンソル。形状は ``(1, 3, H, W)``。
                face_detect.pyでクロッピング済み。値域は ``[0, 1]``。

        Returns:
            BFMパラメータの辞書。各テンソルの形状:
                - ``"id"``: (1, 80)
                - ``"exp"``: (1, 64)
                - ``"tex"``: (1, 80)
                - ``"pose"``: (1, 6)
                - ``"lighting"``: (1, 27)

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self._device)

        with torch.no_grad():
            coeffs = self._model(image)

        # Deep3DFaceReconの出力は (B, 257) の連結ベクトル
        # 分割順序: id(80) + exp(64) + tex(80) + pose(6) + lighting(27)
        id_coeff = coeffs[:, :80].detach()
        exp_coeff = coeffs[:, 80:144].detach()
        tex_coeff = coeffs[:, 144:224].detach()
        pose_coeff = coeffs[:, 224:230].detach()
        lighting_coeff = coeffs[:, 230:257].detach()

        return {
            "id": id_coeff,
            "exp": exp_coeff,
            "tex": tex_coeff,
            "pose": pose_coeff,
            "lighting": lighting_coeff,
        }

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数を返す。

        Returns:
            257 (80+64+80+6+27)。
        """
        return _DEEP3D_TOTAL_DIM

    @property
    def param_keys(self) -> list[str]:
        """出力辞書のキーリストを返す。

        Returns:
            ``["id", "exp", "tex", "pose", "lighting"]``。
        """
        return list(_DEEP3D_PARAM_KEYS)
