"""SMIRK 特徴量抽出モジュール。

SMIRK（Symmetric Multi-view Inverse Rendering of the face using a Knowledge
distillation framework）を使用してFLAMEベースの3DMMパラメータを抽出する。
2024年最新の非対称表情に強いExtractorである。

仕様書§4.1に基づく性能特性:
    - 推論速度: DECAと同等
    - 出力: FLAME exp 50D + pose 6D
    - 非対称表情に強い

Example:
    SMIRKExtractorの使用::

        extractor = SMIRKExtractor(
            model_path="./checkpoints/smirk/smirk_model.pt",
            device="cuda:0",
        )
        params = extractor.extract(image_tensor)
        print(params["exp"].shape)  # (1, 50)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from flare.extractors.base import BaseExtractor
from flare.utils.errors import ModelLoadError

_SMIRK_PARAM_KEYS: list[str] = [
    "shape",
    "exp",
    "pose",
    "cam",
    "eyelid",
]
"""list[str]: SMIRKが出力するパラメータキーのリスト。"""

_SMIRK_PARAM_DIMS: dict[str, int] = {
    "shape": 300,
    "exp": 50,
    "pose": 6,
    "cam": 3,
    "eyelid": 2,
}
"""dict[str, int]: 各パラメータキーの次元数マッピング。"""

_SMIRK_TOTAL_DIM: int = sum(_SMIRK_PARAM_DIMS.values())
"""int: パラメータ総次元数 (300+50+6+3+2=361)。"""


class SMIRKExtractor(BaseExtractor):
    """SMIRK特徴量抽出器。

    SMIRKモデルをラップし、BaseExtractorインターフェースを提供する。
    入力画像はface_detect.pyでクロッピング済みであることを前提とする。

    SMIRKリポジトリ: georgeretsi/smirk

    出力パラメータ:
        - ``shape``: (B, 300) FLAME形状パラメータ
        - ``exp``: (B, 50) 表情パラメータ (FLAME PCA 第1-50主成分)
        - ``pose``: (B, 6) 姿勢 (global_rotation 3D + jaw_pose 3D)
        - ``cam``: (B, 3) カメラパラメータ
        - ``eyelid``: (B, 2) 瞼パラメータ

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: SMIRKモデルチェックポイントのパス。
        _encoder: ロード済みSMIRKエンコーダインスタンス。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/smirk/smirk_encoder.pt",
        device: str = "cuda:0",
        smirk_dir: Optional[str] = None,
    ) -> None:
        """SMIRKExtractorを初期化する。

        Args:
            model_path: SMIRKモデルチェックポイントファイルのパス。
            device: 推論デバイス。例: ``"cuda:0"``, ``"cpu"``。
            smirk_dir: SMIRKリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
                Noneの場合はインポート済みと仮定する。

        Raises:
            ModelLoadError: SMIRKモデルのロードに失敗した場合。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._smirk_dir = smirk_dir
        self._encoder: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """SMIRKモデルをロードする。

        Raises:
            ModelLoadError: モジュールのインポートまたはモデルロードに失敗した場合。
        """
        try:
            if self._smirk_dir is not None:
                smirk_path = str(Path(self._smirk_dir).resolve())
                if smirk_path not in sys.path:
                    sys.path.insert(0, smirk_path)

            from src.smirk_encoder import SmirkEncoder  # type: ignore[import-untyped]

            self._encoder = SmirkEncoder().to(self._device)

            if self._model_path.exists():
                checkpoint = torch.load(
                    str(self._model_path),
                    map_location=self._device,
                    weights_only=False,
                )
                if "state_dict" in checkpoint:
                    self._encoder.load_state_dict(checkpoint["state_dict"])
                else:
                    self._encoder.load_state_dict(checkpoint)

            self._encoder.eval()

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import SMIRK modules. Ensure the SMIRK repository "
                f"is available. Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load SMIRK model from {self._model_path}: {e}"
            ) from e

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """1フレームの画像からSMIRKパラメータを抽出する。

        Args:
            image: 入力画像テンソル。形状は ``(1, 3, H, W)``。
                face_detect.pyでクロッピング済み。値域は ``[0, 1]``。

        Returns:
            SMIRKパラメータの辞書。各テンソルの形状:
                - ``"shape"``: (1, 300)
                - ``"exp"``: (1, 50)
                - ``"pose"``: (1, 6)
                - ``"cam"``: (1, 3)
                - ``"eyelid"``: (1, 2)

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self._device)

        with torch.no_grad():
            outputs = self._encoder(image)

        return {
            "shape": outputs["shape"].detach(),
            "exp": outputs["exp"].detach(),
            "pose": outputs["pose"].detach(),
            "cam": outputs["cam"].detach(),
            "eyelid": outputs["eyelid"].detach(),
        }

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数を返す。

        Returns:
            361 (300+50+6+3+2)。
        """
        return _SMIRK_TOTAL_DIM

    @property
    def param_keys(self) -> list[str]:
        """出力辞書のキーリストを返す。

        Returns:
            ``["shape", "exp", "pose", "cam", "eyelid"]``。
        """
        return list(_SMIRK_PARAM_KEYS)
