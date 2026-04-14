"""3DDFA V2 特徴量抽出モジュール。

3DDFA V2（3D Dense Face Alignment Version 2）はBFM形式の3DMMパラメータを
CPU推論でも高速に抽出できる軽量Extractorである。
1.35ms/imageの高速処理が可能であり、リアルタイムパイプラインに適する。

仕様書§4.1に基づく性能特性:
    - 推論速度: 1.35ms/image（CPU）
    - 出力: BFM shape 40D + exp 10D
    - CPU推論対応でGPU不要

仕様書§8.3の設計に従い、顔検出はface_detect.pyが担当し、
TDDFAのextract()は検出済み画像のみを受け取る。

Example:
    TDDFAExtractorの使用::

        extractor = TDDFAExtractor(
            model_path="./checkpoints/3ddfa/mb1_120x120.onnx",
            device="cpu",
        )
        params = extractor.extract(image_tensor)
        print(params["exp"].shape)  # (1, 10)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from flare.extractors.base import BaseExtractor
from flare.utils.errors import ModelLoadError

_TDDFA_PARAM_KEYS: list[str] = [
    "shape",
    "exp",
]
"""list[str]: 3DDFA V2が出力するパラメータキーのリスト。"""

_TDDFA_PARAM_DIMS: dict[str, int] = {
    "shape": 40,
    "exp": 10,
}
"""dict[str, int]: 各パラメータキーの次元数マッピング。"""

_TDDFA_TOTAL_DIM: int = sum(_TDDFA_PARAM_DIMS.values())
"""int: パラメータ総次元数 (40+10=50)。"""


class TDDFAExtractor(BaseExtractor):
    """3DDFA V2 特徴量抽出器。

    3DDFA V2モデルをラップし、BaseExtractorインターフェースを提供する。
    MobileNetベースの軽量アーキテクチャにより、CPU推論でも1.35ms/imageの
    高速処理を実現する。BFM形式のshapeとexpressionパラメータを出力する。

    3DDFA V2リポジトリ: cleardusk/3DDFA_V2

    出力パラメータ:
        - ``shape``: (B, 40) BFM形状係数
        - ``exp``: (B, 10) BFM表情係数

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: モデルチェックポイントのパス。
        _tddfa: ロード済み3DDFA V2モデルインスタンス。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/3ddfa/mb1_120x120.onnx",
        device: str = "cpu",
        tddfa_dir: Optional[str] = None,
    ) -> None:
        """TDDFAExtractorを初期化する。

        Args:
            model_path: 3DDFA V2モデルチェックポイントファイルのパス。
                ONNXフォーマットまたはPyTorchフォーマットを受け付ける。
            device: 推論デバイス。CPU推論が標準。
                例: ``"cpu"``, ``"cuda:0"``。
            tddfa_dir: 3DDFA V2リポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
                Noneの場合はインポート済みと仮定する。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._tddfa_dir = tddfa_dir
        self._tddfa: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """3DDFA V2モデルをロードする。

        3DDFA V2リポジトリからモデルをインポートし、チェックポイントをロードする。
        ONNX形式が利用可能な場合はONNXランタイムを使用し、
        そうでなければPyTorchモデルをロードする。

        Raises:
            ModelLoadError: モジュールのインポートまたはモデルロードに失敗した場合。
        """
        try:
            if self._tddfa_dir is not None:
                tddfa_path = str(Path(self._tddfa_dir).resolve())
                if tddfa_path not in sys.path:
                    sys.path.insert(0, tddfa_path)

            from TDDFA import TDDFA  # type: ignore[import-untyped]

            cfg = {
                "checkpoint_fp": str(self._model_path),
                "bfm_fp": str(self._model_path.parent / "bfm_noneck_v3.pkl"),
                "size": 120,
                "num_params": 62,
            }

            self._tddfa = TDDFA(**cfg)

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import 3DDFA V2 modules. Ensure the 3DDFA_V2 "
                f"repository is available. Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load 3DDFA V2 model from {self._model_path}: {e}"
            ) from e

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """1フレームの画像からBFM shape/expパラメータを抽出する。

        3DDFA V2は62Dの係数ベクトル（shape 40D + exp 10D + pose 12D）を
        出力する。本メソッドはshapeとexpのみを返す。

        Args:
            image: 入力画像テンソル。形状は ``(1, 3, H, W)``。
                face_detect.pyでクロッピング済み。値域は ``[0, 1]``。

        Returns:
            BFMパラメータの辞書。各テンソルの形状:
                - ``"shape"``: (1, 40)
                - ``"exp"``: (1, 10)

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self._device)

        with torch.no_grad():
            # 3DDFA V2はnumpy/PIL画像を受け取るが、
            # 統一インターフェースのためtorch.Tensorを変換して使用する
            import numpy as np

            img_np = (
                image[0]
                .permute(1, 2, 0)
                .mul(255)
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
            )

            param_lst = self._tddfa(img_np)

            if isinstance(param_lst, list):
                params = param_lst[0]
            else:
                params = param_lst

            if isinstance(params, np.ndarray):
                params = torch.from_numpy(params).float()
            elif not isinstance(params, torch.Tensor):
                params = torch.tensor(params, dtype=torch.float32)

        # 3DDFA V2 output: shape(40) + exp(10) + pose(12) = 62D
        shape_coeff = params[:40].unsqueeze(0).to(self._device)
        exp_coeff = params[40:50].unsqueeze(0).to(self._device)

        return {
            "shape": shape_coeff.detach(),
            "exp": exp_coeff.detach(),
        }

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数を返す。

        Returns:
            50 (40+10)。
        """
        return _TDDFA_TOTAL_DIM

    @property
    def param_keys(self) -> list[str]:
        """出力辞書のキーリストを返す。

        Returns:
            ``["shape", "exp"]``。
        """
        return list(_TDDFA_PARAM_KEYS)
