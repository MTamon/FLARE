"""DECA 特徴量抽出モジュール。

DECA（Detailed Expression Capture and Animation）を使用してFLAMEベースの
3DMMパラメータを抽出する。SIGGRAPH 2021発表、最も広く使われているFLAME
Extractorである。

仕様書§4.1に基づく性能特性:
    - 推論速度: ~8.4ms/image (~119 FPS)
    - 出力: FLAME exp 50D + pose 6D + detail 128D 等

仕様書§8.3の設計に従い、顔検出はface_detect.pyが担当し、
DECAのencode()は検出済み画像のみを受け取る。DECA encode()に
内部FAN処理がないことが確認されているため、共存に改修不要。

Example:
    DECAExtractorの使用::

        extractor = DECAExtractor(
            model_path="./checkpoints/deca_model.tar",
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

_DECA_PARAM_KEYS: list[str] = [
    "shape",
    "tex",
    "exp",
    "pose",
    "cam",
    "light",
    "detail",
]
"""list[str]: DECAが出力するパラメータキーのリスト。"""

_DECA_PARAM_DIMS: dict[str, int] = {
    "shape": 100,
    "tex": 50,
    "exp": 50,
    "pose": 6,
    "cam": 3,
    "light": 27,
    "detail": 128,
}
"""dict[str, int]: 各パラメータキーの次元数マッピング。"""

_DECA_TOTAL_DIM: int = sum(_DECA_PARAM_DIMS.values())
"""int: パラメータ総次元数 (100+50+50+6+3+27+128=364)。"""


class DECAExtractor(BaseExtractor):
    """DECA特徴量抽出器。

    DECAモデルをラップし、BaseExtractorインターフェースを提供する。
    入力画像はface_detect.pyでクロッピング済みであることを前提とする。

    DECAリポジトリ: YadiraF/DECA

    出力パラメータ:
        - ``shape``: (B, 100) FLAME形状パラメータ
        - ``tex``: (B, 50) テクスチャパラメータ
        - ``exp``: (B, 50) 表情パラメータ (FLAME PCA 第1-50主成分)
        - ``pose``: (B, 6) 姿勢 (global_rotation 3D + jaw_pose 3D)
        - ``cam``: (B, 3) カメラパラメータ
        - ``light``: (B, 27) 照明 (9x3 SH)
        - ``detail``: (B, 128) 詳細パラメータ (E_detail ResNet)

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: DECAモデルチェックポイントのパス。
        _deca: ロード済みDECAモデルインスタンス。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/deca_model.tar",
        device: str = "cuda:0",
        deca_dir: Optional[str] = None,
    ) -> None:
        """DECAExtractorを初期化する。

        Args:
            model_path: DECAモデルチェックポイントファイルのパス。
            device: 推論デバイス。例: ``"cuda:0"``, ``"cpu"``。
            deca_dir: DECAリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
                Noneの場合はインポート済みと仮定する。

        Raises:
            ModelLoadError: DECAモデルのロードに失敗した場合。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._deca_dir = deca_dir
        self._deca: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """DECAモデルをロードする。

        DECAリポジトリからモデルをインポートし、チェックポイントをロードする。

        Raises:
            ModelLoadError: モジュールのインポートまたはモデルロードに失敗した場合。
        """
        try:
            if self._deca_dir is not None:
                deca_path = str(Path(self._deca_dir).resolve())
                if deca_path not in sys.path:
                    sys.path.insert(0, deca_path)

            from decalib.deca import DECA  # type: ignore[import-untyped]
            from decalib.utils.config import cfg as deca_cfg  # type: ignore[import-untyped]

            deca_cfg.model.use_tex = True
            deca_cfg.rasterizer_type = "pytorch3d"

            if self._model_path.exists():
                deca_cfg.pretrained_modelpath = str(self._model_path)

            self._deca = DECA(config=deca_cfg, device=self._device)
            self._deca.eval()

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import DECA modules. Ensure the DECA repository "
                f"is available. Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load DECA model from {self._model_path}: {e}"
            ) from e

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """1フレームの画像からDECAパラメータを抽出する。

        Args:
            image: 入力画像テンソル。形状は ``(1, 3, H, W)``。
                face_detect.pyでクロッピング済み。値域は ``[0, 1]``。

        Returns:
            DECAパラメータの辞書。各テンソルの形状:
                - ``"shape"``: (1, 100)
                - ``"tex"``: (1, 50)
                - ``"exp"``: (1, 50)
                - ``"pose"``: (1, 6)
                - ``"cam"``: (1, 3)
                - ``"light"``: (1, 27)
                - ``"detail"``: (1, 128)

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self._device)

        with torch.no_grad():
            codedict = self._deca.encode(image)

        return {
            "shape": codedict["shape"].detach(),
            "tex": codedict["tex"].detach(),
            "exp": codedict["exp"].detach(),
            "pose": codedict["pose"].detach(),
            "cam": codedict["cam"].detach(),
            "light": codedict["light"].reshape(-1, 27).detach(),
            "detail": codedict["detail"].detach(),
        }

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数を返す。

        Returns:
            364 (100+50+50+6+3+27+128)。
        """
        return _DECA_TOTAL_DIM

    @property
    def param_keys(self) -> list[str]:
        """出力辞書のキーリストを返す。

        Returns:
            ``["shape", "tex", "exp", "pose", "cam", "light", "detail"]``。
        """
        return list(_DECA_PARAM_KEYS)
