"""SMIRK 特徴量抽出モジュール。

SMIRK（Semi-supervised 3D Face Reconstruction with Neural Inverse
Rendering）を使用してFLAMEベースの3DMMパラメータを抽出する。
非対称表情に強い 2024 年系 Extractor である。

対応リポジトリ:
    - upstream: georgeretsi/smirk
    - 推奨フォーク: MTamon/smirk@release/cuda128
      (Python 3.11 + CUDA 12.8 + PyTorch 2.9.x、DECA / FlashAvatar の
      install_128.sh と整合する依存ピン。SmirkEncoder.forward に
      ``shape``/``exp``/``pose``/``cam``/``eyelid`` の FLARE 互換エイリアス
      を追加済み)

仕様書§4.1に基づく性能特性:
    - 推論速度: DECAと同等
    - 出力: FLAME exp 50D + pose 6D + cam 3D + eyelid 2D
    - 非対称表情に強い

Example:
    SMIRKExtractorの使用::

        extractor = SMIRKExtractor(
            model_path="./checkpoints/smirk/SMIRK_em1.pt",
            device="cuda:0",
            smirk_dir="./third_party/smirk",
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

_LEGACY_TO_FLARE_KEYS: dict[str, str] = {
    "shape_params": "shape",
    "expression_params": "exp",
    "eyelid_params": "eyelid",
}
"""dict[str, str]: SMIRK upstream の旧キー → FLARE 互換キーのマッピング。

MTamon/smirk@release/cuda128 では SmirkEncoder.forward が直接 FLARE 互換
エイリアス (``shape``/``exp``/``pose``/``cam``/``eyelid``) を返すため、
本マップは upstream の georgeretsi/smirk を直接使う場合のフォールバック。
"""

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
        model_path: str = "./checkpoints/smirk/SMIRK_em1.pt",
        device: str = "cuda:0",
        smirk_dir: Optional[str] = None,
    ) -> None:
        """SMIRKExtractorを初期化する。

        Args:
            model_path: SMIRKモデルチェックポイントファイルのパス。
                MTamon/smirk@release/cuda128 のデフォルト名は ``SMIRK_em1.pt``。
            device: 推論デバイス。例: ``"cuda:0"``, ``"cpu"``。
            smirk_dir: SMIRKリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
                Noneの場合はインポート済みと仮定する。
                推奨: ``./third_party/smirk`` (release/cuda128 ブランチ)。

        Raises:
            ModelLoadError: SMIRKモデルのロードに失敗した場合。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._smirk_dir = smirk_dir
        self._encoder: Any = None

        # SMIRK は MobileNetV3 backbone × 3 を毎フレーム同形状で実行する。
        # cudnn.benchmark=True で初回呼び出し後に最適 kernel 選択を固定でき
        # 推論を 1.5-2x 高速化できる (入力形状が固定のため安全)。
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True

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

        normalized = self._normalize_output_keys(outputs)
        return {
            "shape": normalized["shape"].detach(),
            "exp": normalized["exp"].detach(),
            "pose": normalized["pose"].detach(),
            "cam": normalized["cam"].detach(),
            "eyelid": normalized["eyelid"].detach(),
        }

    def _normalize_output_keys(
        self, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SmirkEncoder の出力を FLARE 互換キーに正規化する。

        MTamon/smirk@release/cuda128 は ``shape``/``exp``/``pose``/``cam``/
        ``eyelid`` の互換エイリアスを直接出力するためそのまま返す。
        upstream の georgeretsi/smirk を直接使う場合は ``shape_params`` などの
        旧キーをマップし、``pose`` は ``pose_params (3) + jaw_params (3)`` を
        cat して 6D に組み直す。

        Args:
            outputs: SmirkEncoder.forward の戻り値。

        Returns:
            ``shape``/``exp``/``pose``/``cam``/``eyelid`` の 5 キーを
            含む辞書。元の辞書はそのまま、足りないキーのみ追加する。
        """
        normalized: dict[str, torch.Tensor] = dict(outputs)

        for legacy_key, flare_key in _LEGACY_TO_FLARE_KEYS.items():
            if flare_key not in normalized and legacy_key in normalized:
                normalized[flare_key] = normalized[legacy_key]

        if "pose" not in normalized:
            if "pose_params" in normalized and "jaw_params" in normalized:
                normalized["pose"] = torch.cat(
                    [normalized["pose_params"], normalized["jaw_params"]], dim=-1
                )
            else:
                raise KeyError(
                    "SmirkEncoder output missing 'pose' (and 'pose_params'/'jaw_params'). "
                    f"Got keys: {sorted(normalized.keys())}"
                )

        return normalized

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
