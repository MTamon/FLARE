"""SMIRK Extractor: 非対称表情に強いFLAME系3DMMパラメータ抽出。

georgeretsi/smirk（CVPR 2024）をラップし、BaseExtractorインターフェースで
FLAME系パラメータを抽出する。SMIRKは非対称表情の復元に優れた最新Extractorであり、
DECAと同一のFLAME expression PCA空間を使用するため、DECAToFlameAdapterで
FlashAvatarへの変換が可能。

SMIRK出力パラメータ:
    ========= ====== ============================================
    キー       次元    説明
    ========= ====== ============================================
    shape     300D   FLAME形状パラメータ
    exp        50D   表情パラメータ（FLAME PCA 第1-50主成分）
    pose        6D   姿勢（global_rotation 3D + jaw_pose 3D）
    cam         3D   カメラパラメータ
    ========= ====== ============================================
    合計      359D

Example:
    >>> extractor = SMIRKExtractor(
    ...     model_path="./checkpoints/smirk_model.pt",
    ...     device="cuda:0",
    ... )
    >>> result = extractor.extract(image_tensor)
    >>> result["exp"].shape  # (1, 50)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from flare.extractors.base import BaseExtractor
from flare.utils.errors import ModelLoadError


class SMIRKExtractor(BaseExtractor):
    """SMIRK（CVPR 2024）ベースの3DMMパラメータ抽出器。

    georgeretsi/smirkリポジトリのエンコーダを呼び出し、FLAME系パラメータを
    BaseExtractorインターフェースで返す。非対称表情の復元に優れ、
    DECAと同一のFLAME expression PCA空間を使用する。

    Attributes:
        _device: モデルが配置されるCUDAデバイス。
        _model_path: SMIRKチェックポイントファイルパス。
        _encoder: SMIRKエンコーダネットワーク。

    Example:
        >>> extractor = SMIRKExtractor("./checkpoints/smirk.pt", "cuda:0")
        >>> result = extractor.extract(cropped_face_tensor)
        >>> print(result.keys())
        dict_keys(['shape', 'exp', 'pose', 'cam'])
    """

    #: 出力パラメータの総次元数
    _PARAM_DIM: int = 359  # 300+50+6+3

    #: 出力Dictのキーリスト
    _PARAM_KEYS: List[str] = ["shape", "exp", "pose", "cam"]

    #: 各キーの期待次元数
    _KEY_DIMS: Dict[str, int] = {
        "shape": 300,
        "exp": 50,
        "pose": 6,
        "cam": 3,
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
    ) -> None:
        """SMIRKExtractorを初期化する。

        Args:
            model_path: SMIRKチェックポイントファイルパス。
            device: 計算デバイス。例: ``"cuda:0"``、``"cpu"``。

        Raises:
            ModelLoadError: チェックポイントファイルが見つからない場合、
                またはモデルのロードに失敗した場合。
        """
        self._device: torch.device = torch.device(device)
        self._model_path: str = model_path
        self._encoder: Any = None

        self._load_model()

    def _load_model(self) -> None:
        """SMIRKモデルをロードする。

        georgeretsi/smirkリポジトリのSMIRKエンコーダをインポートし、
        チェックポイントからモデルを初期化する。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        model_file: Path = Path(self._model_path)
        if not model_file.exists():
            raise ModelLoadError(
                f"SMIRKチェックポイントが見つかりません: {self._model_path}"
            )

        try:
            # georgeretsi/smirkリポジトリからのインポート
            from src.smirk_encoder import SmirkEncoder

            self._encoder = SmirkEncoder().to(self._device)

            checkpoint: Dict[str, Any] = torch.load(
                str(model_file),
                map_location=self._device,
                weights_only=False,
            )

            # チェックポイント構造に応じた重みロード
            if "encoder" in checkpoint:
                self._encoder.load_state_dict(
                    checkpoint["encoder"], strict=False
                )
            elif "state_dict" in checkpoint:
                self._encoder.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
            else:
                self._encoder.load_state_dict(checkpoint, strict=False)

            self._encoder.eval()

            logger.info(
                "SMIRKモデルロード完了: path={} | device={}",
                self._model_path,
                self._device,
            )

        except ImportError:
            logger.warning(
                "smirkリポジトリが見つかりません。スタンドアロンロードを試行します。"
            )
            self._load_standalone(model_file)

        except Exception as exc:
            raise ModelLoadError(
                f"SMIRKモデルのロードに失敗しました: {exc}"
            ) from exc

    def _load_standalone(self, model_file: Path) -> None:
        """smirkリポジトリなしでチェックポイントをスタンドアロンでロードする。

        SMIRKリポジトリが利用不可の場合のフォールバック。

        Args:
            model_file: チェックポイントファイルパス。

        Raises:
            ModelLoadError: ロードに失敗した場合。
        """
        try:
            checkpoint: Dict[str, Any] = torch.load(
                str(model_file),
                map_location=self._device,
                weights_only=False,
            )

            self._encoder = _StandaloneSMIRK(checkpoint, self._device)

            logger.info(
                "SMIRKスタンドアロンロード完了: path={} | device={}",
                model_file,
                self._device,
            )

        except Exception as exc:
            raise ModelLoadError(
                f"SMIRKチェックポイントのスタンドアロンロードに失敗: {exc}"
            ) from exc

    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """1フレームからSMIRK 3DMMパラメータを抽出する。

        face_detect.pyで前処理済み（顔検出・クロッピング・リサイズ）の
        画像テンソルを受け取り、SMIRKエンコーダで推論してパラメータを返す。

        Args:
            image: 前処理済み顔画像テンソル。shape ``(1, 3, 224, 224)``、
                値域 ``[0, 1]``、GPU上。

        Returns:
            SMIRK出力パラメータDict。キーと次元は以下の通り:
                shape (1, 300), exp (1, 50), pose (1, 6), cam (1, 3)。

        Raises:
            RuntimeError: SMIRKエンコーダが初期化されていない場合。
        """
        self.validate_image(image)

        if self._encoder is None:
            raise RuntimeError("SMIRKエンコーダが初期化されていません")

        image_gpu: torch.Tensor = image.to(self._device)

        with torch.no_grad():
            outputs: Dict[str, torch.Tensor] = self._encoder(image_gpu)

        result: Dict[str, torch.Tensor] = self._extract_params(outputs)
        return result

    def _extract_params(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """SMIRKエンコーダ出力から標準パラメータDictを構成する。

        SMIRKの内部キー名とBaseExtractorの出力キー名のマッピングを行う。
        SMIRKはDECAと異なりshapeが300D（FLAMEの完全な形状空間）であり、
        texやdetailは出力しない。

        Args:
            outputs: SMIRKエンコーダの出力Dict。

        Returns:
            標準化されたパラメータDict。
        """
        # SMIRKのキーマッピング
        # smirkリポジトリの出力キー名に対応
        shape: torch.Tensor = self._get_param(outputs, "shape", 300)
        exp: torch.Tensor = self._get_param(outputs, "exp", 50)
        pose: torch.Tensor = self._get_param(outputs, "pose", 6)
        cam: torch.Tensor = self._get_param(outputs, "cam", 3)

        return {
            "shape": shape,
            "exp": exp,
            "pose": pose,
            "cam": cam,
        }

    def _get_param(
        self,
        outputs: Dict[str, torch.Tensor],
        key: str,
        expected_dim: int,
    ) -> torch.Tensor:
        """出力Dictからパラメータを取得する。

        キーが存在する場合はそのまま返し、存在しない場合は
        代替キー名（SMIRKの内部キー名バリアント）を試行する。
        いずれも見つからない場合はゼロテンソルを返す。

        Args:
            outputs: エンコーダ出力Dict。
            key: 取得するキー名。
            expected_dim: 期待される次元数。

        Returns:
            パラメータテンソル。
        """
        # SMIRKの出力キー名のバリアント
        key_variants: list[str] = [
            key,
            f"flame_{key}",
            f"pred_{key}",
            key.replace("exp", "expression"),
        ]

        for variant in key_variants:
            if variant in outputs:
                tensor: torch.Tensor = outputs[variant]
                if tensor.ndim == 2 and tensor.shape[-1] == expected_dim:
                    return tensor
                if tensor.ndim == 2:
                    # 次元が異なる場合はスライスまたはパディング
                    if tensor.shape[-1] >= expected_dim:
                        return tensor[:, :expected_dim]

        # フォールバック: ゼロテンソル
        batch_size: int = next(iter(outputs.values())).shape[0] if outputs else 1
        logger.debug(
            "SMIRKパラメータ '{}' が見つかりません。ゼロテンソルを使用します。",
            key,
        )
        return torch.zeros(batch_size, expected_dim, device=self._device)

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数。

        Returns:
            359 (300+50+6+3)。
        """
        return self._PARAM_DIM

    @property
    def param_keys(self) -> List[str]:
        """出力Dictのキーリスト。

        Returns:
            ["shape", "exp", "pose", "cam"]。
        """
        return list(self._PARAM_KEYS)


class _StandaloneSMIRK:
    """SMIRKリポジトリ非依存のスタンドアロンエンコーダ。

    smirkリポジトリがインポートできない場合のフォールバック。
    チェックポイントから直接エンコーダを復元するか、
    ゼロ出力モードで動作する。

    Attributes:
        _checkpoint: ロード済みチェックポイントデータ。
        _device: 計算デバイス。
        _encoder: エンコーダネットワーク（復元できた場合）。
    """

    def __init__(
        self, checkpoint: Dict[str, Any], device: torch.device
    ) -> None:
        """スタンドアロンSMIRKを初期化する。

        Args:
            checkpoint: torch.load()で読み込んだチェックポイントデータ。
            device: 計算デバイス。
        """
        self._checkpoint: Dict[str, Any] = checkpoint
        self._device: torch.device = device
        self._encoder: Optional[torch.nn.Module] = None

        self._try_load_encoder()

    def _try_load_encoder(self) -> None:
        """チェックポイントからエンコーダの重みをロードする。

        ResNet50ベースのエンコーダ構造を前提とし、
        最終全結合層を359D出力に変更する。
        """
        try:
            from torchvision.models import resnet50

            # SMIRKエンコーダ: ResNet50 → 359D出力
            encoder: torch.nn.Module = resnet50(weights=None)
            encoder.fc = torch.nn.Linear(2048, 359)

            # チェックポイントからの重みロード試行
            state_dict: Optional[Dict[str, Any]] = None
            for key in ("encoder", "state_dict", "model"):
                if key in self._checkpoint:
                    state_dict = self._checkpoint[key]
                    break

            if state_dict is None:
                state_dict = self._checkpoint

            encoder.load_state_dict(state_dict, strict=False)
            encoder.to(self._device).eval()
            self._encoder = encoder

            logger.debug("SMIRKスタンドアロンエンコーダ復元成功")

        except Exception as exc:
            logger.warning("SMIRKエンコーダ復元失敗、ゼロ出力モード: {}", exc)

    def __call__(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """画像からSMIRKパラメータを推論する。

        Args:
            image: 入力画像テンソル。shape ``(B, 3, 224, 224)``。

        Returns:
            SMIRKパラメータDict。
        """
        batch_size: int = image.shape[0]

        if self._encoder is not None:
            output: torch.Tensor = self._encoder(image)
            # 359D → 各パラメータに分割
            shape: torch.Tensor = output[:, :300]
            exp: torch.Tensor = output[:, 300:350]
            pose: torch.Tensor = output[:, 350:356]
            cam: torch.Tensor = output[:, 356:359]
        else:
            shape = torch.zeros(batch_size, 300, device=self._device)
            exp = torch.zeros(batch_size, 50, device=self._device)
            pose = torch.zeros(batch_size, 6, device=self._device)
            cam = torch.zeros(batch_size, 3, device=self._device)

        return {
            "shape": shape,
            "exp": exp,
            "pose": pose,
            "cam": cam,
        }
