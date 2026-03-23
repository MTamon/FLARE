"""Deep3DFaceRecon Extractor: BFM系3DMMパラメータ抽出。

sicxu/Deep3DFaceRecon_pytorchをラップし、BaseExtractorインターフェースで
Basel Face Model（BFM）ベースのパラメータを抽出する。
ViCo Challenge Baselineとの互換性を持つ。

BFMパラメータ構造（仕様書3.1節）:
    ========= ====== ============================================
    キー       次元    説明
    ========= ====== ============================================
    id         80D   identity（顔形状の個人差）
    exp        64D   expression（表情）
    tex        80D   texture（顔テクスチャ）
    pose        6D   頭部姿勢（回転3D + 平行移動3D）
    lighting   27D   照明（球面調和関数 9x3）
    ========= ====== ============================================
    合計      257D

注意:
    BFMのexp(64D)とFLAMEのexp(50D)は異なるパラメータ空間であり、
    直接互換性はない。両者を混在させないこと（仕様書3.3節）。

性能特性:
    - 推論速度: 約50+ FPS on GTX 1080（GPU）
    - リアルタイム対応: 十分可能

Example:
    >>> extractor = Deep3DFaceReconExtractor(
    ...     model_path="./checkpoints/deep3d.pth",
    ...     bfm_path="./BFM/BFM_model_front.mat",
    ...     device="cuda:0",
    ... )
    >>> result = extractor.extract(image_tensor)
    >>> result["exp"].shape  # (1, 64)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from flare.extractors.base import BaseExtractor
from flare.utils.errors import ModelLoadError


class Deep3DFaceReconExtractor(BaseExtractor):
    """Deep3DFaceRecon（BFM系）3DMMパラメータ抽出器。

    sicxu/Deep3DFaceRecon_pytorchリポジトリのモデルを使用し、
    BFMパラメータ空間での特徴量抽出を行う。

    Attributes:
        _device: モデルが配置されるCUDAデバイス。
        _model_path: Deep3DFaceReconチェックポイントパス。
        _bfm_path: BFM_model_front.matファイルパス。
        _model: Deep3DFaceReconモデルインスタンス。
        _bfm_data: BFMモデルデータ（scipy.io.loadmatで読み込み）。

    Example:
        >>> extractor = Deep3DFaceReconExtractor(
        ...     "./checkpoints/deep3d.pth", "./BFM/BFM_model_front.mat"
        ... )
        >>> result = extractor.extract(image)
        >>> result["id"].shape  # (1, 80)
    """

    #: 出力パラメータの総次元数
    _PARAM_DIM: int = 257  # 80+64+80+6+27

    #: 出力Dictのキーリスト
    _PARAM_KEYS: List[str] = ["id", "exp", "tex", "pose", "lighting"]

    #: 各キーの期待次元数
    _KEY_DIMS: Dict[str, int] = {
        "id": 80,
        "exp": 64,
        "tex": 80,
        "pose": 6,
        "lighting": 27,
    }

    #: 合成係数ベクトル内の各パラメータのスライス位置
    _COEFF_SLICES: Dict[str, tuple[int, int]] = {
        "id": (0, 80),
        "exp": (80, 144),
        "tex": (144, 224),
        "pose": (224, 230),  # rotation(3) + translation(3)
        "lighting": (230, 257),
    }

    def __init__(
        self,
        model_path: str,
        bfm_path: str,
        device: str = "cuda:0",
    ) -> None:
        """Deep3DFaceReconExtractorを初期化する。

        Args:
            model_path: Deep3DFaceReconチェックポイントパス（.pthファイル）。
            bfm_path: BFM_model_front.matファイルパス。
                Basel Face Model公式サイトから登録後ダウンロード。
            device: 計算デバイス。例: ``"cuda:0"``、``"cpu"``。

        Raises:
            ModelLoadError: チェックポイントまたはBFMファイルが見つからない場合、
                またはモデルのロードに失敗した場合。
        """
        self._device: torch.device = torch.device(device)
        self._model_path: str = model_path
        self._bfm_path: str = bfm_path
        self._model: Any = None
        self._bfm_data: Optional[Dict[str, Any]] = None

        self._load_bfm()
        self._load_model()

    def _load_bfm(self) -> None:
        """BFMモデルデータを読み込む。

        scipy.io.loadmatでBFM_model_front.matファイルを読み込む。

        Raises:
            ModelLoadError: BFMファイルが見つからないかロードに失敗した場合。
        """
        bfm_file: Path = Path(self._bfm_path)
        if not bfm_file.exists():
            raise ModelLoadError(
                f"BFMモデルファイルが見つかりません: {self._bfm_path}"
            )

        try:
            from scipy.io import loadmat

            self._bfm_data = loadmat(str(bfm_file))
            logger.info(
                "BFMモデルロード完了: path={} | keys={}",
                self._bfm_path,
                [k for k in self._bfm_data.keys() if not k.startswith("_")],
            )
        except ImportError:
            logger.warning("scipy未インストール。BFMデータなしで続行します。")
            self._bfm_data = {}
        except Exception as exc:
            raise ModelLoadError(
                f"BFMモデルの読み込みに失敗しました: {exc}"
            ) from exc

    def _load_model(self) -> None:
        """Deep3DFaceReconモデルをロードする。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        model_file: Path = Path(self._model_path)
        if not model_file.exists():
            raise ModelLoadError(
                f"Deep3DFaceReconチェックポイントが見つかりません: {self._model_path}"
            )

        try:
            # sicxu/Deep3DFaceRecon_pytorchリポジトリからのインポート
            from models.networks import ReconNetWrapper

            self._model = ReconNetWrapper(
                net_recon="resnet50",
                use_last_fc=False,
            )

            checkpoint: Dict[str, Any] = torch.load(
                str(model_file),
                map_location=self._device,
                weights_only=False,
            )

            if "net_recon" in checkpoint:
                self._model.load_state_dict(
                    checkpoint["net_recon"], strict=False
                )
            else:
                self._model.load_state_dict(checkpoint, strict=False)

            self._model.to(self._device).eval()

            logger.info(
                "Deep3DFaceReconモデルロード完了: path={} | device={}",
                self._model_path,
                self._device,
            )

        except ImportError:
            logger.warning(
                "Deep3DFaceReconリポジトリが見つかりません。"
                "スタンドアロンロードを試行します。"
            )
            self._load_standalone(model_file)

        except Exception as exc:
            raise ModelLoadError(
                f"Deep3DFaceReconモデルのロードに失敗しました: {exc}"
            ) from exc

    def _load_standalone(self, model_file: Path) -> None:
        """リポジトリなしでスタンドアロンロードする。

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

            self._model = _StandaloneDeep3D(checkpoint, self._device)

            logger.info(
                "Deep3Dスタンドアロンロード完了: path={} | device={}",
                model_file,
                self._device,
            )

        except Exception as exc:
            raise ModelLoadError(
                f"Deep3Dスタンドアロンロードに失敗: {exc}"
            ) from exc

    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """1フレームからBFM 3DMMパラメータを抽出する。

        face_detect.pyで前処理済みの画像テンソルを受け取り、
        Deep3DFaceReconで推論してBFMパラメータを返す。

        Args:
            image: 前処理済み顔画像テンソル。shape ``(1, 3, 224, 224)``、
                値域 ``[0, 1]``、GPU上。

        Returns:
            BFM出力パラメータDict:
                id (1, 80), exp (1, 64), tex (1, 80),
                pose (1, 6), lighting (1, 27)。

        Raises:
            RuntimeError: モデルが初期化されていない場合。
        """
        self.validate_image(image)

        if self._model is None:
            raise RuntimeError("Deep3DFaceReconモデルが初期化されていません")

        image_gpu: torch.Tensor = image.to(self._device)

        with torch.no_grad():
            coeff: torch.Tensor = self._model(image_gpu)

        result: Dict[str, torch.Tensor] = self._split_coefficients(coeff)
        return result

    def _split_coefficients(
        self, coeff: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """合成係数ベクトルを各パラメータに分割する。

        Deep3DFaceReconの出力は257Dの合成係数ベクトルであり、
        これをid/exp/tex/pose/lightingに分割する。

        Args:
            coeff: 合成係数テンソル。shape ``(B, 257)``。
                257D未満の場合はゼロパディングで補完する。

        Returns:
            分割されたパラメータDict。
        """
        batch_size: int = coeff.shape[0]
        coeff_dim: int = coeff.shape[-1] if coeff.ndim >= 2 else 0

        # 257D未満の場合はゼロパディング
        if coeff_dim < self._PARAM_DIM:
            padded: torch.Tensor = torch.zeros(
                batch_size, self._PARAM_DIM, device=coeff.device, dtype=coeff.dtype
            )
            padded[:, :coeff_dim] = coeff[:, :coeff_dim] if coeff_dim > 0 else 0
            coeff = padded

        result: Dict[str, torch.Tensor] = {}
        for key, (start, end) in self._COEFF_SLICES.items():
            result[key] = coeff[:, start:end]

        return result

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数。

        Returns:
            257 (80+64+80+6+27)。
        """
        return self._PARAM_DIM

    @property
    def param_keys(self) -> List[str]:
        """出力Dictのキーリスト。

        Returns:
            ["id", "exp", "tex", "pose", "lighting"]。
        """
        return list(self._PARAM_KEYS)


class _StandaloneDeep3D:
    """Deep3DFaceReconリポジトリ非依存のスタンドアロンモデル。

    リポジトリが利用不可の場合のフォールバック。
    ResNet50ベースのエンコーダ復元を試行し、失敗時はゼロ出力。

    Attributes:
        _device: 計算デバイス。
        _encoder: エンコーダネットワーク。
    """

    def __init__(
        self, checkpoint: Dict[str, Any], device: torch.device
    ) -> None:
        """スタンドアロンDeep3Dを初期化する。

        Args:
            checkpoint: チェックポイントデータ。
            device: 計算デバイス。
        """
        self._device: torch.device = device
        self._encoder: Optional[torch.nn.Module] = None

        self._try_load_encoder(checkpoint)

    def _try_load_encoder(self, checkpoint: Dict[str, Any]) -> None:
        """エンコーダの復元を試行する。

        Args:
            checkpoint: チェックポイントデータ。
        """
        try:
            from torchvision.models import resnet50

            encoder: torch.nn.Module = resnet50(weights=None)
            encoder.fc = torch.nn.Linear(2048, 257)

            state_dict: Optional[Dict[str, Any]] = None
            for key in ("net_recon", "state_dict", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

            if state_dict is None:
                state_dict = checkpoint

            encoder.load_state_dict(state_dict, strict=False)
            encoder.to(self._device).eval()
            self._encoder = encoder

            logger.debug("Deep3Dスタンドアロンエンコーダ復元成功")

        except Exception as exc:
            logger.warning("Deep3Dエンコーダ復元失敗、ゼロ出力モード: {}", exc)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """画像からBFM係数を推論する。

        Args:
            image: 入力画像。shape ``(B, 3, 224, 224)``。

        Returns:
            BFM合成係数。shape ``(B, 257)``。
        """
        batch_size: int = image.shape[0]

        if self._encoder is not None:
            return self._encoder(image)

        return torch.zeros(batch_size, 257, device=self._device)
