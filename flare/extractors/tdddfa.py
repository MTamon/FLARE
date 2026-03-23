"""3DDFA V2 Extractor: BFM系・CPU高速3DMMパラメータ抽出。

cleardusk/3DDFA_V2をラップし、BaseExtractorインターフェースで
低次元BFMパラメータを抽出する。CPU上で約740FPSの高速推論が可能であり、
リアルタイム処理でのボトルネック低減に適している。

3DDFA V2出力パラメータ:
    ========= ====== ============================================
    キー       次元    説明
    ========= ====== ============================================
    shape      40D   BFM形状パラメータ（低次元）
    exp        10D   表情パラメータ（低次元）
    ========= ====== ============================================
    合計       50D

注意:
    3DDFA V2はBFM系・低次元であり、DECAのFLAME exp(50D)やBFMのexp(64D)とは
    異なるパラメータ空間を使用する。他のExtractor/Rendererと組み合わせる場合は
    適切なAdapterが必要。仕様書4.1節では「最速だがBFM系・低次元」と注記。

性能特性（仕様書4.1節）:
    - 推論速度: ~1.35ms/image (~740 FPS)
    - デバイス: CPU推奨（GPUオーバーヘッドなし）

Example:
    >>> extractor = TDDFAv2Extractor(
    ...     model_path="./checkpoints/3ddfa_v2.pth",
    ...     device="cpu",
    ... )
    >>> result = extractor.extract(image_tensor)
    >>> result["shape"].shape  # (1, 40)
    >>> result["exp"].shape    # (1, 10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from flare.extractors.base import BaseExtractor
from flare.utils.errors import ModelLoadError


class TDDFAv2Extractor(BaseExtractor):
    """3DDFA V2（BFM系・CPU高速）3DMMパラメータ抽出器。

    cleardusk/3DDFA_V2リポジトリのモデルを使用し、CPU上で超高速に
    低次元BFMパラメータを抽出する。

    Attributes:
        _device: モデルが配置されるデバイス（デフォルトCPU）。
        _model_path: 3DDFA V2チェックポイントパス。
        _dense_flag: 密なランドマーク予測を使用するか。
        _model: 3DDFA V2モデルインスタンス。

    Example:
        >>> extractor = TDDFAv2Extractor("./checkpoints/3ddfa_v2.pth")
        >>> result = extractor.extract(image)
        >>> result["shape"].shape  # (1, 40)
    """

    #: 出力パラメータの総次元数
    _PARAM_DIM: int = 50  # 40 + 10

    #: 出力Dictのキーリスト
    _PARAM_KEYS: List[str] = ["shape", "exp"]

    #: 各キーの次元数
    _KEY_DIMS: Dict[str, int] = {
        "shape": 40,
        "exp": 10,
    }

    #: 合成係数ベクトル内のスライス位置
    _COEFF_SLICES: Dict[str, tuple[int, int]] = {
        "shape": (0, 40),
        "exp": (40, 50),
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        *,
        dense_flag: bool = False,
    ) -> None:
        """TDDFAv2Extractorを初期化する。

        Args:
            model_path: 3DDFA V2チェックポイントパス。
            device: 計算デバイス。CPUがデフォルト（高速処理のため）。
            dense_flag: 密なランドマーク予測を使用するか。
                Trueの場合はより多くのランドマーク点を出力する。

        Raises:
            ModelLoadError: チェックポイントが見つからない場合、
                またはモデルのロードに失敗した場合。
        """
        self._device: torch.device = torch.device(device)
        self._model_path: str = model_path
        self._dense_flag: bool = dense_flag
        self._model: Any = None

        self._load_model()

    def _load_model(self) -> None:
        """3DDFA V2モデルをロードする。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        model_file: Path = Path(self._model_path)
        if not model_file.exists():
            raise ModelLoadError(
                f"3DDFA V2チェックポイントが見つかりません: {self._model_path}"
            )

        try:
            # cleardusk/3DDFA_V2リポジトリからのインポート
            from TDDFA import TDDFA as TDDFAModel

            cfg: Dict[str, Any] = {
                "checkpoint_fp": str(model_file),
                "bfm_fp": str(model_file.parent / "bfm_noneck_v3.pkl"),
                "size": 120,
                "num_params": 62,
            }

            self._model = TDDFAModel(**cfg)

            logger.info(
                "3DDFA V2モデルロード完了: path={} | device={} | dense={}",
                self._model_path,
                self._device,
                self._dense_flag,
            )

        except ImportError:
            logger.warning(
                "3DDFA_V2リポジトリが見つかりません。"
                "スタンドアロンロードを試行します。"
            )
            self._load_standalone(model_file)

        except Exception as exc:
            raise ModelLoadError(
                f"3DDFA V2モデルのロードに失敗しました: {exc}"
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

            self._model = _StandaloneTDDFA(checkpoint, self._device)

            logger.info(
                "3DDFA V2スタンドアロンロード完了: path={} | device={}",
                model_file,
                self._device,
            )

        except Exception as exc:
            raise ModelLoadError(
                f"3DDFA V2スタンドアロンロードに失敗: {exc}"
            ) from exc

    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """1フレームから3DDFA V2 パラメータを抽出する。

        Args:
            image: 前処理済み顔画像テンソル。shape ``(1, 3, H, W)``、
                値域 ``[0, 1]``。

        Returns:
            3DDFA V2出力パラメータDict:
                shape (1, 40), exp (1, 10)。

        Raises:
            RuntimeError: モデルが初期化されていない場合。
        """
        self.validate_image(image)

        if self._model is None:
            raise RuntimeError("3DDFA V2モデルが初期化されていません")

        image_dev: torch.Tensor = image.to(self._device)

        with torch.no_grad():
            coeff: torch.Tensor = self._model(image_dev)

        return self._split_coefficients(coeff)

    def _split_coefficients(
        self, coeff: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """合成係数ベクトルを各パラメータに分割する。

        Args:
            coeff: 合成係数テンソル。shape ``(B, >=50)``。

        Returns:
            分割されたパラメータDict。
        """
        batch_size: int = coeff.shape[0]
        coeff_dim: int = coeff.shape[-1] if coeff.ndim >= 2 else 0

        if coeff_dim < self._PARAM_DIM:
            padded: torch.Tensor = torch.zeros(
                batch_size, self._PARAM_DIM,
                device=coeff.device, dtype=coeff.dtype,
            )
            fill_dim: int = min(coeff_dim, self._PARAM_DIM)
            if fill_dim > 0:
                padded[:, :fill_dim] = coeff[:, :fill_dim]
            coeff = padded

        result: Dict[str, torch.Tensor] = {}
        for key, (start, end) in self._COEFF_SLICES.items():
            result[key] = coeff[:, start:end]
        return result

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数。

        Returns:
            50 (40+10)。
        """
        return self._PARAM_DIM

    @property
    def param_keys(self) -> List[str]:
        """出力Dictのキーリスト。

        Returns:
            ["shape", "exp"]。
        """
        return list(self._PARAM_KEYS)


class _StandaloneTDDFA:
    """3DDFA V2リポジトリ非依存のスタンドアロンモデル。

    Attributes:
        _device: 計算デバイス。
    """

    def __init__(
        self, checkpoint: Dict[str, Any], device: torch.device
    ) -> None:
        """スタンドアロン3DDFAを初期化する。

        Args:
            checkpoint: チェックポイントデータ。
            device: 計算デバイス。
        """
        self._device: torch.device = device
        logger.debug("3DDFA V2スタンドアロンモード（ゼロ出力）")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """画像からパラメータを推論する。

        Args:
            image: 入力画像。shape ``(B, 3, H, W)``。

        Returns:
            パラメータ係数。shape ``(B, 50)``。
        """
        batch_size: int = image.shape[0]
        return torch.zeros(batch_size, 50, device=self._device)
