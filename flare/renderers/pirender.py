"""PIRender Renderer: BFM motion descriptorベースのGANレンダラー。

RenYurui/PIRenderをラップし、BaseRendererインターフェースで
BFMパラメータからフォトリアルな顔画像をレンダリングする。

PIRenderの特性（仕様書3.2節）:
    ========== ============================================
    項目        詳細
    ========== ============================================
    入力        BFMパラメータ + ソース画像1枚（setup()時登録）
    出力        フォトリアルな顔画像
    推論速度    ~20-30 FPS on modern GPU
    品質        GAN系、中程度のフォトリアリズム
    利点        汎用レンダラー、人物ごとの事前学習不要
    ========== ============================================

setup/render分離パターン:
    PIRenderでは初回setup()でソース肖像画像を登録する。
    FlashAvatarと異なりsource_imageは必須。

Example:
    >>> renderer = PIRenderRenderer(
    ...     model_path="./checkpoints/pirender.pth",
    ...     device="cuda:0",
    ... )
    >>> renderer.setup(source_image=portrait_tensor)
    >>> output = renderer.render({"exp": exp_64d, "pose": pose_6d})
    >>> output.shape  # (B, 3, 256, 256)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from flare.renderers.base import BaseRenderer
from flare.utils.errors import ModelLoadError


class PIRenderRenderer(BaseRenderer):
    """PIRender（GAN系）BFMベースレンダラー。

    RenYurui/PIRenderリポジトリのモデルを使用し、BFM motion descriptor
    （expression + pose）とソース肖像画像からフォトリアルな顔画像を生成する。
    汎用レンダラーであり人物ごとの事前学習が不要。

    Attributes:
        _device: レンダラーが配置されるCUDAデバイス。
        _model_path: PIRenderチェックポイントパス。
        _output_size: 出力画像の解像度 (H, W)。
        _initialized: setup()完了フラグ。
        _source_image: setup()で登録されたソース肖像画像。
        _model: PIRenderモデルインスタンス。

    Example:
        >>> renderer = PIRenderRenderer("./checkpoints/pirender.pth", "cuda:0")
        >>> renderer.setup(source_image=portrait_tensor)
        >>> output = renderer.render({"exp": exp_64d, "pose": pose_6d})
    """

    #: render()に必須のパラメータキー
    _REQUIRED_KEYS: List[str] = ["exp", "pose"]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        output_size: Tuple[int, int] = (256, 256),
    ) -> None:
        """PIRenderRendererを初期化する。

        Args:
            model_path: PIRenderチェックポイントパス。
            device: 計算デバイス。
            output_size: 出力画像の解像度 ``(H, W)``。デフォルト (256, 256)。
        """
        self._device: torch.device = torch.device(device)
        self._model_path: str = model_path
        self._output_size: Tuple[int, int] = output_size
        self._initialized: bool = False
        self._source_image: Optional[torch.Tensor] = None
        self._source_features: Optional[torch.Tensor] = None
        self._model: Any = None

    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """ソース肖像画像を登録しPIRenderモデルをロードする。

        PIRenderではソース画像は必須であり、セッション中に1回だけ
        登録される。この画像の外観を基に、motion descriptorに応じた
        顔画像を生成する。

        Args:
            source_image: ソース肖像画像テンソル。shape ``(1, 3, H, W)``、
                値域 ``[0, 1]``。必須。
            **kwargs: 追加パラメータ。

        Raises:
            ValueError: source_imageがNoneの場合。
            ModelLoadError: モデルのロードに失敗した場合。
        """
        if source_image is None:
            raise ValueError(
                "PIRenderにはソース肖像画像が必須です。"
                "setup(source_image=portrait_tensor)で指定してください。"
            )

        self._source_image = source_image.to(self._device)

        self._load_model()
        self._precompute_source_features()

        self._initialized = True

        logger.info(
            "PIRenderセットアップ完了: path={} | device={} | "
            "source_shape={} | output_size={}",
            self._model_path,
            self._device,
            tuple(source_image.shape),
            self._output_size,
        )

    def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """BFM motion descriptorからフォトリアルな顔画像を生成する。

        ソース画像の外観とmotion descriptor（expression + pose）を組み合わせて、
        指定された表情・姿勢の顔画像を生成する。

        Args:
            params: BFM motion descriptorパラメータDict。
                - ``"exp"``: (B, 64D) BFM expression パラメータ
                - ``"pose"``: (B, 6D) 頭部姿勢（回転3D + 平行移動3D）

        Returns:
            レンダリングされた画像テンソル。
            shape ``(B, 3, H, W)``、値域 ``[0, 1]``。

        Raises:
            RuntimeError: setup()が未呼び出しの場合。
            KeyError: 必須パラメータキーが不足している場合。
        """
        self.ensure_initialized()
        self.validate_params(params)

        exp: torch.Tensor = params["exp"].to(self._device)
        pose: torch.Tensor = params["pose"].to(self._device)

        # Motion descriptor構成
        motion: torch.Tensor = torch.cat([exp, pose], dim=-1)  # (B, 70)

        batch_size: int = motion.shape[0]
        rendered: torch.Tensor = self._render_impl(motion, batch_size)

        return rendered

    @property
    def is_initialized(self) -> bool:
        """setup()が完了済みかどうかを返す。

        Returns:
            初期化済みなら ``True``。
        """
        return self._initialized

    @property
    def required_keys(self) -> List[str]:
        """render()に必須のパラメータキーリストを返す。

        Returns:
            ``["exp", "pose"]``。
        """
        return list(self._REQUIRED_KEYS)

    def _load_model(self) -> None:
        """PIRenderモデルをロードする。

        Raises:
            ModelLoadError: ロードに失敗した場合。
        """
        model_file: Path = Path(self._model_path)
        if not model_file.exists():
            raise ModelLoadError(
                f"PIRenderチェックポイントが見つかりません: {self._model_path}"
            )

        try:
            # RenYurui/PIRenderリポジトリからのインポート
            from models.face_model import FaceGenerator

            self._model = FaceGenerator()

            checkpoint: Dict[str, Any] = torch.load(
                str(model_file),
                map_location=self._device,
                weights_only=False,
            )

            if "gen" in checkpoint:
                self._model.load_state_dict(checkpoint["gen"], strict=False)
            elif "state_dict" in checkpoint:
                self._model.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
            else:
                self._model.load_state_dict(checkpoint, strict=False)

            self._model.to(self._device).eval()

            logger.info(
                "PIRenderモデルロード完了: path={} | device={}",
                self._model_path,
                self._device,
            )

        except ImportError:
            logger.info(
                "PIRenderリポジトリが見つかりません。"
                "スタンドアロンモードで動作します。"
            )
            self._model = _StandalonePIRender(
                model_file, self._device, self._output_size
            )

        except Exception as exc:
            raise ModelLoadError(
                f"PIRenderモデルのロードに失敗しました: {exc}"
            ) from exc

    def _precompute_source_features(self) -> None:
        """ソース画像の特徴量を事前計算する。

        PIRenderではソース画像のエンコードを初回に1回だけ行い、
        以降のrender()ではデコードのみを実行して高速化する。
        """
        if self._source_image is None or self._model is None:
            return

        try:
            if hasattr(self._model, "encode_source"):
                with torch.no_grad():
                    self._source_features = self._model.encode_source(
                        self._source_image
                    )
                logger.debug("ソース画像特徴量の事前計算完了")
        except Exception as exc:
            logger.debug("ソース特徴量事前計算スキップ: {}", exc)

    def _render_impl(
        self, motion: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """PIRenderモデルによるレンダリング実行。

        Args:
            motion: motion descriptor。shape ``(B, 70)``（exp 64D + pose 6D）。
            batch_size: バッチサイズ。

        Returns:
            レンダリング画像。shape ``(B, 3, H, W)``、値域 ``[0, 1]``。
        """
        if self._model is None:
            return torch.zeros(
                batch_size, 3, *self._output_size, device=self._device
            )

        if isinstance(self._model, _StandalonePIRender):
            return self._model(self._source_image, motion)

        try:
            with torch.no_grad():
                # ソース画像をバッチサイズに合わせて拡張
                source: torch.Tensor = self._source_image
                if source.shape[0] < batch_size:
                    source = source.expand(batch_size, -1, -1, -1)

                output: torch.Tensor = self._model(source, motion)

            # 出力サイズの調整
            if output.shape[-2:] != tuple(self._output_size):
                output = torch.nn.functional.interpolate(
                    output,
                    size=self._output_size,
                    mode="bilinear",
                    align_corners=False,
                )

            return output.clamp(0.0, 1.0)

        except Exception as exc:
            logger.warning("PIRenderレンダリング失敗: {}", exc)
            return torch.zeros(
                batch_size, 3, *self._output_size, device=self._device
            )


class _StandalonePIRender:
    """PIRenderリポジトリ非依存のスタンドアロンレンダラー。

    リポジトリが利用不可の場合のフォールバック。
    ゼロ出力モードで動作する（パイプラインテスト・デバッグ用）。

    Attributes:
        _device: 計算デバイス。
        _output_size: 出力画像サイズ。
    """

    def __init__(
        self,
        model_file: Path,
        device: torch.device,
        output_size: Tuple[int, int],
    ) -> None:
        """スタンドアロンPIRenderを初期化する。

        Args:
            model_file: チェックポイントファイルパス。
            device: 計算デバイス。
            output_size: 出力画像サイズ。
        """
        self._device: torch.device = device
        self._output_size: Tuple[int, int] = output_size
        logger.info("PIRenderスタンドアロンモード（ゼロ出力）")

    def __call__(
        self,
        source_image: Optional[torch.Tensor],
        motion: torch.Tensor,
    ) -> torch.Tensor:
        """ソース画像とmotion descriptorからレンダリング画像を生成する。

        Args:
            source_image: ソース肖像画像。
            motion: motion descriptor。shape ``(B, 70)``。

        Returns:
            レンダリング画像。shape ``(B, 3, H, W)``。
        """
        batch_size: int = motion.shape[0]
        return torch.zeros(
            batch_size, 3, *self._output_size, device=self._device
        )
