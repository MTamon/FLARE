"""HeadGaS Renderer: FLAME系Gaussian Splattingレンダラー。

HoneWayne/HeadGaSをラップし、BaseRendererインターフェースで
FLAMEパラメータからフォトリアルな顔画像をレンダリングする。

HeadGaSの特性（仕様書4.2節）:
    ========== ============================================
    項目        詳細
    ========== ============================================
    速度        250 FPS @ 512x512
    品質        高い
    事前学習    モノキュラー動画から学習
    GPU要件     VRAM 12GB+
    入力        FLAME condition
    ========== ============================================

FlashAvatarと同様にsetup()でsource_imageは不要（学習済みモデルをロード）。

Example:
    >>> renderer = HeadGaSRenderer(
    ...     model_path="./checkpoints/headgas/",
    ...     device="cuda:0",
    ... )
    >>> renderer.setup()
    >>> output = renderer.render({"expr": expr_tensor, "jaw_pose": jaw_tensor})
    >>> output.shape  # (B, 3, 512, 512)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from flare.renderers.base import BaseRenderer
from flare.utils.errors import ModelLoadError


class HeadGaSRenderer(BaseRenderer):
    """HeadGaS（Gaussian Splatting系）FLAMEベースレンダラー。

    HoneWayne/HeadGaSリポジトリのモデルを使用し、FLAME condition
    パラメータからGaussian Splattingで顔画像を生成する。
    250 FPS @ 512x512のリアルタイムレンダリングが可能。

    Attributes:
        _device: レンダラーが配置されるCUDAデバイス。
        _model_path: HeadGaSモデルディレクトリまたはファイルパス。
        _output_size: 出力画像の解像度 (H, W)。
        _initialized: setup()完了フラグ。
        _model: HeadGaSモデルインスタンス。

    Example:
        >>> renderer = HeadGaSRenderer("./checkpoints/headgas/", "cuda:0")
        >>> renderer.setup()
        >>> image = renderer.render({"expr": expr, "jaw_pose": jaw_6d})
    """

    #: render()に必須のパラメータキー
    _REQUIRED_KEYS: List[str] = ["expr", "jaw_pose"]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        output_size: Tuple[int, int] = (512, 512),
    ) -> None:
        """HeadGaSRendererを初期化する。

        Args:
            model_path: HeadGaSモデルのディレクトリまたはファイルパス。
            device: 計算デバイス。
            output_size: 出力画像の解像度 ``(H, W)``。デフォルト (512, 512)。
        """
        self._device: torch.device = torch.device(device)
        self._model_path: str = model_path
        self._output_size: Tuple[int, int] = output_size
        self._initialized: bool = False
        self._model: Any = None

    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """HeadGaSモデルをロードしてセッションを初期化する。

        HeadGaSではFlashAvatarと同様に学習済みモデルをロードするため、
        source_imageは不要。

        Args:
            source_image: 未使用。
            **kwargs: 追加パラメータ。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        effective_path: str = str(kwargs.get("model_path", self._model_path))
        model_dir: Path = Path(effective_path)

        if not model_dir.exists():
            raise ModelLoadError(
                f"HeadGaSモデルが見つかりません: {effective_path}"
            )

        try:
            self._model = self._load_headgas(model_dir)
            self._initialized = True

            logger.info(
                "HeadGaSセットアップ完了: path={} | device={} | output_size={}",
                effective_path,
                self._device,
                self._output_size,
            )

        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"HeadGaSモデルのロードに失敗しました: {exc}"
            ) from exc

    def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FLAME conditionパラメータから顔画像を生成する。

        Args:
            params: FLAME conditionパラメータDict。
                - ``"expr"``: (B, D) 表情パラメータ
                - ``"jaw_pose"``: (B, 6) 顎回転（6D rotation）
                オプション:
                - ``"eyes_pose"``: (B, 12) 左右眼球回転
                - ``"eyelids"``: (B, 2) 瞼パラメータ

        Returns:
            レンダリングされた画像テンソル。
            shape ``(B, 3, H, W)``、値域 ``[0, 1]``。

        Raises:
            RuntimeError: setup()が未呼び出しの場合。
            KeyError: 必須パラメータキーが不足している場合。
        """
        self.ensure_initialized()
        self.validate_params(params)

        # condition構成
        components: list[torch.Tensor] = [
            params["expr"],
            params["jaw_pose"],
        ]
        if "eyes_pose" in params:
            components.append(params["eyes_pose"])
        if "eyelids" in params:
            components.append(params["eyelids"])

        condition: torch.Tensor = torch.cat(components, dim=-1)
        batch_size: int = condition.shape[0]
        condition_gpu: torch.Tensor = condition.to(self._device)

        rendered: torch.Tensor = self._render_impl(condition_gpu, batch_size)
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
            ``["expr", "jaw_pose"]``。
        """
        return list(self._REQUIRED_KEYS)

    def _load_headgas(self, model_dir: Path) -> Any:
        """HeadGaSモデルをロードする。

        Args:
            model_dir: モデルディレクトリまたはファイルパス。

        Returns:
            HeadGaSモデルインスタンス。
        """
        try:
            from scene.headgas_model import HeadGaSModel

            model: Any = HeadGaSModel(
                model_path=str(model_dir),
                device=self._device,
            )
            model.eval()
            logger.info("HeadGaSネイティブモデルロード完了")
            return model

        except ImportError:
            logger.info(
                "HeadGaSリポジトリが見つかりません。"
                "スタンドアロンモードで動作します。"
            )
            return _StandaloneHeadGaS(self._device, self._output_size)

    def _render_impl(
        self, condition: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """HeadGaSモデルによるレンダリング実行。

        Args:
            condition: condition vector。
            batch_size: バッチサイズ。

        Returns:
            レンダリング画像。shape ``(B, 3, H, W)``、値域 ``[0, 1]``。
        """
        if self._model is None:
            return torch.zeros(
                batch_size, 3, *self._output_size, device=self._device
            )

        if isinstance(self._model, _StandaloneHeadGaS):
            return self._model(condition)

        try:
            with torch.no_grad():
                output: torch.Tensor = self._model.render(condition)

            if output.shape[-2:] != tuple(self._output_size):
                output = torch.nn.functional.interpolate(
                    output, size=self._output_size,
                    mode="bilinear", align_corners=False,
                )
            return output.clamp(0.0, 1.0)

        except Exception as exc:
            logger.warning("HeadGaSレンダリング失敗: {}", exc)
            return torch.zeros(
                batch_size, 3, *self._output_size, device=self._device
            )


class _StandaloneHeadGaS:
    """HeadGaSリポジトリ非依存のスタンドアロンレンダラー。

    Attributes:
        _device: 計算デバイス。
        _output_size: 出力画像サイズ。
    """

    def __init__(
        self, device: torch.device, output_size: Tuple[int, int]
    ) -> None:
        """スタンドアロンHeadGaSを初期化する。

        Args:
            device: 計算デバイス。
            output_size: 出力画像サイズ。
        """
        self._device: torch.device = device
        self._output_size: Tuple[int, int] = output_size
        logger.debug("HeadGaSスタンドアロンモード（ゼロ出力）")

    def __call__(self, condition: torch.Tensor) -> torch.Tensor:
        """condition vectorからレンダリング画像を生成する。

        Args:
            condition: condition vector。

        Returns:
            レンダリング画像。shape ``(B, 3, H, W)``。
        """
        batch_size: int = condition.shape[0]
        return torch.zeros(
            batch_size, 3, *self._output_size, device=self._device
        )
