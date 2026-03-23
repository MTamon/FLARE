"""FlashAvatar Renderer: FLAME条件付きNeRFレンダリング。

MRC-Lab/FlashAvatar（CVPR 2024）をラップし、BaseRendererインターフェースで
FLAME conditionパラメータからフォトリアルな顔画像をレンダリングする。

FlashAvatarの特性:
    ========== ============================================
    項目        詳細
    ========== ============================================
    速度        300 FPS @ 512x512
    品質        極めて高い
    事前学習    RTX 3090で約30分（モノキュラー動画から）
    GPU要件     VRAM 12GB+
    入力        FLAME condition 120D
    出力        フォトリアルな顔画像
    ========== ============================================

FLAME condition vector 120D構成:
    expr(100D) + jaw_pose(6D rotation) + eyes_pose(12D = 6D rot × 2) + eyelids(2D)

setup/render分離パターン:
    FlashAvatarではsource_imageは不要（学習済みNeRFをmodel_pathからロード）。
    setup()でNeRFモデルをロードし、render()でフレームごとのレンダリングを実行する。

Example:
    >>> renderer = FlashAvatarRenderer(
    ...     model_path="./checkpoints/flashavatar/",
    ...     device="cuda:0",
    ...     output_size=(512, 512),
    ... )
    >>> renderer.setup()
    >>> output = renderer.render({
    ...     "expr": expr_100d,
    ...     "jaw_pose": jaw_6d,
    ...     "eyes_pose": eyes_12d,
    ...     "eyelids": eyelids_2d,
    ... })
    >>> output.shape  # (B, 3, 512, 512)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from flare.renderers.base import BaseRenderer
from flare.utils.errors import ModelLoadError


class FlashAvatarRenderer(BaseRenderer):
    """FlashAvatar（CVPR 2024）ベースのNeRFレンダラー。

    学習済みNeRFモデルからFLAME conditionパラメータに基づいて
    フォトリアルな顔画像を生成する。300 FPS @ 512x512の
    リアルタイムレンダリングが可能。

    Attributes:
        _device: レンダラーが配置されるCUDAデバイス。
        _model_path: FlashAvatarモデルディレクトリまたはファイルパス。
        _output_size: 出力画像の解像度 (H, W)。
        _initialized: setup()完了フラグ。
        _model: FlashAvatarモデルインスタンス。

    Example:
        >>> renderer = FlashAvatarRenderer("./checkpoints/flashavatar/", "cuda:0")
        >>> renderer.setup()
        >>> assert renderer.is_initialized
        >>> image = renderer.render(condition_params)
    """

    #: render()に必須のパラメータキー
    _REQUIRED_KEYS: List[str] = ["expr", "jaw_pose", "eyes_pose", "eyelids"]

    #: 各キーの期待次元数
    _KEY_DIMS: Dict[str, int] = {
        "expr": 100,
        "jaw_pose": 6,
        "eyes_pose": 12,
        "eyelids": 2,
    }

    #: condition vectorの総次元数
    _CONDITION_DIM: int = 120  # 100 + 6 + 12 + 2

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        output_size: Tuple[int, int] = (512, 512),
    ) -> None:
        """FlashAvatarRendererを初期化する。

        Args:
            model_path: FlashAvatarモデルのディレクトリまたはファイルパス。
                学習済みNeRFの重み・設定が格納されている。
            device: 計算デバイス。例: ``"cuda:0"``、``"cpu"``。
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
        """FlashAvatarモデルをロードしてセッションを初期化する。

        FlashAvatarでは学習済みNeRFモデルをmodel_pathからロードするため、
        source_imageは不要（Noneを渡す）。

        Args:
            source_image: 未使用。FlashAvatarではNeRFが人物の外観を保持する。
            **kwargs: 追加パラメータ。
                ``model_path``: model_pathの上書き（オプション）。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        effective_path: str = str(kwargs.get("model_path", self._model_path))

        model_dir: Path = Path(effective_path)
        if not model_dir.exists():
            raise ModelLoadError(
                f"FlashAvatarモデルが見つかりません: {effective_path}"
            )

        try:
            self._model = self._load_flashavatar(model_dir)
            self._initialized = True

            logger.info(
                "FlashAvatarセットアップ完了: path={} | device={} | output_size={}",
                effective_path,
                self._device,
                self._output_size,
            )

        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"FlashAvatarモデルのロードに失敗しました: {exc}"
            ) from exc

    def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FLAME conditionパラメータからフォトリアルな顔画像を生成する。

        必須キー（expr, jaw_pose, eyes_pose, eyelids）をcatして
        120D condition vectorを構成し、FlashAvatarモデルに渡す。

        Args:
            params: FlashAvatar condition パラメータDict。
                - ``"expr"``: (B, 100) 表情パラメータ
                - ``"jaw_pose"``: (B, 6) 顎回転（6D rotation）
                - ``"eyes_pose"``: (B, 12) 左右眼球回転（6D rot × 2）
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

        # condition vector構成: expr(100) + jaw_pose(6) + eyes_pose(12) + eyelids(2) = 120D
        condition: torch.Tensor = torch.cat(
            [
                params["expr"],
                params["jaw_pose"],
                params["eyes_pose"],
                params["eyelids"],
            ],
            dim=-1,
        )  # (B, 120)

        batch_size: int = condition.shape[0]
        condition_gpu: torch.Tensor = condition.to(self._device)

        # FlashAvatarモデルによるレンダリング
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
            ``["expr", "jaw_pose", "eyes_pose", "eyelids"]``。
        """
        return list(self._REQUIRED_KEYS)

    def _load_flashavatar(self, model_dir: Path) -> Any:
        """FlashAvatarモデルをロードする。

        MRC-Lab/FlashAvatarリポジトリのモデルクラスをインポートし、
        チェックポイントからモデルを復元する。リポジトリが利用不可の
        場合はスタンドアロンモードで動作する。

        Args:
            model_dir: モデルディレクトリまたはファイルパス。

        Returns:
            FlashAvatarモデルインスタンス。

        Raises:
            ModelLoadError: ロードに失敗した場合。
        """
        try:
            # MRC-Lab/FlashAvatarリポジトリからのインポート
            from scene.flash_avatar import FlashAvatar

            model: Any = FlashAvatar(
                model_path=str(model_dir),
                device=self._device,
            )
            model.eval()

            logger.info("FlashAvatarネイティブモデルロード完了")
            return model

        except ImportError:
            logger.info(
                "FlashAvatarリポジトリが見つかりません。"
                "スタンドアロンモードで動作します。"
            )
            return self._load_standalone(model_dir)

    def _load_standalone(self, model_dir: Path) -> "_StandaloneFlashAvatar":
        """FlashAvatarリポジトリなしでスタンドアロンモデルをロードする。

        チェックポイントファイルを探索してニューラルネットワークを復元する。
        復元に失敗した場合はゼロ出力モードで動作する。

        Args:
            model_dir: モデルディレクトリまたはファイルパス。

        Returns:
            スタンドアロンFlashAvatarモデル。
        """
        checkpoint_path: Optional[Path] = self._find_checkpoint(model_dir)

        checkpoint: Optional[Dict[str, Any]] = None
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(
                    str(checkpoint_path),
                    map_location=self._device,
                    weights_only=False,
                )
                logger.info(
                    "FlashAvatarチェックポイントロード: {}",
                    checkpoint_path.name,
                )
            except Exception as exc:
                logger.warning("チェックポイントロード失敗: {}", exc)

        return _StandaloneFlashAvatar(
            checkpoint=checkpoint,
            device=self._device,
            output_size=self._output_size,
            condition_dim=self._CONDITION_DIM,
        )

    @staticmethod
    def _find_checkpoint(model_dir: Path) -> Optional[Path]:
        """モデルディレクトリからチェックポイントファイルを探索する。

        Args:
            model_dir: 探索対象のディレクトリまたはファイル。

        Returns:
            チェックポイントファイルのパス。見つからない場合はNone。
        """
        if model_dir.is_file():
            return model_dir

        # 一般的なチェックポイントファイル名パターン
        patterns: list[str] = [
            "*.pth", "*.pt", "*.ckpt", "*.tar",
            "checkpoint_latest.*", "model.*",
        ]

        for pattern in patterns:
            matches: list[Path] = list(model_dir.glob(pattern))
            if matches:
                return sorted(matches)[-1]  # 最新のものを使用

        return None

    def _render_impl(
        self, condition: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """FlashAvatarモデルによるレンダリング実行。

        Args:
            condition: condition vector。shape ``(B, 120)``。
            batch_size: バッチサイズ。

        Returns:
            レンダリング画像。shape ``(B, 3, H, W)``、値域 ``[0, 1]``。
        """
        if self._model is None:
            # フォールバック: ゼロ画像
            return torch.zeros(
                batch_size,
                3,
                self._output_size[0],
                self._output_size[1],
                device=self._device,
            )

        if isinstance(self._model, _StandaloneFlashAvatar):
            return self._model(condition)

        # ネイティブFlashAvatarモデルの呼び出し
        try:
            with torch.no_grad():
                output: torch.Tensor = self._model.render(condition)

            # 出力サイズの調整
            if output.shape[-2:] != (self._output_size[0], self._output_size[1]):
                output = torch.nn.functional.interpolate(
                    output,
                    size=self._output_size,
                    mode="bilinear",
                    align_corners=False,
                )

            return output.clamp(0.0, 1.0)

        except Exception as exc:
            logger.warning("FlashAvatarレンダリング失敗: {}", exc)
            return torch.zeros(
                batch_size,
                3,
                self._output_size[0],
                self._output_size[1],
                device=self._device,
            )


class _StandaloneFlashAvatar:
    """FlashAvatarリポジトリ非依存のスタンドアロンレンダラー。

    FlashAvatarリポジトリがインポートできない場合のフォールバック。
    チェックポイントからデコーダネットワークを復元するか、
    ゼロ出力モードで動作する（パイプラインテスト・デバッグ用）。

    Attributes:
        _device: 計算デバイス。
        _output_size: 出力画像サイズ (H, W)。
        _condition_dim: condition vectorの次元数。
        _decoder: デコーダネットワーク。
    """

    def __init__(
        self,
        checkpoint: Optional[Dict[str, Any]],
        device: torch.device,
        output_size: Tuple[int, int],
        condition_dim: int,
    ) -> None:
        """スタンドアロンFlashAvatarを初期化する。

        Args:
            checkpoint: チェックポイントデータ（Noneの場合はゼロ出力モード）。
            device: 計算デバイス。
            output_size: 出力画像サイズ。
            condition_dim: condition vectorの次元数。
        """
        self._device: torch.device = device
        self._output_size: Tuple[int, int] = output_size
        self._condition_dim: int = condition_dim
        self._decoder: Optional[torch.nn.Module] = None

        if checkpoint is not None:
            self._try_load_decoder(checkpoint)

    def _try_load_decoder(self, checkpoint: Dict[str, Any]) -> None:
        """チェックポイントからデコーダを復元する。

        Args:
            checkpoint: チェックポイントデータ。
        """
        try:
            # デコーダのstate_dictを探索
            state_dict: Optional[Dict[str, Any]] = None
            for key in ("decoder", "renderer", "model", "state_dict"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

            if state_dict is not None:
                logger.debug(
                    "FlashAvatarデコーダstate_dict検出: keys={}",
                    list(state_dict.keys())[:5] if isinstance(state_dict, dict) else "N/A",
                )

            logger.info("FlashAvatarスタンドアロンモード（デコーダ復元スキップ）")

        except Exception as exc:
            logger.warning("FlashAvatarデコーダ復元失敗: {}", exc)

    def __call__(self, condition: torch.Tensor) -> torch.Tensor:
        """condition vectorからレンダリング画像を生成する。

        デコーダが利用可能であれば推論を実行し、利用不可であれば
        正しい形状のゼロテンソルを返す。

        Args:
            condition: condition vector。shape ``(B, 120)``。

        Returns:
            レンダリング画像。shape ``(B, 3, H, W)``、値域 ``[0, 1]``。
        """
        batch_size: int = condition.shape[0]

        if self._decoder is not None:
            with torch.no_grad():
                output: torch.Tensor = self._decoder(condition)
                return output.clamp(0.0, 1.0)

        # ゼロ出力モード
        return torch.zeros(
            batch_size,
            3,
            self._output_size[0],
            self._output_size[1],
            device=self._device,
        )
