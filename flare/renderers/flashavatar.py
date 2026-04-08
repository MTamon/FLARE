"""FlashAvatar レンダリングモジュール。

FlashAvatarは3D Gaussian Splattingベースのフォトリアル顔レンダリングエンジンである。
FLAME condition vector (120D) を入力として高品質な顔画像を生成する。

仕様書§4.2に基づく性能特性:
    - レンダリング速度: 300 FPS @ 512x512
    - 品質: 極めて高い
    - 事前学習: RTX 3090で約30分（対象人物ごと）
    - VRAM: 12GB+

FlashAvatar condition vector (120D) の構成:
    - expr: (B, 100) FLAME表情パラメータ
    - jaw_pose: (B, 6) 顎回転 (rotation_6d)
    - eyes_pose: (B, 12) 左右眼球回転 (6D rotation × 2)
    - eyelids: (B, 2) 瞼パラメータ

Example:
    FlashAvatarRendererの使用::

        renderer = FlashAvatarRenderer(
            model_path="./checkpoints/flashavatar/",
            device="cuda:0",
        )
        renderer.setup()
        output = renderer.render({
            "expr": expr_tensor,       # (1, 100)
            "jaw_pose": jaw_tensor,    # (1, 6)
            "eyes_pose": eyes_tensor,  # (1, 12)
            "eyelids": lids_tensor,    # (1, 2)
        })
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from flare.renderers.base import BaseRenderer
from flare.utils.errors import ModelLoadError, RendererNotInitializedError

_CONDITION_DIM: int = 120
"""int: FlashAvatar condition vectorの総次元数 (100+6+12+2)。"""

_CONDITION_KEYS: list[str] = ["expr", "jaw_pose", "eyes_pose", "eyelids"]
"""list[str]: condition vectorを構成するパラメータキー。"""

_CONDITION_DIMS: dict[str, int] = {
    "expr": 100,
    "jaw_pose": 6,
    "eyes_pose": 12,
    "eyelids": 2,
}
"""dict[str, int]: 各conditionキーの次元数。"""


class FlashAvatarRenderer(BaseRenderer):
    """FlashAvatar 3D Gaussianベースレンダラー。

    学習済みの3D Gaussian Splattingモデルを使用して、FLAME condition vectorから
    フォトリアルな顔画像を生成する。

    FlashAvatarリポジトリ: MingSHTM/FlashAvatar

    setup/render分離パターン:
        - ``setup()``: 学習済みGaussianモデルをロード
        - ``render(params)``: condition vector (120D) から画像生成

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: FlashAvatarモデルディレクトリのパス。
        _output_size: 出力画像サイズ [width, height]。
        _initialized: setup()が完了したかどうか。
        _model: ロード済みFlashAvatarモデルインスタンス。
        _renderer_internal: 内部Gaussianラスタライザ。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/flashavatar/",
        device: str = "cuda:0",
        output_size: Optional[list[int]] = None,
        flashavatar_dir: Optional[str] = None,
    ) -> None:
        """FlashAvatarRendererを初期化する。

        setup()が呼ばれるまでレンダリングは実行できない。

        Args:
            model_path: FlashAvatarモデルディレクトリのパス。
                学習済みGaussianパラメータとFLAMEトポロジーを含む。
            device: 推論デバイス。例: ``"cuda:0"``。
            output_size: 出力画像サイズ ``[width, height]``。
                Noneの場合は ``[512, 512]``。
            flashavatar_dir: FlashAvatarリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._output_size = output_size if output_size is not None else [512, 512]
        self._flashavatar_dir = flashavatar_dir
        self._initialized = False
        self._model: Any = None
        self._renderer_internal: Any = None

    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """FlashAvatarモデルをロードしレンダラーを初期化する。

        FlashAvatarは対象人物ごとに学習済みの3D Gaussianモデルを使用するため、
        source_imageは不要（Noneで可）。モデルパスから学習済みパラメータを
        ロードする。

        Args:
            source_image: FlashAvatarでは不要。互換性のためNoneを受け付ける。
            **kwargs: 追加パラメータ。
                - ``model_path``: モデルパスの上書き (str)。
                - ``iteration``: ロードするイテレーション番号 (int)。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        if "model_path" in kwargs:
            self._model_path = Path(str(kwargs["model_path"]))

        try:
            if self._flashavatar_dir is not None:
                fa_path = str(Path(self._flashavatar_dir).resolve())
                if fa_path not in sys.path:
                    sys.path.insert(0, fa_path)

            from scene.gaussian_model import GaussianModel  # type: ignore[import-untyped]
            from gaussian_renderer import render as gaussian_render  # type: ignore[import-untyped]

            self._model = GaussianModel(sh_degree=3)

            ckpt_path = self._model_path / "point_cloud" / "iteration_30000"
            if "iteration" in kwargs:
                ckpt_path = (
                    self._model_path / "point_cloud" / f"iteration_{kwargs['iteration']}"
                )

            ply_path = ckpt_path / "point_cloud.ply"
            if ply_path.exists():
                self._model.load_ply(str(ply_path))
            else:
                raise FileNotFoundError(f"FlashAvatar PLY not found: {ply_path}")

            self._model.to(self._device)
            self._renderer_internal = gaussian_render
            self._initialized = True

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import FlashAvatar modules. Ensure the FlashAvatar "
                f"repository and diff-gaussian-rasterization are installed. "
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to setup FlashAvatar from {self._model_path}: {e}"
            ) from e

    def render(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """FLAME condition vectorから顔画像をレンダリングする。

        120D condition vector（expr 100D + jaw_pose 6D + eyes_pose 12D +
        eyelids 2D）を受け取り、3D Gaussian Splattingでレンダリングする。

        Args:
            params: FlashAvatar condition vectorの辞書。必須キー:
                - ``"expr"``: (B, 100) FLAME表情パラメータ
                - ``"jaw_pose"``: (B, 6) 顎回転 (rotation_6d)
                - ``"eyes_pose"``: (B, 12) 眼球回転 (6D rot × 2)
                - ``"eyelids"``: (B, 2) 瞼パラメータ

        Returns:
            レンダリング済み画像テンソル。形状は ``(B, 3, H, W)``。
            値域は ``[0, 1]``。

        Raises:
            RendererNotInitializedError: setup()が未完了の場合。
            KeyError: 必要なキーがparamsに存在しない場合。
            RuntimeError: レンダリングに失敗した場合。
        """
        if not self._initialized:
            raise RendererNotInitializedError(
                "FlashAvatarRenderer.setup() must be called before render()"
            )

        for key in _CONDITION_KEYS:
            if key not in params:
                raise KeyError(
                    f"Missing required key {key!r} in params. "
                    f"Required: {_CONDITION_KEYS}"
                )

        expr = params["expr"].to(self._device)
        jaw_pose = params["jaw_pose"].to(self._device)
        eyes_pose = params["eyes_pose"].to(self._device)
        eyelids = params["eyelids"].to(self._device)

        condition = torch.cat([expr, jaw_pose, eyes_pose, eyelids], dim=-1)
        batch_size = condition.shape[0]

        rendered_images: list[torch.Tensor] = []
        for i in range(batch_size):
            cond_i = condition[i : i + 1]
            render_out = self._renderer_internal(
                self._model,
                cond_i,
                bg_color=torch.zeros(3, device=self._device),
            )
            image = render_out["render"]
            rendered_images.append(image)

        output = torch.stack(rendered_images, dim=0)

        if output.shape[-2:] != tuple(reversed(self._output_size)):
            output = torch.nn.functional.interpolate(
                output,
                size=(self._output_size[1], self._output_size[0]),
                mode="bilinear",
                align_corners=False,
            )

        return output.clamp(0.0, 1.0)

    @property
    def is_initialized(self) -> bool:
        """setup()が完了しているかどうかを返す。

        Returns:
            setup()が正常に完了していればTrue。
        """
        return self._initialized
