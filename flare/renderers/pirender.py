"""PIRender レンダリングモジュール。

PIRenderはNeRFベースの顔画像レンダリングエンジンであり、
BFM形式のexp+pose+transパラメータからフォトリアルな顔画像を生成する。
Route Aの標準Rendererとして使用される。

仕様書§3.2に基づく設計:
    - setup(source_image): ソース画像からモーションディスクリプタを初期化
    - render(params): BFM exp+pose+trans を受け取り顔画像を返す
    - source_imageの外見を保持しつつ、3DMMパラメータで表情・姿勢を制御

PIRenderの動作原理:
    1. setup()でソース画像のモーションディスクリプタ（ポーズ・表情の基準点）を抽出
    2. render()で入力パラメータとソースのディスクリプタからフローフィールドを推定
    3. フローフィールドでソース画像をワープし最終画像を生成

Example:
    PIRenderRendererの使用::

        renderer = PIRenderRenderer(
            model_path="./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt",
            device="cuda:0",
        )
        renderer.setup(source_image=source_tensor)
        output = renderer.render({
            "exp": exp_tensor,     # (1, 64)
            "pose": pose_tensor,   # (1, 6)
            "trans": trans_tensor,  # (1, 3)
        })
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from flare.renderers.base import BaseRenderer
from flare.utils.errors import ModelLoadError, RendererNotInitializedError

_PIRENDER_PARAM_KEYS: list[str] = ["exp", "pose", "trans"]
"""list[str]: PIRenderが受け取るパラメータキー。"""

_PIRENDER_PARAM_DIMS: dict[str, int] = {
    "exp": 64,
    "pose": 6,
    "trans": 3,
}
"""dict[str, int]: 各パラメータキーの次元数マッピング。"""


class PIRenderRenderer(BaseRenderer):
    """PIRender NeRFベースレンダラー。

    ソース画像の外見を保持しつつ、BFM 3DMMパラメータ（exp+pose+trans）から
    表情・姿勢を制御してフォトリアルな顔画像を生成する。

    PIRenderリポジトリ: RenYurui/PIRender

    setup/render分離パターン:
        - ``setup(source_image)``: ソース画像からモーションディスクリプタを抽出
        - ``render(params)``: BFMパラメータから画像生成

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: PIRenderチェックポイントのパス。
        _output_size: 出力画像サイズ [width, height]。
        _initialized: setup()が完了したかどうか。
        _model: ロード済みPIRenderモデルインスタンス。
        _source_descriptor: ソース画像のモーションディスクリプタ。
        _source_image: セットアップ時のソース画像テンソル。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt",
        device: str = "cuda:0",
        output_size: Optional[list[int]] = None,
        pirender_dir: Optional[str] = None,
    ) -> None:
        """PIRenderRendererを初期化する。

        setup()が呼ばれるまでレンダリングは実行できない。

        Args:
            model_path: PIRenderチェックポイントファイルのパス。
            device: 推論デバイス。例: ``"cuda:0"``。
            output_size: 出力画像サイズ ``[width, height]``。
                Noneの場合は ``[256, 256]``。
            pirender_dir: PIRenderリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._output_size = output_size if output_size is not None else [256, 256]
        self._pirender_dir = pirender_dir
        self._initialized = False
        self._model: Any = None
        self._source_descriptor: Optional[torch.Tensor] = None
        self._source_image: Optional[torch.Tensor] = None

    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """PIRenderモデルをロードしソース画像を初期化する。

        ソース画像からモーションディスクリプタを抽出し、レンダリング時の
        基準点として保持する。PIRenderはソース画像の外見を維持しつつ
        パラメータで動きを制御するため、setup()にソース画像が必要である。

        Args:
            source_image: ソース画像テンソル。形状は ``(1, 3, H, W)``。
                値域は ``[0, 1]``。レンダリング時の外見基準となる。
                Noneの場合はモデルロードのみ行う。
            **kwargs: 追加パラメータ。
                - ``model_path``: モデルパスの上書き (str)。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        if "model_path" in kwargs:
            self._model_path = Path(str(kwargs["model_path"]))

        try:
            if self._pirender_dir is not None:
                pi_path = str(Path(self._pirender_dir).resolve())
                if pi_path not in sys.path:
                    sys.path.insert(0, pi_path)

            from models.face_model import FaceGenerator  # type: ignore[import-untyped]

            self._model = FaceGenerator().to(self._device)

            if self._model_path.exists():
                checkpoint = torch.load(
                    str(self._model_path),
                    map_location=self._device,
                    weights_only=False,
                )
                if "gen" in checkpoint:
                    self._model.load_state_dict(checkpoint["gen"])
                elif "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                else:
                    self._model.load_state_dict(checkpoint)

            self._model.eval()

            if source_image is not None:
                self._source_image = source_image.to(self._device)
                with torch.no_grad():
                    self._source_descriptor = self._model.extract_descriptor(
                        self._source_image
                    )

            self._initialized = True

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import PIRender modules. Ensure the PIRender "
                f"repository is available. Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to setup PIRender from {self._model_path}: {e}"
            ) from e

    def render(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """BFMパラメータから顔画像をレンダリングする。

        ソース画像の外見を保持しつつ、入力されたexp+pose+transパラメータに
        基づいて表情と姿勢を変化させた画像を生成する。

        Args:
            params: BFMパラメータの辞書。必須キー:
                - ``"exp"``: (B, 64) BFM表情係数
                - ``"pose"``: (B, 6) 姿勢（rotation 3D + translation 3D）
                - ``"trans"``: (B, 3) 平行移動パラメータ

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
                "PIRenderRenderer.setup() must be called before render()"
            )

        for key in _PIRENDER_PARAM_KEYS:
            if key not in params:
                raise KeyError(
                    f"Missing required key {key!r} in params. "
                    f"Required: {_PIRENDER_PARAM_KEYS}"
                )

        exp = params["exp"].to(self._device)
        pose = params["pose"].to(self._device)
        trans = params["trans"].to(self._device)

        # PIRenderはモーションディスクリプタと3DMMパラメータを組み合わせて
        # フローフィールドを推定し画像を生成する
        motion_params = torch.cat([exp, pose, trans], dim=-1)
        batch_size = motion_params.shape[0]

        rendered_images: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(batch_size):
                motion_i = motion_params[i : i + 1]
                output = self._model(
                    self._source_image,
                    motion_i,
                    self._source_descriptor,
                )
                if isinstance(output, dict):
                    image = output.get("fake_image", output.get("output", next(iter(output.values()))))
                elif isinstance(output, tuple):
                    image = output[0]
                else:
                    image = output
                rendered_images.append(image)

        result = torch.cat(rendered_images, dim=0)

        if result.shape[-2:] != (self._output_size[1], self._output_size[0]):
            result = torch.nn.functional.interpolate(
                result,
                size=(self._output_size[1], self._output_size[0]),
                mode="bilinear",
                align_corners=False,
            )

        return result.clamp(0.0, 1.0)

    @property
    def is_initialized(self) -> bool:
        """setup()が完了しているかどうかを返す。

        Returns:
            setup()が正常に完了していればTrue。
        """
        return self._initialized
