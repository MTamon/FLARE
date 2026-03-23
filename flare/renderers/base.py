"""BaseRenderer: フォトリアルレンダリングの抽象基底クラス。

仕様書8.2節「BaseRenderer（v2.0: setup/render分離）」に基づき、
全Rendererが実装すべきインターフェースを定義する。

setup/render分離パターンの設計根拠:
    PIRenderもFlashAvatarもソース画像はセッション開始時に1回設定するものであり、
    フレームごとに変わるものではない。``setup()`` と ``render()`` を分離することで、
    Rendererの種類によらず統一的なパイプライン制御が可能になる。

各Rendererの特性:
    ============== ======================== =====================================
    Renderer       ソース画像の扱い           フレームごとの入力
    ============== ======================== =====================================
    PIRender       初回setup()でソース肖像登録 motion descriptor (BFM exp+pose+trans)
    FlashAvatar    不要（学習済みNeRFロード）  FLAME condition 120D
    HeadGaS        不要（学習済みモデルロード）  FLAME condition
    ============== ======================== =====================================

Example:
    >>> class FlashAvatarRenderer(BaseRenderer):
    ...     # 抽象メソッド・プロパティを全て実装
    ...     ...
    >>> renderer = FlashAvatarRenderer(device=torch.device("cuda:0"))
    >>> renderer.setup(model_path="./checkpoints/flashavatar/")
    >>> output = renderer.render(params)  # (B, 3, 512, 512)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch


class BaseRenderer(ABC):
    """フォトリアルレンダリングの抽象基底クラス。

    全てのRenderer実装（PIRender, FlashAvatar, HeadGaS等）はこのクラスを継承し、
    ``setup()``・``render()``・``is_initialized`` を実装する。

    ``setup()`` でセッション初期化（ソース画像登録またはNeRFモデルロード）を行い、
    ``render()`` でフレームごとのレンダリングを実行する。``render()`` は
    ``setup()`` 完了後にのみ呼び出し可能。

    Attributes:
        _device: レンダラーが配置されるCUDAデバイス。サブクラスの ``__init__`` で設定する。

    Example:
        >>> class MyRenderer(BaseRenderer):
        ...     def __init__(self, device: torch.device) -> None:
        ...         self._device = device
        ...         self._initialized = False
        ...
        ...     def setup(self, source_image=None, **kwargs) -> None:
        ...         self._initialized = True
        ...
        ...     def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...         self.ensure_initialized()
        ...         return torch.zeros(1, 3, 512, 512, device=self._device)
        ...
        ...     @property
        ...     def is_initialized(self) -> bool:
        ...         return self._initialized
    """

    _device: torch.device

    @abstractmethod
    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """セッション開始時の初期化を行う。

        Rendererの種類に応じてソース画像の登録やモデルのロードを行う。
        パイプライン開始時に1回だけ呼び出される。

        Args:
            source_image: ソース肖像画像テンソル。shape ``(1, 3, H, W)``。
                PIRenderではソース画像の登録に使用する。
                FlashAvatarでは不要（``None`` を渡す）。
            **kwargs: Renderer固有の追加パラメータ。
                例: ``model_path``（FlashAvatarのNeRFモデルパス）。

        Example:
            >>> # PIRender: ソース画像を登録
            >>> renderer.setup(source_image=portrait_tensor)
            >>> # FlashAvatar: NeRFモデルをロード
            >>> renderer.setup(model_path="./checkpoints/flashavatar/")
        """

    @abstractmethod
    def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """フレームごとのレンダリングを実行する。

        ``setup()`` で初期化済みのRendererに対し、3DMMパラメータから
        フォトリアルな顔画像を生成する。

        Args:
            params: レンダラー固有のパラメータDict。
                PIRender: BFM exp + pose + trans 等。
                FlashAvatar: FLAME condition 120D
                (expr:100D + jaw_pose:6D + eyes_pose:12D + eyelids:2D)。

        Returns:
            レンダリングされた画像テンソル。shape ``(B, 3, H, W)``、値域 ``[0, 1]``。

        Raises:
            RuntimeError: ``setup()`` が未呼び出しの場合。
            KeyError: 必須パラメータキーが不足している場合
                （``validate_params()`` を呼び出した場合）。

        Example:
            >>> output = renderer.render({"expr": expr_tensor, "jaw_pose": jaw_tensor})
            >>> output.shape  # (1, 3, 512, 512)
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """setup()が完了済みかどうかを返す。

        Returns:
            ``setup()`` が正常に完了していれば ``True``、未呼び出しなら ``False``。
        """

    @property
    def device(self) -> torch.device:
        """レンダラーが配置されているデバイスを返す。

        サブクラスの ``__init__`` で設定された ``self._device`` を参照する。

        Returns:
            レンダラーのデバイス（例: ``torch.device("cuda:0")``）。

        Raises:
            AttributeError: サブクラスで ``_device`` が設定されていない場合。
        """
        return self._device

    @property
    def required_keys(self) -> List[str]:
        """render()に必須のパラメータキーリストを返す。

        デフォルトでは空リストを返す。サブクラスでオーバーライドして
        Renderer固有の必須キーを指定する。

        Returns:
            必須パラメータキー名のリスト。

        Example:
            >>> # FlashAvatarRenderer での実装例
            >>> @property
            ... def required_keys(self) -> List[str]:
            ...     return ["expr", "jaw_pose", "eyes_pose", "eyelids"]
        """
        return []

    def validate_params(self, params: Dict[str, torch.Tensor]) -> None:
        """レンダリングパラメータの必須キーを検証する。

        ``required_keys`` プロパティで定義されたキーが ``params`` に
        全て含まれているかを確認する。不足キーがあれば ``KeyError`` を送出する。

        Args:
            params: 検証対象のパラメータDict。

        Raises:
            KeyError: 必須キーが1つ以上不足している場合。
                エラーメッセージに不足キーの一覧を含む。

        Example:
            >>> renderer.validate_params({"expr": tensor})  # required: expr, jaw_pose
            KeyError: "必須パラメータキーが不足しています: {'jaw_pose'}"
        """
        keys_required: set[str] = set(self.required_keys)
        keys_provided: set[str] = set(params.keys())
        missing: set[str] = keys_required - keys_provided

        if missing:
            raise KeyError(
                f"必須パラメータキーが不足しています: {missing}"
            )

    def ensure_initialized(self) -> None:
        """setup()が完了済みであることを確認する。

        ``is_initialized`` が ``False`` の場合に ``RuntimeError`` を送出する。
        ``render()`` の先頭で呼び出すことを想定したヘルパーメソッド。

        Raises:
            RuntimeError: ``setup()`` が未呼び出しの場合。

        Example:
            >>> def render(self, params):
            ...     self.ensure_initialized()
            ...     # レンダリング処理...
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Rendererが初期化されていません。render()の前にsetup()を呼び出してください。"
            )
