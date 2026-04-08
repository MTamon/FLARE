"""レンダリングの抽象基底クラス。

3DMMパラメータからフォトリアルな顔画像を生成するRendererの共通インターフェースを
定義する。全てのRenderer実装（PIRender, FlashAvatar, HeadGaS等）はBaseRendererを
継承し、抽象メソッドを実装しなければならない。

v2.0変更点:
    - ``setup()`` / ``render()`` 分離パターンを導入。
    - ソース画像はセッション開始時に ``setup()`` で1回だけ設定する。
    - ``render()`` はフレームごとのパラメータのみを受け取る。

設計根拠:
    PIRenderもFlashAvatarもソース画像はセッション開始時に1回設定するものであり、
    フレームごとに変わるものではない。setup()とrender()を分離することで、
    Rendererの種類によらず統一的なパイプライン制御が可能になる。

Example:
    BaseRendererを継承した具体クラスの実装::

        class FlashAvatarRenderer(BaseRenderer):
            def setup(self, source_image=None, **kwargs):
                # NeRFモデルをロード
                ...
                self._initialized = True

            def render(self, params):
                # condition vector (120D) から画像生成
                ...

            @property
            def is_initialized(self) -> bool:
                return self._initialized
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseRenderer(ABC):
    """フォトリアルレンダリングの抽象基底クラス。

    setup()でセッション初期化を行い、render()でフレームごとの画像生成を行う。
    Route A（BFM系: PIRender）およびRoute B（FLAME系: FlashAvatar, HeadGaS）の
    両方のRendererがこのインターフェースを実装する。

    典型的な使用フロー:
        1. ``setup(source_image=...)`` でセッションを初期化
        2. ``render(params)`` をフレームごとに呼び出し
        3. ``is_initialized`` で初期化状態を確認

    Attributes:
        is_initialized: setup()が完了しているかどうか。
    """

    @abstractmethod
    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """セッション開始時の初期化を行う。

        Rendererの種類に応じた初期化処理を実行する。
        PIRenderではソース肖像画像の登録、FlashAvatarではNeRFモデルのロード等。

        Args:
            source_image: ソース肖像画像テンソル。形状は ``(1, 3, H, W)`` または
                ``(3, H, W)``。PIRenderでは必須、FlashAvatarでは不要（None可）。
                値域は ``[0, 1]``。
            **kwargs: Renderer固有の追加パラメータ。
                例: ``model_path``、``resolution`` 等。

        Raises:
            RuntimeError: モデルのロードに失敗した場合。
            FileNotFoundError: 指定されたモデルパスが存在しない場合。
        """

    @abstractmethod
    def render(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """フレームごとのレンダリングを実行する。

        setup()で初期化済みの状態で、3DMMパラメータから画像を生成する。
        setup()が未完了の状態で呼び出された場合の動作は未定義。

        Args:
            params: レンダリングパラメータの辞書。Rendererの種類に応じた
                キーと形状を持つ。
                FlashAvatar例::

                    {
                        "expr":      Tensor(B, 100),  # FLAME表情
                        "jaw_pose":  Tensor(B, 6),    # 顎回転 (rotation_6d)
                        "eyes_pose": Tensor(B, 12),   # 眼球回転 (6D rot x2)
                        "eyelids":   Tensor(B, 2),    # 瞼パラメータ
                    }

                PIRender例::

                    {
                        "exp":  Tensor(B, 64),  # BFM表情
                        "pose": Tensor(B, 6),   # 姿勢
                    }

        Returns:
            レンダリング済み画像テンソル。形状は ``(B, 3, H, W)``。
            値域は ``[0, 1]``。

        Raises:
            RuntimeError: setup()が未完了、またはレンダリングに失敗した場合。
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """setup()が完了しているかどうかを返す。

        Returns:
            setup()が正常に完了していればTrue、未完了ならFalse。
        """
