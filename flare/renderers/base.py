"""BaseRenderer ABC (setup/render 分離)

Section 8.2 BaseRenderer:
- setup(): セッション開始時の初期化。ソース画像の登録等。
- render(): フレームごとのレンダリング。params のみ受け取る。
- is_initialized: setup() 済みかどうか。

設計根拠 (Section 8.2):
  PIRender も FlashAvatar もソース画像はセッション開始時に 1 回設定するものであり、
  フレームごとに変わるものではない。setup() と render() を分離することで、
  Renderer の種類によらず統一的なパイプライン制御が可能になる。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch


class BaseRenderer(ABC):
    """フォトリアル顔画像レンダリングの抽象基底クラス。"""

    @abstractmethod
    def setup(
        self,
        source_image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """セッション開始時の初期化。

        - FlashAvatar: 学習済み NeRF モデルのロード
        - PIRender: ソース肖像画像の登録

        Args:
            source_image: ソース画像テンソル (1, C, H, W)。
                          FlashAvatar では不要（None でよい）。
            **kwargs: Renderer 固有の追加パラメータ。
        """
        ...

    @abstractmethod
    def render(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """フレームごとのレンダリング。

        Args:
            params: Renderer が必要とする 3DMM パラメータの辞書。
                    FlashAvatar: expr(100D), jaw_pose(6D), eyes_pose(12D), eyelids(2D)
                    PIRender: motion descriptor (BFM exp+pose+trans)

        Returns:
            レンダリングされた顔画像テンソル (1, C, H, W)。
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """setup() が完了しているかどうか。"""
        ...