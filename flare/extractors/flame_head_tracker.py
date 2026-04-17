"""flame-head-tracker 特徴量抽出モジュール（オフライン前処理用）。

flame-head-tracker（PeizhiYan/flame-head-tracker）を使用して、
動画から FLAME パラメータとカメラキャリブレーション（K/R/t）を抽出する。

このツールはオフライン処理専用であり、リアルタイム用途には使用できない。
FlashAvatar の公式データセット（metrical-tracker 相当）のような
精度の高いカメラ外部パラメータを取得するための前処理として使用する。

出力するカメラパラメータは FlashAvatar の .frame 形式で使用できる:
    ``opencv.K``  : (3, 3) カメラ内部パラメータ
    ``opencv.R``  : (3, 3) 回転行列
    ``opencv.t``  : (3,)  並進ベクトル

リポジトリ: https://github.com/PeizhiYan/flame-head-tracker
ライセンス: Apache 2.0
動作環境: Ubuntu Linux, Python 3.10, NVIDIA GPU (≥8GB)

TODO (SMIRK cuda128 統合時に実装):
    1. ``third_party/flame-head-tracker`` にリポジトリを追加
    2. ``_load_model()`` を実装してモデルをロード
    3. ``extract()`` で per-frame の FLAME + K/R/t を返すよう実装
    4. ``scripts/extract_flame_head_tracker.py`` 前処理スクリプトを作成

Example:
    使用予定のインターフェース::

        tracker = FlameHeadTrackerExtractor(
            config_path="./third_party/flame-head-tracker/config.yaml",
            device="cuda:0",
        )
        params = tracker.extract(image_tensor)
        # params["K"].shape == (1, 3, 3)
        # params["R"].shape == (1, 3, 3)
        # params["t"].shape == (1, 3)
        # params["exp"].shape == (1, 50)
"""

from __future__ import annotations

from typing import Any

import torch

from flare.extractors.base import BaseExtractor

_FHT_PARAM_KEYS: list[str] = [
    "shape",
    "exp",
    "pose",
    "eyelid",
    "K",
    "R",
    "t",
]

_FHT_PARAM_DIMS: dict[str, int] = {
    "shape": 300,
    "exp": 50,
    "pose": 6,
    "eyelid": 2,
    "K": 9,
    "R": 9,
    "t": 3,
}

_FHT_TOTAL_DIM: int = sum(_FHT_PARAM_DIMS.values())


class FlameHeadTrackerExtractor(BaseExtractor):
    """flame-head-tracker ベースのカメラキャリブレーション付き抽出器（スタブ）。

    SMIRK では弱透視カメラしか得られないが、flame-head-tracker は
    光度最適化（photometric optimization）により実際のカメラ内部・外部
    パラメータ（K/R/t）を推定する。FlashAvatar 学習データの精度向上に使用。

    制約: リアルタイム処理不可。前処理（オフライン）専用。
    速度: フレームあたり数秒オーダー（photometric optimization のため）。

    統合ステータス: 未実装（SMIRK cuda128 版作成後に実装予定）。
    """

    def __init__(
        self,
        config_path: str = "./third_party/flame-head-tracker/config.yaml",
        device: str = "cuda:0",
        fht_dir: str | None = None,
    ) -> None:
        """FlameHeadTrackerExtractor を初期化する（未実装）。

        Args:
            config_path: flame-head-tracker 設定ファイルパス。
            device: 推論デバイス。
            fht_dir: flame-head-tracker リポジトリルートパス。

        Raises:
            NotImplementedError: 常に発生。実装待ち。
        """
        raise NotImplementedError(
            "FlameHeadTrackerExtractor は未実装です。"
            " SMIRK cuda128 版の統合後に実装予定。"
            " リポジトリ: https://github.com/PeizhiYan/flame-head-tracker"
        )

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """1フレームの画像から FLAME + カメラパラメータを抽出する（未実装）。

        Args:
            image: 入力画像テンソル。形状は ``(1, 3, H, W)``。

        Returns:
            パラメータ辞書（K/R/t + FLAME params）。

        Raises:
            NotImplementedError: 常に発生。
        """
        raise NotImplementedError

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数を返す。"""
        return _FHT_TOTAL_DIM

    @property
    def param_keys(self) -> list[str]:
        """出力辞書のキーリストを返す。"""
        return list(_FHT_PARAM_KEYS)
