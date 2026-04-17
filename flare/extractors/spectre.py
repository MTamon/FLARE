"""SPECTRE 特徴量抽出モジュール（時系列一貫性付き FLAME 抽出）。

SPECTRE（Facial Landmark Analysis & Rendering Engine, CVPR 2023）は
動画全体の時系列一貫性を考慮した FLAME パラメータを推定する。
通常の DECA / SMIRK は per-frame 独立推定だが、SPECTRE は
Lipreading ロス + 時系列 CNN（MobileNet v2 + Temporal Conv）を使用し
特に口唇動作の時系列一貫性を改善する。

DECA をベースに構築されており、DECA 形式の FLAME パラメータを出力する。
カメラパラメータは DECA 形式（弱透視: scale, tx, ty）。

リポジトリ: https://github.com/filby89/spectre
ライセンス: academic use only（DECA ライセンスに準ずる）
動作環境: Python 3.6+, PyTorch, CUDA

特性:
    - 入力: 動画フレーム列（時系列が必要）
    - 出力: per-frame FLAME params（時系列一貫性保証）
    - 速度: per-frame DECA より遅い（時系列 CNN のため）
    - リアルタイム: 未確認（フレームバッファが必要なため困難）

TODO (SMIRK cuda128 統合時に実装):
    1. ``third_party/spectre`` にリポジトリを追加
    2. ``_load_model()`` で SPECTRE モデルをロード
    3. ``extract_sequence()`` でバッファ入力から時系列推定を実装
    4. per-frame ``extract()`` は直近 T フレームのリングバッファを使用

Example:
    使用予定のインターフェース::

        extractor = SPECTREExtractor(
            model_path="./checkpoints/spectre/spectre_model.pt",
            device="cuda:0",
        )
        # 動画フレームをバッファに追加しながら時系列推定
        params = extractor.extract_sequence(image_sequence)
        # params["exp"].shape == (T, 50)
        # params["pose"].shape == (T, 6)
"""

from __future__ import annotations

import torch

from flare.extractors.base import BaseExtractor

_SPECTRE_PARAM_KEYS: list[str] = [
    "shape",
    "exp",
    "pose",
    "cam",
]

_SPECTRE_PARAM_DIMS: dict[str, int] = {
    "shape": 100,
    "exp": 50,
    "pose": 6,
    "cam": 3,
}

_SPECTRE_TOTAL_DIM: int = sum(_SPECTRE_PARAM_DIMS.values())


class SPECTREExtractor(BaseExtractor):
    """SPECTRE ベースの時系列一貫性付き FLAME 抽出器（スタブ）。

    DECA と同じパラメータ形式で出力するが、Lipreading ロスによる
    学習のため口唇動作の時系列一貫性が向上する。

    LHG（Listening Head Generation）のように自然な頭部運動を必要とする
    アプリケーションで、DECA の per-frame ノイズを低減できる。

    制約: 時系列入力が必要（T フレームのバッファ）。
    統合ステータス: 未実装（SMIRK cuda128 版作成後に実装予定）。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/spectre/spectre_model.pt",
        device: str = "cuda:0",
        spectre_dir: str | None = None,
    ) -> None:
        """SPECTREExtractor を初期化する（未実装）。

        Args:
            model_path: SPECTRE モデルチェックポイントパス。
            device: 推論デバイス。
            spectre_dir: SPECTRE リポジトリルートパス。

        Raises:
            NotImplementedError: 常に発生。実装待ち。
        """
        raise NotImplementedError(
            "SPECTREExtractor は未実装です。"
            " SMIRK cuda128 版の統合後に実装予定。"
            " リポジトリ: https://github.com/filby89/spectre"
        )

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """1フレームの画像から SPECTRE パラメータを抽出する（未実装）。

        時系列一貫性のため、実際の実装ではリングバッファを使用する。

        Args:
            image: 入力画像テンソル。形状は ``(1, 3, H, W)``。

        Returns:
            DECA 互換パラメータ辞書（時系列一貫性保証）。

        Raises:
            NotImplementedError: 常に発生。
        """
        raise NotImplementedError

    def extract_sequence(
        self, images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """動画フレーム列から時系列一貫性のある FLAME パラメータを抽出する（未実装）。

        Args:
            images: フレーム列テンソル。形状は ``(T, 3, H, W)``。

        Returns:
            per-frame パラメータ辞書。各テンソルの形状は ``(T, D)``。

        Raises:
            NotImplementedError: 常に発生。
        """
        raise NotImplementedError

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数を返す。"""
        return _SPECTRE_TOTAL_DIM

    @property
    def param_keys(self) -> list[str]:
        """出力辞書のキーリストを返す。"""
        return list(_SPECTRE_PARAM_KEYS)
