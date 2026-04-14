"""Learning to Listen (L2L) モデルインターフェースモジュール。

L2L（Learning to Listen）はVQ-VAEベースのListening Head Generationモデルであり、
話者の音声特徴量と動作パラメータからリスナーの自然な頭部動作を生成する。

仕様書§4.3に基づく設計:
    - ウィンドウレベル入力: window_size = 64 フレーム
    - 入力: audio_features (B, T, D_a) + speaker_motion (B, T, D_s)
    - 出力: listener_motion (B, T, D_out)
    - VQ-VAEのコードブックサイズはconfigから設定可能

Example:
    L2LModelの使用::

        model = L2LModel(
            model_path="./checkpoints/l2l/l2l_vqvae.pth",
            device="cuda:0",
            window_size=64,
            codebook_size=256,
        )
        listener_motion = model.predict(audio_features, speaker_motion)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from flare.model_interface.base import BaseLHGModel
from flare.utils.errors import ModelLoadError


class L2LModel(BaseLHGModel):
    """Learning to Listen VQ-VAEモデル。

    音声特徴量と話者動作からリスナーの頭部動作を予測する。
    ウィンドウレベル入力（64フレーム単位）を受け取り、
    対応するリスナー動作シーケンスを出力する。

    L2Lリポジトリ: dc3ea9f/vico_challenge_baseline

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: チェックポイントファイルのパス。
        _window_size: ウィンドウフレーム数。
        _codebook_size: VQ-VAEコードブックサイズ。
        _model: ロード済みL2Lモデルインスタンス。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/l2l/l2l_vqvae.pth",
        device: str = "cuda:0",
        window_size: int = 64,
        codebook_size: int = 256,
        l2l_dir: Optional[str] = None,
    ) -> None:
        """L2LModelを初期化する。

        Args:
            model_path: VQ-VAEチェックポイントファイルのパス。
            device: 推論デバイス。例: ``"cuda:0"``, ``"cpu"``。
            window_size: ウィンドウフレーム数。L2Lデフォルトは64。
            codebook_size: VQ-VAEコードブックサイズ。デフォルトは256。
            l2l_dir: L2Lリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
                Noneの場合はインポート済みと仮定する。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._window_size = window_size
        self._codebook_size = codebook_size
        self._l2l_dir = l2l_dir
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """L2L VQ-VAEモデルをロードする。

        Raises:
            ModelLoadError: モジュールのインポートまたはモデルロードに失敗した場合。
        """
        try:
            if self._l2l_dir is not None:
                l2l_path = str(Path(self._l2l_dir).resolve())
                if l2l_path not in sys.path:
                    sys.path.insert(0, l2l_path)

            from vqvae.vqvae import VQVAE  # type: ignore[import-untyped]

            self._model = VQVAE(
                codebook_size=self._codebook_size,
            ).to(self._device)

            if self._model_path.exists():
                checkpoint = torch.load(
                    str(self._model_path),
                    map_location=self._device,
                    weights_only=False,
                )
                if "state_dict" in checkpoint:
                    self._model.load_state_dict(checkpoint["state_dict"])
                elif "model" in checkpoint:
                    self._model.load_state_dict(checkpoint["model"])
                else:
                    self._model.load_state_dict(checkpoint)

            self._model.eval()

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import L2L modules. Ensure the L2L repository "
                f"is available. Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load L2L model from {self._model_path}: {e}"
            ) from e

    def predict(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> torch.Tensor:
        """音声特徴量と話者動作からリスナー動作を予測する。

        VQ-VAEエンコーダで話者動作をコード化し、音声特徴量と組み合わせて
        デコーダでリスナー動作シーケンスを生成する。

        Args:
            audio_features: 音声特徴量テンソル。
                形状は ``(B, T, D_a)``。Tはwindow_sizeと同じフレーム数。
            speaker_motion: 話者の動作パラメータテンソル。
                形状は ``(B, T, D_s)``。

        Returns:
            予測されたリスナー動作テンソル。形状は ``(B, T, D_out)``。

        Raises:
            RuntimeError: モデル推論に失敗した場合。
        """
        audio_features = audio_features.to(self._device)
        speaker_motion = speaker_motion.to(self._device)

        with torch.no_grad():
            output = self._model(
                audio_features,
                speaker_motion,
            )

        if isinstance(output, tuple):
            listener_motion = output[0]
        elif isinstance(output, dict):
            listener_motion = output.get(
                "listener_motion", output.get("output", next(iter(output.values())))
            )
        else:
            listener_motion = output

        return listener_motion.detach()

    @property
    def requires_window(self) -> bool:
        """ウィンドウレベル入力が必要であることを返す。

        Returns:
            常に ``True``。L2Lは64フレームのウィンドウ単位で処理する。
        """
        return True

    @property
    def window_size(self) -> Optional[int]:
        """ウィンドウフレーム数を返す。

        Returns:
            ウィンドウフレーム数（デフォルト: 64）。
        """
        return self._window_size
