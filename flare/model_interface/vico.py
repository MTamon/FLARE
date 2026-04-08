"""ViCo Baseline モデルインターフェースモジュール。

ViCo（Visual Conversation）Challenge Baselineはフレームレベルの
Listening Head Generationモデルである。L2Lのウィンドウレベル処理とは異なり、
フレーム単位でリスナー動作を予測する。

仕様書§3.3に基づく設計:
    - フレームレベル入力: ウィンドウ不要 (requires_window = False)
    - 入力: audio_features (B, D_a) + speaker_motion (B, D_s)
    - 出力: listener_motion (B, D_out)
    - ViCo Challenge Baselineの推論パイプラインを使用

Example:
    ViCoModelの使用::

        model = ViCoModel(
            model_path="./checkpoints/vico_baseline.pth",
            device="cuda:0",
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


class ViCoModel(BaseLHGModel):
    """ViCo Challenge Baseline モデル。

    音声特徴量と話者動作からリスナーの頭部動作をフレーム単位で予測する。
    L2Lとは異なりウィンドウレベルの処理は不要であり、
    各フレームを独立して処理する。

    ViCo Challenge Baselineリポジトリ: dc3ea9f/vico_challenge_baseline

    Attributes:
        _device: 推論に使用するデバイス。
        _model_path: チェックポイントファイルのパス。
        _model: ロード済みViCoモデルインスタンス。
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/vico_baseline.pth",
        device: str = "cuda:0",
        vico_dir: Optional[str] = None,
    ) -> None:
        """ViCoModelを初期化する。

        Args:
            model_path: ViCoチェックポイントファイルのパス。
            device: 推論デバイス。例: ``"cuda:0"``, ``"cpu"``。
            vico_dir: ViCo Baselineリポジトリのルートディレクトリパス。
                sys.pathに追加してインポートを可能にする。
                Noneの場合はインポート済みと仮定する。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        self._vico_dir = vico_dir
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """ViCo Baselineモデルをロードする。

        Raises:
            ModelLoadError: モジュールのインポートまたはモデルロードに失敗した場合。
        """
        try:
            if self._vico_dir is not None:
                vico_path = str(Path(self._vico_dir).resolve())
                if vico_path not in sys.path:
                    sys.path.insert(0, vico_path)

            from vico.model import ViCoBaseline  # type: ignore[import-untyped]

            self._model = ViCoBaseline().to(self._device)

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
                f"Failed to import ViCo modules. Ensure the ViCo Baseline "
                f"repository is available. Error: {e}"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load ViCo model from {self._model_path}: {e}"
            ) from e

    def predict(
        self,
        audio_features: torch.Tensor,
        speaker_motion: torch.Tensor,
    ) -> torch.Tensor:
        """音声特徴量と話者動作からリスナー動作を予測する。

        フレーム単位で処理し、各フレームのリスナー動作を出力する。
        ウィンドウレベルの入力には対応しない。

        Args:
            audio_features: 音声特徴量テンソル。
                形状は ``(B, D_a)``。フレームレベル入力。
            speaker_motion: 話者の動作パラメータテンソル。
                形状は ``(B, D_s)``。

        Returns:
            予測されたリスナー動作テンソル。形状は ``(B, D_out)``。

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
        """ウィンドウレベル入力が不要であることを返す。

        Returns:
            常に ``False``。ViCoはフレーム単位で処理する。
        """
        return False

    @property
    def window_size(self) -> Optional[int]:
        """ウィンドウフレーム数を返す。

        Returns:
            ``None``。ViCoはフレーム単位処理のためウィンドウサイズ不要。
        """
        return None
