"""run_converter_chain() の統合テスト。

converter_chain に MediaPipe 補完フラグ付き Adapter が含まれる場合に、
FaceDetector の eyes_pose / eyelids が正しく注入されることを検証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from flare.converters.deca_to_flame import DECAToFlameAdapter
from flare.converters.identity import IdentityAdapter
from flare.converters.smirk_to_flashavatar import SmirkToFlashAvatarAdapter
from flare.pipeline.converter_runner import run_converter_chain


class TestRunConverterChainEmpty:
    """空チェーンの挙動テスト。"""

    def test_empty_chain_returns_input(self) -> None:
        """空チェーンは入力をそのまま返すこと。"""
        params = {"exp": torch.zeros(1, 50)}
        result = run_converter_chain([], params)
        assert result is params


class TestRunConverterChainNoSupplement:
    """MediaPipe 補完が必要ない場合の挙動テスト。"""

    def test_deca_chain_runs_without_face_detector(self) -> None:
        """フラグ OFF の Adapter は face_detector なしで実行できること。"""
        adapter = DECAToFlameAdapter()  # use_mediapipe_supplement=False (default)
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        result = run_converter_chain([adapter], source)
        assert result["expr"].shape == (1, 100)

    def test_identity_then_deca(self) -> None:
        """Identity → DECA の 2 段チェーンが動くこと (補完なし)。"""
        chain = [IdentityAdapter(), DECAToFlameAdapter()]
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        result = run_converter_chain(chain, source)
        assert result["expr"].shape == (1, 100)


class TestRunConverterChainWithSupplement:
    """MediaPipe 補完フラグ ON 時の挙動テスト。"""

    def _make_face_detector_mock(self) -> MagicMock:
        """detect_eye_pose() が決め打ち値を返す MagicMock。"""
        eyes_pose = torch.full((1, 12), 0.3)
        eyelids = torch.tensor([[0.4, 0.6]])
        mock = MagicMock()
        mock.detect_eye_pose.return_value = (eyes_pose, eyelids)
        return mock

    def test_deca_supplement_injects_eyes_pose(self) -> None:
        """フラグ ON のとき MediaPipe 由来 eyes_pose が adapter に注入されること。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        face_detector = self._make_face_detector_mock()
        result = run_converter_chain(
            [adapter],
            source,
            frame=MagicMock(),
            bbox=(0, 0, 100, 100),
            face_detector=face_detector,
        )
        assert torch.allclose(result["eyes_pose"], torch.full((1, 12), 0.3))
        assert torch.allclose(result["eyelids"], torch.tensor([[0.4, 0.6]]))
        face_detector.detect_eye_pose.assert_called_once()

    def test_smirk_supplement_injects_eyes_pose_only(self) -> None:
        """SMIRK ルートでは eyes_pose のみ注入され、eyelids はネイティブが優先。"""
        adapter = SmirkToFlashAvatarAdapter(use_mediapipe_supplement=True)
        source = {
            "shape": torch.zeros(1, 300),
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
            "cam": torch.zeros(1, 3),
            "eyelid": torch.tensor([[0.9, 0.1]]),  # SMIRK ネイティブ
        }
        face_detector = self._make_face_detector_mock()
        result = run_converter_chain(
            [adapter],
            source,
            frame=MagicMock(),
            bbox=(0, 0, 100, 100),
            face_detector=face_detector,
        )
        # eyes_pose は MediaPipe 注入値
        assert torch.allclose(result["eyes_pose"], torch.full((1, 12), 0.3))
        # eyelids は SMIRK ネイティブが優先される (run_converter_chain は setdefault
        # を使うため source の eyelid は既存。また SmirkToFlashAvatarAdapter は
        # eyelid キーを直接見るので "eyelids" 注入の有無に関わらず eyelid を使う)
        assert torch.allclose(result["eyelids"], torch.tensor([[0.9, 0.1]]))

    def test_supplement_does_not_overwrite_existing_eyes_pose(self) -> None:
        """source_params に既に eyes_pose があれば MediaPipe 値で上書きしないこと。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        existing_eyes = torch.full((1, 12), 0.7)
        source = {
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
            "eyes_pose": existing_eyes,
        }
        face_detector = self._make_face_detector_mock()
        result = run_converter_chain(
            [adapter],
            source,
            frame=MagicMock(),
            bbox=(0, 0, 100, 100),
            face_detector=face_detector,
        )
        assert torch.allclose(result["eyes_pose"], existing_eyes)

    def test_supplement_does_not_mutate_source_params(self) -> None:
        """注入時に元の source_params を改変しないこと (shallow copy)。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        original_keys = set(source.keys())
        face_detector = self._make_face_detector_mock()
        run_converter_chain(
            [adapter],
            source,
            frame=MagicMock(),
            bbox=(0, 0, 100, 100),
            face_detector=face_detector,
        )
        assert set(source.keys()) == original_keys


class TestRunConverterChainErrors:
    """エラー条件の挙動テスト。"""

    def test_missing_face_detector_raises_when_supplement_on(self) -> None:
        """フラグ ON で face_detector を渡さないと ValueError。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        with pytest.raises(ValueError, match="use_mediapipe_supplement"):
            run_converter_chain([adapter], source)

    def test_missing_bbox_raises_when_supplement_on(self) -> None:
        """フラグ ON で bbox を渡さないと ValueError。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        with pytest.raises(ValueError, match="use_mediapipe_supplement"):
            run_converter_chain(
                [adapter],
                source,
                frame=MagicMock(),
                face_detector=MagicMock(),
            )

    def test_missing_frame_raises_when_supplement_on(self) -> None:
        """フラグ ON で frame を渡さないと ValueError。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        with pytest.raises(ValueError, match="use_mediapipe_supplement"):
            run_converter_chain(
                [adapter],
                source,
                bbox=(0, 0, 100, 100),
                face_detector=MagicMock(),
            )

    def test_only_head_flag_is_inspected(self) -> None:
        """先頭 Adapter のフラグのみが見られること (チェーン途中の Adapter フラグは無視)。"""
        # 先頭が IdentityAdapter (フラグなし) なので、後続の Adapter が
        # use_mediapipe_supplement=True を持っていても MediaPipe は呼ばれない。
        chain = [
            IdentityAdapter(),
            DECAToFlameAdapter(use_mediapipe_supplement=True),
        ]
        source = {"exp": torch.zeros(1, 50), "pose": torch.zeros(1, 6)}
        # face_detector を渡さなくても ValueError にならないこと。
        result = run_converter_chain(chain, source)
        assert result["expr"].shape == (1, 100)
