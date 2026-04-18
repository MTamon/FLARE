"""DECAToFlameAdapter / SmirkToFlashAvatarAdapter の変換ロジックテスト。

仕様書§8.2のゼロパディング変換を検証する:
    - exp 50D → expr 100D
    - jaw_pose axis-angle(3D) → rotation_6d(6D)
    - eyes_pose identity_6d × 2 (12D)
    - eyelids ゼロ埋め (2D)  [DECA] / SMIRK ネイティブ値パススルー [SMIRK]
"""

from __future__ import annotations

import torch

from flare.converters.deca_to_flame import DECAToFlameAdapter
from flare.converters.smirk_to_flashavatar import SmirkToFlashAvatarAdapter


class TestDECAToFlameShapes:
    """変換後テンソルの形状テスト。"""

    def test_expr_shape(self, dummy_deca_output: dict[str, torch.Tensor]) -> None:
        """exp 50D → expr 100D のパディング後の形状が (B, 100) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["expr"].shape == (2, 100)

    def test_jaw_pose_shape(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """jaw_pose の rotation_6d 変換後の形状が (B, 6) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["jaw_pose"].shape == (2, 6)

    def test_eyes_pose_shape(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """eyes_pose の形状が (B, 12) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["eyes_pose"].shape == (2, 12)

    def test_eyelids_shape(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """eyelids の形状が (B, 2) であること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert result["eyelids"].shape == (2, 2)


class TestDECAToFlameValues:
    """変換後テンソルの値テスト。"""

    def test_expr_first_50_preserved(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """expr の最初の 50D が元の exp と一致すること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert torch.allclose(
            result["expr"][:, :50], dummy_deca_output["exp"]
        )

    def test_expr_last_50_zero(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """expr の最後の 50D がゼロであること（ゼロパディング）。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert torch.allclose(
            result["expr"][:, 50:], torch.zeros(2, 50)
        )

    def test_eyelids_zero(
        self, dummy_deca_output: dict[str, torch.Tensor]
    ) -> None:
        """eyelids が全てゼロであること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert(dummy_deca_output)
        assert torch.allclose(result["eyelids"], torch.zeros(2, 2))

    def test_eyes_pose_is_identity_6d_repeated(self) -> None:
        """eyes_pose が単位回転行列の6D表現 × 2 であること。"""
        adapter = DECAToFlameAdapter()
        source = {
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
        }
        result = adapter.convert(source)
        eyes = result["eyes_pose"]

        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(eyes, expected, atol=1e-6)

    def test_jaw_pose_identity_for_zero_input(self) -> None:
        """jaw_pose がゼロ入力で単位回転行列の6D表現になること。"""
        adapter = DECAToFlameAdapter()
        source = {
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
        }
        result = adapter.convert(source)
        jaw = result["jaw_pose"]

        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        assert torch.allclose(jaw, identity_6d, atol=1e-6)


class TestDECAToFlameAdapterProperties:
    """Adapter プロパティテスト。"""

    def test_source_format(self) -> None:
        """source_format が 'deca' であること。"""
        adapter = DECAToFlameAdapter()
        assert adapter.source_format == "deca"

    def test_target_format(self) -> None:
        """target_format が 'flash_avatar' であること。"""
        adapter = DECAToFlameAdapter()
        assert adapter.target_format == "flash_avatar"

    def test_missing_key_raises(self) -> None:
        """必要なキーが欠損している場合にKeyErrorが発生すること。"""
        adapter = DECAToFlameAdapter()
        with pytest.raises(KeyError):
            adapter.convert({"exp": torch.zeros(1, 50)})

    def test_default_flag_is_false(self) -> None:
        """use_mediapipe_supplement のデフォルトが False であること。"""
        adapter = DECAToFlameAdapter()
        assert adapter.use_mediapipe_supplement is False


class TestDECAToFlameMediaPipeSupplement:
    """MediaPipe 補完フラグの挙動テスト。

    DECA本来は eyes_pose / eyelids を出力しない。フラグONのときのみ
    呼び出し側が事前計算した値を注入できる。
    """

    def _make_source(self, with_supplement: bool = False) -> dict[str, torch.Tensor]:
        source: dict[str, torch.Tensor] = {
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
        }
        if with_supplement:
            source["eyes_pose"] = torch.full((1, 12), 0.5)
            source["eyelids"] = torch.tensor([[0.7, 0.3]])
        return source

    def test_flag_off_ignores_injected_eyes_pose(self) -> None:
        """フラグ OFF のとき source 提供の eyes_pose は無視され identity になること。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=False)
        result = adapter.convert(self._make_source(with_supplement=True))
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(result["eyes_pose"], expected, atol=1e-6)

    def test_flag_off_ignores_injected_eyelids(self) -> None:
        """フラグ OFF のとき source 提供の eyelids は無視されゼロになること。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=False)
        result = adapter.convert(self._make_source(with_supplement=True))
        assert torch.allclose(result["eyelids"], torch.zeros(1, 2))

    def test_flag_on_uses_injected_eyes_pose(self) -> None:
        """フラグ ON で eyes_pose 注入があれば優先採用されること。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source = self._make_source(with_supplement=True)
        result = adapter.convert(source)
        assert torch.allclose(result["eyes_pose"], source["eyes_pose"])

    def test_flag_on_uses_injected_eyelids(self) -> None:
        """フラグ ON で eyelids 注入があれば優先採用されること。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source = self._make_source(with_supplement=True)
        result = adapter.convert(source)
        assert torch.allclose(result["eyelids"], source["eyelids"])

    def test_flag_on_falls_back_when_no_injection(self) -> None:
        """フラグ ON でも eyes_pose / eyelids 未指定なら identity / ゼロにフォールバック。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        result = adapter.convert(self._make_source(with_supplement=False))

        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(result["eyes_pose"], expected, atol=1e-6)
        assert torch.allclose(result["eyelids"], torch.zeros(1, 2))

    def test_flag_on_partial_injection(self) -> None:
        """eyes_pose のみ注入、eyelids 未指定の場合に正しく振り分けられること。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source: dict[str, torch.Tensor] = {
            "exp": torch.zeros(1, 50),
            "pose": torch.zeros(1, 6),
            "eyes_pose": torch.full((1, 12), 0.5),
        }
        result = adapter.convert(source)
        assert torch.allclose(result["eyes_pose"], source["eyes_pose"])
        assert torch.allclose(result["eyelids"], torch.zeros(1, 2))

    def test_flag_on_device_dtype_alignment(self) -> None:
        """注入された eyes_pose が exp の device/dtype に整列されること。"""
        adapter = DECAToFlameAdapter(use_mediapipe_supplement=True)
        source: dict[str, torch.Tensor] = {
            "exp": torch.zeros(1, 50, dtype=torch.float64),
            "pose": torch.zeros(1, 6, dtype=torch.float64),
            "eyes_pose": torch.full((1, 12), 0.5, dtype=torch.float32),
            "eyelids": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        }
        result = adapter.convert(source)
        assert result["eyes_pose"].dtype == torch.float64
        assert result["eyelids"].dtype == torch.float64


class TestDECAToFlameBatchSize:
    """異なるバッチサイズでの動作テスト。"""

    def test_batch_size_1(self) -> None:
        """バッチサイズ1で正しく動作すること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert({
            "exp": torch.randn(1, 50),
            "pose": torch.randn(1, 6),
        })
        assert result["expr"].shape == (1, 100)

    def test_batch_size_16(self) -> None:
        """バッチサイズ16で正しく動作すること。"""
        adapter = DECAToFlameAdapter()
        result = adapter.convert({
            "exp": torch.randn(16, 50),
            "pose": torch.randn(16, 6),
        })
        assert result["expr"].shape == (16, 100)
        assert result["jaw_pose"].shape == (16, 6)
        assert result["eyes_pose"].shape == (16, 12)
        assert result["eyelids"].shape == (16, 2)


import pytest  # noqa: E402 (for TestDECAToFlameAdapterProperties.test_missing_key_raises)


# =========================================================================
# SmirkToFlashAvatarAdapter
# =========================================================================


def _make_smirk_source(
    batch_size: int = 2,
    *,
    with_eyes_pose: bool = False,
    eyelid_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """SMIRK 出力相当のダミー dict を返す共通ヘルパー。"""
    eyelid = (
        torch.full((batch_size, 2), eyelid_value)
        if eyelid_value is not None
        else torch.rand(batch_size, 2)
    )
    source: dict[str, torch.Tensor] = {
        "shape": torch.randn(batch_size, 300),
        "exp": torch.randn(batch_size, 50),
        "pose": torch.randn(batch_size, 6),
        "cam": torch.randn(batch_size, 3),
        "eyelid": eyelid,
    }
    if with_eyes_pose:
        source["eyes_pose"] = torch.full((batch_size, 12), 0.5)
    return source


class TestSmirkToFlashAvatarShapes:
    """SMIRK→FlashAvatar 変換後テンソルの形状テスト。"""

    def test_expr_shape(self) -> None:
        """exp 50D → expr 100D の形状が (B, 100) であること。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(2))
        assert result["expr"].shape == (2, 100)

    def test_jaw_pose_shape(self) -> None:
        """jaw_pose の rotation_6d 変換後の形状が (B, 6) であること。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(2))
        assert result["jaw_pose"].shape == (2, 6)

    def test_eyes_pose_shape(self) -> None:
        """eyes_pose の形状が (B, 12) であること。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(2))
        assert result["eyes_pose"].shape == (2, 12)

    def test_eyelids_shape(self) -> None:
        """eyelids の形状が (B, 2) であること。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(2))
        assert result["eyelids"].shape == (2, 2)

    def test_shape_key_is_dropped(self) -> None:
        """変換結果に SMIRK の shape (300D) は含まれないこと。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(2))
        assert "shape" not in result

    def test_cam_key_is_dropped(self) -> None:
        """変換結果に SMIRK の cam (3D) は含まれないこと。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(2))
        assert "cam" not in result


class TestSmirkToFlashAvatarValues:
    """SMIRK→FlashAvatar 変換後の値テスト。"""

    def test_expr_first_50_preserved(self) -> None:
        """expr の最初の 50D が SMIRK exp と一致すること。"""
        adapter = SmirkToFlashAvatarAdapter()
        source = _make_smirk_source(2)
        result = adapter.convert(source)
        assert torch.allclose(result["expr"][:, :50], source["exp"])

    def test_expr_last_50_zero(self) -> None:
        """expr の最後の 50D がゼロパディングされること。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(2))
        assert torch.allclose(result["expr"][:, 50:], torch.zeros(2, 50))

    def test_eyelids_passthrough(self) -> None:
        """eyelids は SMIRK の eyelid をそのまま渡すこと (ネイティブ出力)。"""
        adapter = SmirkToFlashAvatarAdapter()
        source = _make_smirk_source(2, eyelid_value=0.42)
        result = adapter.convert(source)
        assert torch.allclose(result["eyelids"], source["eyelid"])

    def test_eyes_pose_default_is_identity_6d_repeated(self) -> None:
        """フラグ OFF のとき eyes_pose は単位回転 6D 表現 × 2 になること。"""
        adapter = SmirkToFlashAvatarAdapter()
        source = _make_smirk_source(1)
        source["pose"] = torch.zeros(1, 6)
        result = adapter.convert(source)

        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(result["eyes_pose"], expected, atol=1e-6)

    def test_jaw_pose_identity_for_zero_pose(self) -> None:
        """pose=0 のとき jaw_pose が単位回転 6D 表現になること。"""
        adapter = SmirkToFlashAvatarAdapter()
        source = _make_smirk_source(1)
        source["pose"] = torch.zeros(1, 6)
        result = adapter.convert(source)

        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        assert torch.allclose(result["jaw_pose"], identity_6d, atol=1e-6)


class TestSmirkToFlashAvatarProperties:
    """Adapter プロパティ・例外テスト。"""

    def test_source_format(self) -> None:
        """source_format が 'smirk' であること。"""
        adapter = SmirkToFlashAvatarAdapter()
        assert adapter.source_format == "smirk"

    def test_target_format(self) -> None:
        """target_format が 'flash_avatar' であること。"""
        adapter = SmirkToFlashAvatarAdapter()
        assert adapter.target_format == "flash_avatar"

    def test_default_flag_is_false(self) -> None:
        """use_mediapipe_supplement のデフォルトが False であること。"""
        adapter = SmirkToFlashAvatarAdapter()
        assert adapter.use_mediapipe_supplement is False

    def test_missing_exp_raises(self) -> None:
        """exp が欠損していると KeyError が発生すること。"""
        adapter = SmirkToFlashAvatarAdapter()
        with pytest.raises(KeyError):
            adapter.convert({
                "pose": torch.zeros(1, 6),
                "eyelid": torch.zeros(1, 2),
            })

    def test_missing_pose_raises(self) -> None:
        """pose が欠損していると KeyError が発生すること。"""
        adapter = SmirkToFlashAvatarAdapter()
        with pytest.raises(KeyError):
            adapter.convert({
                "exp": torch.zeros(1, 50),
                "eyelid": torch.zeros(1, 2),
            })

    def test_missing_eyelid_raises(self) -> None:
        """eyelid が欠損していると KeyError が発生すること。"""
        adapter = SmirkToFlashAvatarAdapter()
        with pytest.raises(KeyError):
            adapter.convert({
                "exp": torch.zeros(1, 50),
                "pose": torch.zeros(1, 6),
            })


class TestSmirkToFlashAvatarMediaPipeSupplement:
    """MediaPipe 補完フラグの挙動テスト (eyes_pose のみ対象)。

    SMIRK は eyelid をネイティブ出力するため、フラグの制御対象は eyes_pose のみ。
    """

    def test_flag_off_ignores_injected_eyes_pose(self) -> None:
        """フラグ OFF のとき source 提供の eyes_pose は無視され identity になること。"""
        adapter = SmirkToFlashAvatarAdapter(use_mediapipe_supplement=False)
        source = _make_smirk_source(1, with_eyes_pose=True)
        source["pose"] = torch.zeros(1, 6)
        result = adapter.convert(source)
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(result["eyes_pose"], expected, atol=1e-6)

    def test_flag_on_uses_injected_eyes_pose(self) -> None:
        """フラグ ON で eyes_pose 注入があれば優先採用されること。"""
        adapter = SmirkToFlashAvatarAdapter(use_mediapipe_supplement=True)
        source = _make_smirk_source(1, with_eyes_pose=True)
        result = adapter.convert(source)
        assert torch.allclose(result["eyes_pose"], source["eyes_pose"])

    def test_flag_on_falls_back_when_no_injection(self) -> None:
        """フラグ ON でも eyes_pose 未指定なら identity にフォールバック。"""
        adapter = SmirkToFlashAvatarAdapter(use_mediapipe_supplement=True)
        source = _make_smirk_source(1, with_eyes_pose=False)
        source["pose"] = torch.zeros(1, 6)
        result = adapter.convert(source)
        identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        expected = identity_6d.repeat(1, 2)
        assert torch.allclose(result["eyes_pose"], expected, atol=1e-6)

    def test_flag_on_does_not_override_eyelids(self) -> None:
        """フラグ ON でも eyelids は常に SMIRK のネイティブ値を使うこと。"""
        adapter = SmirkToFlashAvatarAdapter(use_mediapipe_supplement=True)
        source = _make_smirk_source(1, eyelid_value=0.7)
        # MediaPipe 由来の eyelids 値を渡しても無視される (SMIRK ネイティブを優先)。
        source["eyelids"] = torch.tensor([[0.1, 0.2]])
        result = adapter.convert(source)
        assert torch.allclose(result["eyelids"], source["eyelid"])

    def test_flag_on_device_dtype_alignment(self) -> None:
        """注入された eyes_pose が exp の dtype に整列されること。"""
        adapter = SmirkToFlashAvatarAdapter(use_mediapipe_supplement=True)
        source: dict[str, torch.Tensor] = {
            "exp": torch.zeros(1, 50, dtype=torch.float64),
            "pose": torch.zeros(1, 6, dtype=torch.float64),
            "eyelid": torch.zeros(1, 2, dtype=torch.float32),
            "eyes_pose": torch.full((1, 12), 0.5, dtype=torch.float32),
        }
        result = adapter.convert(source)
        assert result["eyes_pose"].dtype == torch.float64
        assert result["eyelids"].dtype == torch.float64


class TestSmirkToFlashAvatarBatchSize:
    """異なるバッチサイズでの動作テスト。"""

    def test_batch_size_1(self) -> None:
        """バッチサイズ1で正しく動作すること。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(1))
        assert result["expr"].shape == (1, 100)

    def test_batch_size_16(self) -> None:
        """バッチサイズ16で正しく動作すること。"""
        adapter = SmirkToFlashAvatarAdapter()
        result = adapter.convert(_make_smirk_source(16))
        assert result["expr"].shape == (16, 100)
        assert result["jaw_pose"].shape == (16, 6)
        assert result["eyes_pose"].shape == (16, 12)
        assert result["eyelids"].shape == (16, 2)


class TestSmirkToFlashAvatarRegistry:
    """AdapterRegistry 統合テスト。"""

    def test_registry_integration(self) -> None:
        """AdapterRegistry に登録・取得できること。"""
        from flare.converters.registry import AdapterRegistry

        registry = AdapterRegistry()
        adapter = SmirkToFlashAvatarAdapter()
        registry.register(adapter)
        got = registry.get("smirk", "flash_avatar")
        assert got is adapter

    def test_registry_build_chain(self) -> None:
        """build_chain で type 名 'smirk_to_flash_avatar' から取得できること。"""
        from flare.converters.registry import AdapterRegistry

        registry = AdapterRegistry()
        adapter = SmirkToFlashAvatarAdapter()
        registry.register(adapter)
        chain = registry.build_chain([{"type": "smirk_to_flash_avatar"}])
        assert len(chain) == 1
        assert chain[0] is adapter

    def test_registry_build_chain_with_flag_kwargs(self) -> None:
        """build_chain で use_mediapipe_supplement=True を kwargs 経由で渡せること。"""
        from flare.converters.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(SmirkToFlashAvatarAdapter())
        chain = registry.build_chain(
            [{"type": "smirk_to_flash_avatar", "use_mediapipe_supplement": True}]
        )
        assert len(chain) == 1
        assert isinstance(chain[0], SmirkToFlashAvatarAdapter)
        assert chain[0].use_mediapipe_supplement is True

    def test_registry_build_chain_kwargs_creates_new_instance(self) -> None:
        """kwargs ありの場合、登録済みインスタンスとは別の新規インスタンスが返ること。"""
        from flare.converters.registry import AdapterRegistry

        registry = AdapterRegistry()
        prototype = SmirkToFlashAvatarAdapter()
        registry.register(prototype)
        chain = registry.build_chain(
            [{"type": "smirk_to_flash_avatar", "use_mediapipe_supplement": True}]
        )
        assert chain[0] is not prototype
        assert chain[0].use_mediapipe_supplement is True
        assert prototype.use_mediapipe_supplement is False


class TestDECAToFlameAdapterRegistry:
    """DECAToFlameAdapter の registry 経由フラグ注入テスト (SMIRK と同じ wiring)。"""

    def test_build_chain_with_mediapipe_flag(self) -> None:
        """build_chain で DECA 用の use_mediapipe_supplement=True を渡せること。"""
        from flare.converters.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(DECAToFlameAdapter())
        chain = registry.build_chain(
            [{"type": "deca_to_flash_avatar", "use_mediapipe_supplement": True}]
        )
        assert len(chain) == 1
        assert isinstance(chain[0], DECAToFlameAdapter)
        assert chain[0].use_mediapipe_supplement is True


class TestConverterChainConfigParsing:
    """PipelineSettings.converter_chain が kwargs 付きエントリを受け入れること。"""

    def test_converter_chain_accepts_bool_kwargs(self) -> None:
        """converter_chain エントリに bool kwargs を含められること。"""
        from flare.config import PipelineSettings

        settings = PipelineSettings(
            converter_chain=[
                {"type": "deca_to_flash_avatar", "use_mediapipe_supplement": True},
            ]
        )
        assert settings.converter_chain[0]["type"] == "deca_to_flash_avatar"
        assert settings.converter_chain[0]["use_mediapipe_supplement"] is True

    def test_converter_chain_type_only_still_works(self) -> None:
        """既存の type のみのエントリも引き続き受け入れられること (後方互換)。"""
        from flare.config import PipelineSettings

        settings = PipelineSettings(
            converter_chain=[{"type": "deca_to_flash_avatar"}]
        )
        assert settings.converter_chain[0] == {"type": "deca_to_flash_avatar"}
