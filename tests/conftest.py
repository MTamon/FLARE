"""FLARE テストスイート共有フィクスチャ。

全テストファイルで使用される共通のpytestフィクスチャを定義する。
FLAREConfig、画像テンソル、一時ディレクトリ等の生成を提供する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flare.config import (
    ConverterChainItemConfig,
    ExtractorConfig,
    FLAREConfig,
    LHGModelConfig,
    PipelineConfig,
    RendererConfig,
)


@pytest.fixture()
def dummy_flare_config() -> FLAREConfig:
    """最小有効設定のFLAREConfigインスタンスを返す。

    必須フィールドのみ指定し、オプションフィールドはデフォルト値を使用する。

    Returns:
        バリデーション済みのFLAREConfig。
    """
    return FLAREConfig(
        pipeline=PipelineConfig(
            name="test_pipeline",
            converter_chain=[
                ConverterChainItemConfig(type="identity"),
            ],
        ),
        extractor=ExtractorConfig(
            type="deca",
            model_path="./checkpoints/deca_model.tar",
            return_keys=["shape", "exp", "pose", "detail"],
        ),
        renderer=RendererConfig(
            type="flash_avatar",
            model_path="./checkpoints/flashavatar/",
        ),
        lhg_model=LHGModelConfig(
            type="learning2listen",
            model_path="./checkpoints/l2l_vqvae.pth",
        ),
    )


@pytest.fixture()
def dummy_config_yaml(tmp_path: Path) -> Path:
    """最小有効config.yamlを一時ファイルとして作成して返す。

    Args:
        tmp_path: pytestが提供する一時ディレクトリ。

    Returns:
        作成されたYAMLファイルのパス。
    """
    yaml_content: str = """\
pipeline:
  name: "test_pipeline"
  fps: 30
  device: "cuda:0"
  converter_chain:
    - type: "identity"

extractor:
  type: "deca"
  model_path: "./checkpoints/deca_model.tar"
  input_size: 224
  return_keys:
    - "shape"
    - "exp"
    - "pose"
    - "detail"

renderer:
  type: "flash_avatar"
  model_path: "./checkpoints/flashavatar/"
  output_size:
    - 512
    - 512

lhg_model:
  type: "learning2listen"
  model_path: "./checkpoints/l2l_vqvae.pth"
  window_size: 64
  codebook_size: 256

audio:
  sample_rate: 16000
  feature_type: "mel"
  n_mels: 128

buffer:
  max_size: 256
  timeout_sec: 0.5
  overflow_policy: "drop_oldest"

device_map:
  extractor: "cuda:0"
  lhg_model: "cuda:0"
  renderer: "cuda:0"

logging:
  level: "INFO"
  file: "./logs/pipeline.log"
  rotation: "10 MB"

checkpoint:
  enabled: true
  save_dir: "./checkpoints/batch/"
  format: "json"
"""
    config_path: Path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return config_path


@pytest.fixture()
def dummy_image_tensor() -> "torch.Tensor":
    """shape (1, 3, 224, 224) のランダム画像テンソルを返す。

    値域は [0, 1] のfloat32。Extractor入力を模擬する。

    Returns:
        ランダム画像テンソル。
    """
    import torch

    return torch.rand(1, 3, 224, 224)


@pytest.fixture()
def dummy_batch_tensor() -> "torch.Tensor":
    """shape (4, 3, 224, 224) のランダムバッチ画像テンソルを返す。

    値域は [0, 1] のfloat32。バッチ処理テスト用。

    Returns:
        ランダムバッチ画像テンソル。
    """
    import torch

    return torch.rand(4, 3, 224, 224)


@pytest.fixture()
def dummy_bgr_image() -> np.ndarray:
    """shape (480, 640, 3) のランダムBGR画像を返す。

    dtype: uint8、値域 [0, 255]。face_detect等のテスト用。

    Returns:
        ランダムBGR画像。
    """
    rng: np.random.Generator = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """一時出力ディレクトリを作成して返す。

    Args:
        tmp_path: pytestが提供する一時ディレクトリ。

    Returns:
        作成された出力ディレクトリのパス。
    """
    output_dir: Path = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
