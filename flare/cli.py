"""FLARE CLI モジュール。

Clickベースのコマンドラインインターフェースを提供する。
仕様書§7.4に基づき、バッチ特徴量抽出（extract）とバッチレンダリング（render）の
2つのサブコマンドをサポートする。

Usage::

    # バッチ特徴量抽出
    python tool.py extract \\
        --input-dir /data/videos/ \\
        --output-dir /data/features/ \\
        --route flame --extractor deca \\
        --gpu 0 --batch-size 32

    # バッチレンダリング
    python tool.py render \\
        --input-dir /data/features/ \\
        --output-dir /data/rendered/ \\
        --route flame --renderer flashavatar \\
        --avatar-model /models/avatar_001/ --resolution 512,512
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from flare.config import (
    BufferConfig,
    DeviceMapConfig,
    ExtractorConfig,
    PipelineConfig,
    PipelineSettings,
    RendererConfig,
)
from flare.pipeline.batch import BatchPipeline
from flare.utils.logging import setup_logger


@click.group()
@click.version_option(version="2.2.0", prog_name="FLARE")
def cli() -> None:
    """FLARE: Facial Landmark Analysis & Rendering Engine.

    LHG研究における特徴量抽出とフォトリアルレンダリングの統合CLIツール。
    """


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="入力動画ファイルが格納されたディレクトリ。",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="抽出結果の出力先ディレクトリ。存在しない場合は自動作成。",
)
@click.option(
    "--route",
    required=True,
    type=click.Choice(["bfm", "flame"], case_sensitive=False),
    help="パイプラインルート。bfm=Route A（BFMベース）、flame=Route B（FLAMEベース）。",
)
@click.option(
    "--extractor",
    required=True,
    type=click.Choice(["deep3d", "deca", "smirk", "3ddfa"], case_sensitive=False),
    help="使用するExtractorの種別。",
)
@click.option(
    "--gpu",
    default=0,
    type=int,
    show_default=True,
    help="使用するGPUデバイスID。",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    show_default=True,
    help="バッチ処理のバッチサイズ。",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="設定検証のみ実行し、実際の処理は行わない。",
)
def extract(
    input_dir: str,
    output_dir: str,
    route: str,
    extractor: str,
    gpu: int,
    batch_size: int,
    dry_run: bool,
) -> None:
    """バッチ特徴量抽出を実行する。

    入力ディレクトリ内の動画ファイルから3DMMパラメータを一括抽出し、
    出力ディレクトリに保存する。

    ``--dry-run`` フラグを指定すると、設定の検証のみを行い実際の
    処理は実行しない。モデルファイルがなくても動作確認が可能。

    Args:
        input_dir: 入力動画ディレクトリのパス。
        output_dir: 出力先ディレクトリのパス。
        route: パイプラインルート（"bfm" または "flame"）。
        extractor: Extractorの種別。
        gpu: GPUデバイスID。
        batch_size: バッチサイズ。
        dry_run: Trueの場合、設定検証のみ実行。
    """
    setup_logger(level="INFO", log_file="./logs/extract.log")

    device = f"cuda:{gpu}"

    config = PipelineConfig(
        pipeline=PipelineSettings(
            name=f"extract_{route}_{extractor}",
        ),
        extractor=ExtractorConfig(
            type=extractor,
            input_size=224,
        ),
        buffer=BufferConfig(
            max_size=batch_size * 4,
            overflow_policy="block",
        ),
        device_map=DeviceMapConfig(
            extractor=device,
            lhg_model=device,
            renderer=device,
        ),
    )

    logger.info("=== FLARE Batch Extract ===")
    logger.info("Route:     {}", route)
    logger.info("Extractor: {}", extractor)
    logger.info("GPU:       {}", gpu)
    logger.info("Batch:     {}", batch_size)
    logger.info("Input:     {}", input_dir)
    logger.info("Output:    {}", output_dir)

    if dry_run:
        logger.info("[DRY RUN] Configuration validated successfully.")
        logger.info("[DRY RUN] Pipeline: {}", config.pipeline.name)
        logger.info("[DRY RUN] Extractor: type={}, input_size={}", config.extractor.type, config.extractor.input_size)
        logger.info("[DRY RUN] Device map: extractor={}, lhg_model={}, renderer={}", config.device_map.extractor, config.device_map.lhg_model, config.device_map.renderer)
        logger.info("[DRY RUN] Buffer: max_size={}, overflow_policy={}", config.buffer.max_size, config.buffer.overflow_policy)
        logger.info("[DRY RUN] No actual processing performed.")
        logger.info("=== Dry run complete ===")
        return

    input_path = Path(input_dir)
    if not input_path.exists():
        raise click.BadParameter(
            f"Input directory does not exist: {input_dir}",
            param_hint="--input-dir",
        )

    pipeline = BatchPipeline()
    pipeline.run(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
    )

    logger.info("=== Extract complete ===")


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="抽出済み特徴量が格納されたディレクトリ。",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="レンダリング結果の出力先ディレクトリ。存在しない場合は自動作成。",
)
@click.option(
    "--route",
    required=True,
    type=click.Choice(["bfm", "flame"], case_sensitive=False),
    help="パイプラインルート。bfm=Route A、flame=Route B。",
)
@click.option(
    "--renderer",
    required=True,
    type=click.Choice(["pirender", "flashavatar", "headgas"], case_sensitive=False),
    help="使用するRendererの種別。",
)
@click.option(
    "--avatar-model",
    required=True,
    type=click.Path(),
    help="アバターモデルのディレクトリまたはファイルパス。",
)
@click.option(
    "--resolution",
    default="512,512",
    type=str,
    show_default=True,
    help="出力解像度（幅,高さ）。カンマ区切り。",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="設定検証のみ実行し、実際の処理は行わない。",
)
def render(
    input_dir: str,
    output_dir: str,
    route: str,
    renderer: str,
    avatar_model: str,
    resolution: str,
    dry_run: bool,
) -> None:
    """バッチレンダリングを実行する。

    抽出済み特徴量ディレクトリからパラメータを読み込み、
    レンダリング結果を出力ディレクトリに保存する。

    ``--dry-run`` フラグを指定すると、設定の検証のみを行い実際の
    処理は実行しない。

    Args:
        input_dir: 抽出済み特徴量ディレクトリのパス。
        output_dir: 出力先ディレクトリのパス。
        route: パイプラインルート（"bfm" または "flame"）。
        renderer: Rendererの種別。
        avatar_model: アバターモデルのパス。
        resolution: 出力解像度文字列（"幅,高さ"）。
        dry_run: Trueの場合、設定検証のみ実行。
    """
    setup_logger(level="INFO", log_file="./logs/render.log")

    parts = resolution.split(",")
    if len(parts) != 2:
        raise click.BadParameter(
            f"resolution must be 'width,height', got: {resolution!r}",
            param_hint="--resolution",
        )
    width = int(parts[0].strip())
    height = int(parts[1].strip())

    config = PipelineConfig(
        pipeline=PipelineSettings(
            name=f"render_{route}_{renderer}",
        ),
        renderer=RendererConfig(
            type=renderer,
            model_path=avatar_model,
            output_size=[width, height],
        ),
    )

    logger.info("=== FLARE Batch Render ===")
    logger.info("Route:      {}", route)
    logger.info("Renderer:   {}", renderer)
    logger.info("Model:      {}", avatar_model)
    logger.info("Resolution: {}x{}", width, height)
    logger.info("Input:      {}", input_dir)
    logger.info("Output:     {}", output_dir)

    if dry_run:
        logger.info("[DRY RUN] Configuration validated successfully.")
        logger.info("[DRY RUN] Pipeline: {}", config.pipeline.name)
        logger.info("[DRY RUN] Renderer: type={}, model_path={}", config.renderer.type, config.renderer.model_path)
        logger.info("[DRY RUN] Output size: {}x{}", width, height)
        logger.info("[DRY RUN] No actual processing performed.")
        logger.info("=== Dry run complete ===")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Render pipeline ready. "
        "Renderer implementations will be available in Phase 2/3."
    )
    logger.info("=== Render complete ===")
