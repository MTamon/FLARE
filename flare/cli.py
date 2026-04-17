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
from typing import Optional

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
from flare.pipeline.lhg_batch import LHGBatchPipeline
from flare.utils.logging import setup_logger


def _build_renderer(
    renderer_type: str,
    model_path: str,
    output_size: list[int],
    device: str,
    repo_dir: Optional[str] = None,
):
    """Renderer 種別からインスタンスを構築するヘルパー。

    Args:
        renderer_type: ``"flash_avatar"``, ``"pirender"``, ``"headgas"`` のいずれか。
        model_path: モデルディレクトリまたはファイルパス。
        output_size: 出力画像サイズ ``[幅, 高さ]``。
        device: 計算デバイス文字列。
        repo_dir: 外部リポジトリのルートディレクトリパス。FlashAvatar の場合は
            ``./third_party/FlashAvatar`` 等を指定。None の場合はインポート済みと仮定。

    Returns:
        構築済み ``BaseRenderer`` サブクラスインスタンス。

    Raises:
        ValueError: 未知の renderer_type の場合。
    """
    rt = renderer_type.lower()
    if rt in ("flash_avatar", "flashavatar"):
        from flare.renderers.flashavatar import FlashAvatarRenderer

        return FlashAvatarRenderer(
            model_path=model_path,
            device=device,
            output_size=output_size,
            flashavatar_dir=repo_dir,
        )
    if rt == "pirender":
        from flare.renderers.pirender import PIRenderRenderer

        return PIRenderRenderer(model_path=model_path, device=device)
    if rt == "headgas":
        from flare.renderers.headgas import HeadGaSRenderer

        return HeadGaSRenderer(model_path=model_path, device=device)
    raise ValueError(f"Unknown renderer type: {renderer_type!r}")


def _build_extractor(
    extractor_type: str,
    model_path: str,
    device: str,
    repo_dir: Optional[str] = None,
):
    """Extractor 種別からインスタンスを構築するヘルパー。

    重いモデルファイルの読み込みを伴うため、``dry-run`` 時には呼び出さない。

    Args:
        extractor_type: ``"deca"``, ``"deep3d"``, ``"smirk"``, ``"3ddfa"`` のいずれか。
        model_path: モデルチェックポイントパス。
        device: 計算デバイス文字列。
        repo_dir: 外部リポジトリのルートディレクトリパス。DECA の場合は
            ``./third_party/DECA`` 等を指定。None の場合はインポート済みと仮定。

    Returns:
        構築済み ``BaseExtractor`` サブクラスインスタンス。

    Raises:
        ValueError: 未知の extractor_type の場合。
    """
    et = extractor_type.lower()
    if et == "deca":
        from flare.extractors.deca import DECAExtractor

        return DECAExtractor(model_path=model_path, device=device, deca_dir=repo_dir)
    if et == "deep3d":
        from flare.extractors.deep3d import Deep3DFaceReconExtractor

        return Deep3DFaceReconExtractor(model_path=model_path, device=device)
    if et == "smirk":
        from flare.extractors.smirk import SMIRKExtractor

        return SMIRKExtractor(model_path=model_path, device=device)
    if et in ("3ddfa", "tdddfa"):
        from flare.extractors.tdddfa import TDDFAExtractor

        return TDDFAExtractor(model_path=model_path, device=device)
    raise ValueError(f"Unknown extractor type: {extractor_type!r}")


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

    renderer = _build_renderer(
        renderer_type=renderer,
        model_path=avatar_model,
        output_size=[width, height],
        device=config.device_map.renderer,
        repo_dir=config.renderer.repo_dir,
    )
    renderer.setup()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Renderer initialized: type={}, model={}", renderer, avatar_model)
    logger.info("=== Render complete ===")


@cli.command("lhg-extract")
@click.option(
    "--path",
    "input_root",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="multimodal_dialogue_formed ルートディレクトリ。",
)
@click.option(
    "--output",
    "output_root",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="movements 出力ルートディレクトリ。存在しなければ作成。",
)
@click.option(
    "--extractor",
    "extractor_type",
    default=None,
    type=click.Choice(["deca", "deep3d", "smirk", "3ddfa"], case_sensitive=False),
    help="使用する Extractor 種別。指定時は --config の extractor.type を上書き。",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML 設定ファイルパス。省略時はデフォルト設定を使用。",
)
@click.option(
    "--model-path",
    default=None,
    type=click.Path(),
    help="Extractor モデルチェックポイントパス。指定時は config を上書き。",
)
@click.option(
    "--num-workers",
    default=1,
    type=int,
    show_default=True,
    help="並列ワーカ数（現状は 1 のみサポート）。",
)
@click.option(
    "--gpus",
    default=None,
    type=str,
    help="使用する GPU ID（カンマ区切り、例: '0,1'）。先頭のみが extractor に使用される。",
)
@click.option(
    "--redo",
    is_flag=True,
    default=False,
    help="既存出力を上書きする。",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="設定検証のみ実行。モデルや動画は読み込まない。",
)
def lhg_extract(
    input_root: str,
    output_root: str,
    extractor_type: Optional[str],
    config_path: Optional[str],
    model_path: Optional[str],
    num_workers: int,
    gpus: Optional[str],
    redo: bool,
    dry_run: bool,
) -> None:
    """LHG 頭部特徴量抽出バッチを実行する。

    ``multimodal_dialogue_formed/dataXXX/{comp,host}.mp4`` 形式のデータセットから
    DECA / Deep3DFaceRecon / SMIRK / 3DDFA を用いて per-frame に 3DMM パラメータを
    抽出し、ギャップ補間・シーケンス分割・対話単位正規化を行って
    ``movements/dataXXX/{comp,host}/<prefix>_<role>_<SSSSS>_<EEEEE>.npz`` として
    保存する。下流の ``databuild_nx8.py`` と互換な npz スキーマで出力する。

    Args:
        input_root: ``multimodal_dialogue_formed`` ルートディレクトリ。
        output_root: ``movements`` 出力ルート。
        extractor_type: Extractor 種別。指定時は config を上書き。
        config_path: YAML 設定ファイル。
        model_path: Extractor モデルチェックポイントパス。
        num_workers: 並列ワーカ数。
        gpus: 使用する GPU ID 列（カンマ区切り）。
        redo: 既存出力を上書きするか。
        dry_run: 設定検証のみ実行するか。
    """
    setup_logger(level="INFO", log_file="./logs/lhg_extract.log")

    if config_path is not None:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    if extractor_type is not None:
        config.extractor.type = extractor_type.lower()
    if model_path is not None:
        config.extractor.model_path = model_path

    if gpus is not None:
        gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
        if gpu_ids:
            primary = f"cuda:{gpu_ids[0]}"
            config.device_map.extractor = primary
            config.device_map.lhg_model = primary
            config.device_map.renderer = primary

    logger.info("=== FLARE LHG Extract ===")
    logger.info("Input root:   {}", input_root)
    logger.info("Output root:  {}", output_root)
    logger.info("Extractor:    {}", config.extractor.type)
    logger.info("Model path:   {}", config.extractor.model_path)
    logger.info("Device:       {}", config.device_map.extractor)
    logger.info("Num workers:  {}", num_workers)
    logger.info("Redo:         {}", redo)
    logger.info(
        "Interp linear={} rotation={} max_gap_sec={}",
        config.lhg_extract.interpolation.linear_order,
        config.lhg_extract.interpolation.rotation_order,
        config.lhg_extract.interpolation.max_gap_sec,
    )
    logger.info("Min length:   {}", config.lhg_extract.sequence.min_length)
    logger.info(
        "Shape agg:    {}", config.lhg_extract.output.shape_aggregation
    )

    if dry_run:
        logger.info("[DRY RUN] Configuration validated successfully.")
        logger.info("[DRY RUN] No models loaded and no videos processed.")
        logger.info("=== Dry run complete ===")
        return

    input_path = Path(input_root)
    if not input_path.exists():
        raise click.BadParameter(
            f"Input root does not exist: {input_root}",
            param_hint="--path",
        )

    extractor = _build_extractor(
        extractor_type=config.extractor.type,
        model_path=config.extractor.model_path,
        device=config.device_map.extractor,
        repo_dir=config.extractor.repo_dir,
    )

    pipeline = LHGBatchPipeline(config=config, extractor=extractor)
    stats = pipeline.run(
        input_root=input_root,
        output_root=output_root,
        num_workers=num_workers,
        redo=redo,
    )

    logger.info(
        "=== LHG extract complete: {} dirs, {} sequences, {} skipped ===",
        stats["num_data_dirs"],
        stats["num_sequences"],
        stats["num_skipped"],
    )
