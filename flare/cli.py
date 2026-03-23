"""FLARE CLIインターフェース。

仕様書7.4節のCLIインターフェースに従い、clickベースのコマンドラインツールを提供する。

Usage:
    バッチ特徴量抽出::

        python -m flare.cli extract \\
            --input-dir /data/videos/ \\
            --output-dir /data/features/ \\
            --route flame --extractor deca \\
            --gpu 0 --batch-size 32

    バッチレンダリング::

        python -m flare.cli render \\
            --input-dir /data/features/ \\
            --output-dir /data/rendered/ \\
            --route flame --renderer flashavatar \\
            --avatar-model /models/avatar_001/ --resolution 512

    リアルタイムモード::

        python -m flare.cli realtime \\
            --config config.yaml --camera-id 0 --gpu 0
"""

from __future__ import annotations

from pathlib import Path

import click
from flare.config import FLAREConfig
from loguru import logger


@click.group(name="flare")
@click.version_option(version="0.1.0", prog_name="FLARE")
def cli() -> None:
    """FLARE: Facial Landmark Analysis & Rendering Engine.

    LHG研究における特徴量抽出・レンダリング統合ツール。
    """


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="入力動画ディレクトリ",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="出力特徴量ディレクトリ",
)
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    type=click.Path(),
    help="設定ファイルパス",
    show_default=True,
)
@click.option(
    "--route",
    type=click.Choice(["flame", "bfm"], case_sensitive=False),
    default="flame",
    help="処理ルート",
    show_default=True,
)
@click.option(
    "--extractor",
    type=click.Choice(["deca", "smirk", "deep3d"], case_sensitive=False),
    default="deca",
    help="Extractorの種別",
    show_default=True,
)
@click.option("--gpu", default=0, type=int, help="使用するGPU ID", show_default=True)
@click.option(
    "--batch-size", default=32, type=int, help="バッチサイズ", show_default=True
)
@click.option("--resume/--no-resume", default=True, help="チェックポイントから再開")
def extract(
    input_dir: str,
    output_dir: str,
    config_path: str,
    route: str,
    extractor: str,
    gpu: int,
    batch_size: int,
    resume: bool,
) -> None:
    """バッチ特徴量抽出を実行する。

    入力ディレクトリ内の動画ファイルから3DMMパラメータを一括抽出し、
    出力ディレクトリにnpz/npy形式で保存する。

    Args:
        input_dir: 入力動画ディレクトリ。
        output_dir: 出力特徴量ディレクトリ。
        config_path: YAML設定ファイルパス。
        route: 処理ルート（flame/bfm）。
        extractor: Extractorの種別。
        gpu: GPU ID。
        batch_size: バッチサイズ。
        resume: チェックポイントからの再開。
    """
    from flare.config import load_config
    from flare.pipeline.batch import BatchPipeline
    from flare.utils.logging import setup_logger

    config = _load_and_override_config(
        config_path,
        overrides={
            "extractor_type": extractor,
            "device": f"cuda:{gpu}",
            "route": route,
        },
    )

    setup_logger(
        level=config.logging.level,
        log_file=config.logging.file,
        rotation=config.logging.rotation,
    )

    logger.info(
        "バッチ抽出開始: route={} | extractor={} | gpu={} | batch_size={}",
        route,
        extractor,
        gpu,
        batch_size,
    )

    pipeline = BatchPipeline(config)
    pipeline.run(input_dir, output_dir, resume=resume)


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="入力特徴量ディレクトリ",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="出力レンダリングディレクトリ",
)
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    type=click.Path(),
    help="設定ファイルパス",
    show_default=True,
)
@click.option(
    "--route",
    type=click.Choice(["flame", "bfm"], case_sensitive=False),
    default="flame",
    help="処理ルート",
    show_default=True,
)
@click.option(
    "--renderer",
    type=click.Choice(["flashavatar", "pirender", "headgas"], case_sensitive=False),
    default="flashavatar",
    help="Rendererの種別",
    show_default=True,
)
@click.option(
    "--avatar-model",
    default=None,
    type=click.Path(),
    help="アバターモデルのパスまたはディレクトリ",
)
@click.option(
    "--resolution", default=512, type=int, help="出力解像度", show_default=True
)
def render(
    input_dir: str,
    output_dir: str,
    config_path: str,
    route: str,
    renderer: str,
    avatar_model: str | None,
    resolution: int,
) -> None:
    """バッチレンダリングを実行する。

    抽出済み特徴量からフォトリアルな顔画像を一括生成し、
    出力ディレクトリに動画/フレームシーケンスとして保存する。

    Args:
        input_dir: 入力特徴量ディレクトリ。
        output_dir: 出力レンダリングディレクトリ。
        config_path: YAML設定ファイルパス。
        route: 処理ルート（flame/bfm）。
        renderer: Rendererの種別。
        avatar_model: アバターモデルパス。
        resolution: 出力解像度。
    """
    from flare.config import load_config
    from flare.utils.logging import setup_logger

    config = _load_and_override_config(
        config_path,
        overrides={
            "renderer_type": renderer,
            "route": route,
        },
    )

    setup_logger(
        level=config.logging.level,
        log_file=config.logging.file,
        rotation=config.logging.rotation,
    )

    logger.info(
        "バッチレンダリング開始: route={} | renderer={} | resolution={} | avatar_model={}",
        route,
        renderer,
        resolution,
        avatar_model,
    )

    # Phase 2以降でレンダリングパイプラインを接続
    click.echo(
        f"レンダリングパイプライン: {input_dir} → {output_dir} "
        f"(route={route}, renderer={renderer}, resolution={resolution})"
    )
    logger.info("レンダリングパイプラインはPhase 2以降で完全実装されます")


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="設定ファイルパス",
)
@click.option(
    "--camera-id", default=0, type=int, help="カメラデバイスID", show_default=True
)
@click.option("--gpu", default=0, type=int, help="使用するGPU ID", show_default=True)
def realtime(config_path: str, camera_id: int, gpu: int) -> None:
    """リアルタイムモードを起動する。

    Webカメラ入力からリアルタイムに顔検出・3DMM抽出・
    LHGモデル推論・レンダリング・表示を行う。

    Args:
        config_path: YAML設定ファイルパス。
        camera_id: WebカメラのデバイスID。
        gpu: GPU ID。
    """
    from flare.config import load_config
    from flare.pipeline.realtime import RealtimePipeline
    from flare.utils.logging import setup_logger

    config = load_config(config_path)

    setup_logger(
        level=config.logging.level,
        log_file=config.logging.file,
        rotation=config.logging.rotation,
    )

    logger.info(
        "リアルタイムモード起動: camera={} | gpu={} | config={}",
        camera_id,
        gpu,
        config.pipeline.name,
    )

    pipeline = RealtimePipeline(config)
    pipeline.run(camera_id=camera_id)


def _load_and_override_config(
    config_path: str,
    overrides: dict[str, str],
) -> FLAREConfig:
    """設定ファイルを読み込み、CLIオプションでオーバーライドする。

    設定ファイルが存在しない場合はデフォルト設定を生成する。

    Args:
        config_path: YAML設定ファイルパス。
        overrides: CLIオプションによるオーバーライド値。

    Returns:
        バリデーション済みのFLARE設定。
    """
    from flare.config import (
        ExtractorConfig,
        FLAREConfig,
        LHGModelConfig,
        PipelineConfig,
        RendererConfig,
        load_config,
    )

    if Path(config_path).exists():
        config: FLAREConfig = load_config(config_path)
    else:
        logger.warning(
            "設定ファイルが見つかりません: {} — デフォルト設定を使用します",
            config_path,
        )
        config = FLAREConfig(
            pipeline=PipelineConfig(name="flare_default"),
            extractor=ExtractorConfig(
                type=overrides.get("extractor_type", "deca"),
                model_path="./checkpoints/deca_model.tar",
                return_keys=["shape", "exp", "pose", "detail"],
            ),
            renderer=RendererConfig(
                type=overrides.get("renderer_type", "flash_avatar"),
                model_path="./checkpoints/flashavatar/",
            ),
            lhg_model=LHGModelConfig(
                type="learning2listen",
                model_path="./checkpoints/l2l_vqvae.pth",
            ),
        )

    return config


if __name__ == "__main__":
    cli()
