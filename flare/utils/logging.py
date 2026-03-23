"""FLAREロギングフレームワーク。

仕様書8.6節「ロギングフレームワーク」に基づき、Loguruベースの
ロギング設定ユーティリティを提供する。

Loguruの採用理由:
    研究用ツールにおいて最もシンプルかつ十分な機能を提供する。
    ゼロ設定、カラー出力、ファイルローテーション、例外キャッチが
    ``pip install loguru`` のみで即使用可能。

Example:
    >>> from flare.utils.logging import setup_logger, get_pipeline_logger
    >>> setup_logger(level="DEBUG", log_file="./logs/debug.log")
    >>> log = get_pipeline_logger("extractor")
    >>> log.info("DECAモデルをロードしました")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

from loguru import logger

#: ログフォーマット文字列（仕様書8.6節準拠）
LOG_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}"
)

#: setup_logger()で追加したハンドラIDを追跡する内部リスト
_handler_ids: list[int] = []


def setup_logger(
    level: str = "INFO",
    log_file: str = "./logs/pipeline.log",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Loguruのloggerを初期設定する。

    既存のハンドラを全て除去した後、stderr出力とファイル出力の
    2つのハンドラを追加する。ログファイルの親ディレクトリが
    存在しない場合は自動的に作成する。

    二重登録防止のため、呼び出しごとに既存ハンドラを一度除去してから
    新しいハンドラを追加する。

    Args:
        level: ログレベル。``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"`` のいずれか。
        log_file: ログファイルの出力パス。
        rotation: ログファイルのローテーション条件。
            例: ``"10 MB"``, ``"1 day"``, ``"00:00"``。
        retention: ログファイルの保持期間。
            例: ``"30 days"``, ``"1 week"``。

    Example:
        >>> setup_logger(level="DEBUG", log_file="./logs/debug.log")
        >>> from loguru import logger
        >>> logger.info("ログ設定完了")
    """
    global _handler_ids

    # 既存ハンドラを全て除去（二重登録防止）
    for handler_id in _handler_ids:
        try:
            logger.remove(handler_id)
        except ValueError:
            # 既に除去済みの場合は無視
            pass
    _handler_ids.clear()

    # デフォルトのstderrハンドラを除去
    try:
        logger.remove(0)
    except ValueError:
        pass

    # stderr ハンドラを追加
    stderr_id: int = logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        colorize=True,
    )
    _handler_ids.append(stderr_id)

    # ファイルハンドラを追加
    log_path: Path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_id: int = logger.add(
        str(log_path),
        level=level,
        format=LOG_FORMAT,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,  # スレッドセーフな非同期書き込み
    )
    _handler_ids.append(file_id)


def get_pipeline_logger(name: str) -> "logger":
    """名前付きパイプラインロガーを返す。

    Loguruの ``bind()`` を使用して ``name`` フィールドを付加した
    ロガーインスタンスを返す。ログ出力時にコンポーネント名を
    識別するために使用する。

    Args:
        name: ロガーに付加するコンポーネント名。
            例: ``"extractor"``, ``"renderer"``, ``"pipeline"``。

    Returns:
        nameフィールドがバインドされたLoguruロガー。

    Example:
        >>> log = get_pipeline_logger("deca_extractor")
        >>> log.info("パラメータ抽出完了: {count}フレーム", count=100)
    """
    return logger.bind(name=name)


def log_frame_drop(frame_index: int, policy: str) -> None:
    """フレームドロップをWARNINGレベルで記録する。

    リアルタイムパイプラインでフレームドロップが発生した際に、
    統一的なフォーマットでログを出力するユーティリティ関数。

    Args:
        frame_index: ドロップされたフレームのインデックス。
        policy: 適用されたドロップポリシー名。
            例: ``"drop_oldest"``, ``"interpolate"``。

    Example:
        >>> log_frame_drop(frame_index=142, policy="drop_oldest")
        # WARNING | フレームドロップ: frame_index=142 | policy=drop_oldest
    """
    logger.warning(
        "フレームドロップ: frame_index={} | policy={}",
        frame_index,
        policy,
    )
