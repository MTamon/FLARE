"""Loguruベースのロギング設定モジュール。

パイプライン全体で使用するロガーの初期化と取得を提供する。
仕様書§8.6に基づき、Loguruを採用する。研究用ツールにおいて最もシンプルかつ
十分な機能を提供するフレームワークである。

機能:
    - ゼロ設定でのカラーコンソール出力
    - ファイルローテーション（サイズベース）
    - 設定可能なリテンション期間
    - 統一フォーマット文字列

Example:
    ロガーのセットアップと使用::

        from flare.utils.logging import setup_logger, get_logger

        setup_logger(level="INFO", log_file="./logs/pipeline.log")
        logger = get_logger()
        logger.info("Pipeline started")
        logger.debug("Debug info: {}", some_variable)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

from loguru import logger

_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}"
)
"""str: 仕様書§8.6準拠のログフォーマット文字列。"""

_initialized: bool = False
"""bool: setup_logger()が呼び出し済みかどうかのフラグ。"""


def setup_logger(
    level: str = "INFO",
    log_file: Union[str, Path] = "./logs/pipeline.log",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """パイプライン用のLoguruロガーを設定する。

    既存のデフォルトハンドラ（stderr）を除去し、統一フォーマットで
    コンソール出力とファイル出力の両方を設定する。
    複数回呼び出された場合、既存のハンドラを全て除去してから再設定する。

    Args:
        level: ログレベル。``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"`` のいずれか。
        log_file: ログファイルの出力パス。親ディレクトリが存在しない場合は
            自動作成される。文字列またはPathオブジェクト。
        rotation: ログファイルのローテーション閾値。
            例: ``"10 MB"``, ``"100 MB"``, ``"1 week"``。
        retention: ログファイルの保持期間。
            例: ``"30 days"``, ``"1 week"``, ``"10 files"``。

    Raises:
        ValueError: 無効なログレベルが指定された場合（Loguruが送出）。
    """
    global _initialized  # noqa: PLW0603

    logger.remove()

    logger.add(
        sys.stderr,
        format=_LOG_FORMAT,
        level=level,
        colorize=True,
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format=_LOG_FORMAT,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    _initialized = True
    logger.debug("Logger initialized: level={}, file={}", level, log_path)


def get_logger() -> logger.__class__:
    """設定済みのLoguruロガーインスタンスを返す。

    setup_logger()が未呼び出しの場合、デフォルト設定（INFO, stderr のみ）で
    自動初期化を行ってからロガーを返す。

    Returns:
        Loguruのloggerモジュール。logger.info()等のメソッドが使用可能。

    Example:
        ::

            logger = get_logger()
            logger.info("Processing frame {}", frame_idx)
    """
    if not _initialized:
        setup_logger()
    return logger
