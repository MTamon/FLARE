"""Loguru ベースロギング設定 (Section 8.6)

Section 8.6 採用理由:
  Loguru — ゼロ設定、カラー出力、ファイルローテーション、例外キャッチ。
  研究ツールに最適。pip install loguru のみで即使用可能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    fmt: Optional[str] = None,
) -> None:
    """Loguru のグローバルロガーを設定する。

    Section 8.6 のサンプル設定に準拠:
      - ファイルローテーション (rotation)
      - 保持期間 (retention)
      - レベル・フォーマット指定

    既存のハンドラを一度除去してから再登録するため、
    複数回呼び出しても安全。

    Args:
        level: ログレベル ("TRACE", "DEBUG", "INFO", …)。
        log_file: ログ出力先ファイルパス。None の場合 stderr のみ。
        rotation: ファイルローテーションの閾値 (例: "10 MB")。
        retention: ログ保持期間 (例: "30 days")。
        fmt: ログフォーマット文字列。None ならデフォルト。
    """
    if fmt is None:
        fmt = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
            "{module}:{line} | {message}"
        )

    # 既存ハンドラをリセット
    logger.remove()

    # stderr (コンソール) ハンドラ
    logger.add(
        sink=lambda msg: __import__("sys").stderr.write(msg),
        level=level,
        format=fmt,
        colorize=True,
    )

    # ファイルハンドラ
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=str(log_path),
            level=level,
            format=fmt,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )


def setup_logging_from_config(cfg) -> None:
    """ToolConfig.logging (LoggingConfig) から setup_logging を呼ぶヘルパー。

    Args:
        cfg: lhg_toolkit.config.LoggingConfig インスタンス。
    """
    setup_logging(
        level=cfg.level,
        log_file=cfg.file,
        rotation=cfg.rotation,
    )