"""FLARE CLIエントリポイント。

プロジェクトルートから FLARE の CLI を実行するためのシンプルなエントリポイント。

Usage::

    python tool.py extract --help
    python tool.py render --help
"""

from __future__ import annotations

from flare.cli import cli

if __name__ == "__main__":
    cli()
