#!/usr/bin/env python
"""モデルチェックポイントの自動ダウンロードスクリプト。

自動ダウンロード可能なモデルを一括または個別にダウンロードする。
学術ライセンスが必要なモデル（FLAME, BFM 等）は手動ダウンロードが必要であり、
本スクリプトでは対応していない。

使用方法
--------

.. code-block:: bash

    # 全ての自動ダウンロード可能なモデルをダウンロード
    python scripts/download_checkpoints.py --all

    # 特定のモデルのみ
    python scripts/download_checkpoints.py --model deca
    python scripts/download_checkpoints.py --model 3ddfa

    # ダウンロード先を確認（実行しない）
    python scripts/download_checkpoints.py --all --dry-run

    # 手動ダウンロードが必要なモデルの一覧を表示
    python scripts/download_checkpoints.py --list-manual
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# ダウンロード定義
# ---------------------------------------------------------------------------

# 自動ダウンロード可能なモデル
# url が None の場合は git clone + ファイルコピーが必要
AUTO_DOWNLOADABLE: dict[str, dict] = {
    "deca": {
        "description": "DECA 学習済みモデル (ResNet-50)",
        "url": None,  # Google Drive — gdown が必要
        "gdown_id": "1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje",
        "dest": "checkpoints/deca/deca_model.tar",
        "note": "Google Drive からダウンロード (gdown 使用)",
    },
    "3ddfa": {
        "description": "3DDFA V2 MobileNet ONNX + BFM データ",
        "clone_url": "https://github.com/cleardusk/3DDFA_V2.git",
        "files": {
            "weights/mb1_120x120.onnx": "checkpoints/3ddfa/mb1_120x120.onnx",
            "configs/bfm_noneck_v3.pkl": "checkpoints/3ddfa/bfm_noneck_v3.pkl",
        },
        "note": "3DDFA_V2 リポジトリからファイルをコピー",
    },
}

# 手動ダウンロードが必要なモデル
MANUAL_REQUIRED: dict[str, dict] = {
    "flame": {
        "description": "FLAME パラメトリック顔モデル",
        "url": "https://flame.is.tue.mpg.de/",
        "dest": "checkpoints/flame/generic_model.pkl",
        "reason": "Max Planck Institute の学術ライセンスへの同意が必要",
    },
    "deep3d": {
        "description": "Deep3DFaceRecon + BFM 基底モデル",
        "url": "https://github.com/sicxu/Deep3DFaceRecon_pytorch",
        "dest": "checkpoints/deep3d/deep3d_epoch20.pth",
        "reason": "BFM モデルに学術ライセンスが必要",
    },
    "smirk": {
        "description": "SMIRK エンコーダモデル",
        "url": "https://github.com/georgeretsi/smirk",
        "dest": "checkpoints/smirk/smirk_encoder.pt",
        "reason": "リポジトリの指示に従い手動ダウンロードが必要",
    },
    "pirender": {
        "description": "PIRender 事前学習済みモデル",
        "url": "https://github.com/RenYurui/PIRender",
        "dest": "checkpoints/pirender/epoch_00190_iteration_000400000_checkpoint.pt",
        "reason": "リポジトリの指示に従い手動ダウンロードが必要",
    },
    "flashavatar": {
        "description": "FlashAvatar (対象人物ごとに学習が必要)",
        "url": "https://github.com/MingZhongCodes/FlashAvatar",
        "dest": "checkpoints/flashavatar/",
        "reason": "対象人物ごとの個別学習が必要（汎用モデルなし）",
    },
    "l2l": {
        "description": "Learning2Listen VQ-VAE モデル",
        "url": "https://github.com/evonneng/learning2listen",
        "dest": "checkpoints/l2l/l2l_vqvae.pth",
        "reason": "リポジトリの指示に従い手動ダウンロードが必要",
    },
}


# ---------------------------------------------------------------------------
# ダウンロード関数
# ---------------------------------------------------------------------------


def download_deca(dry_run: bool = False) -> bool:
    """DECA モデルを Google Drive からダウンロードする。"""
    dest = PROJECT_ROOT / AUTO_DOWNLOADABLE["deca"]["dest"]
    if dest.exists():
        print(f"  [skip] Already exists: {dest}")
        return True

    gdown_id = AUTO_DOWNLOADABLE["deca"]["gdown_id"]
    print(f"  Downloading DECA model from Google Drive (id={gdown_id})...")

    if dry_run:
        print(f"  [dry-run] Would download to: {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore[import-untyped]

        gdown.download(id=gdown_id, output=str(dest), quiet=False)
        print(f"  [ok] Saved: {dest}")
        return True
    except ImportError:
        print(
            "  [error] gdown is not installed. Install it with: pip install gdown"
        )
        print(f"  Or download manually from Google Drive (file id: {gdown_id})")
        print(f"  and place it at: {dest}")
        return False
    except Exception as e:
        print(f"  [error] Download failed: {e}")
        return False


def download_3ddfa(dry_run: bool = False) -> bool:
    """3DDFA V2 モデルをリポジトリクローンからコピーする。"""
    info = AUTO_DOWNLOADABLE["3ddfa"]
    all_exist = all(
        (PROJECT_ROOT / dest).exists() for dest in info["files"].values()
    )
    if all_exist:
        print("  [skip] All 3DDFA files already exist")
        return True

    if dry_run:
        for src, dest in info["files"].items():
            print(f"  [dry-run] Would copy {src} → {dest}")
        return True

    # 一時ディレクトリにクローン
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "3DDFA_V2"
        print(f"  Cloning 3DDFA_V2 repository...")

        result = subprocess.run(
            ["git", "clone", "--depth", "1", info["clone_url"], str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [error] git clone failed: {result.stderr}")
            return False

        success = True
        for src_rel, dest_rel in info["files"].items():
            src = clone_dir / src_rel
            dest = PROJECT_ROOT / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if src.exists():
                shutil.copy2(str(src), str(dest))
                print(f"  [ok] Copied: {dest}")
            else:
                print(f"  [warn] Source not found: {src}")
                print(f"         3DDFA_V2 のディレクトリ構成が変わった可能性があります")
                success = False

    return success


# モデル名 → ダウンロード関数のマッピング
_DOWNLOAD_FUNCS: dict[str, callable] = {
    "deca": download_deca,
    "3ddfa": download_3ddfa,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_manual() -> None:
    """手動ダウンロードが必要なモデルの一覧を表示する。"""
    print("\n=== 手動ダウンロードが必要なモデル ===\n")
    for name, info in MANUAL_REQUIRED.items():
        print(f"  [{name}] {info['description']}")
        print(f"    配置先: {info['dest']}")
        print(f"    理由:   {info['reason']}")
        print(f"    URL:    {info['url']}")
        readme = PROJECT_ROOT / Path(info["dest"]).parent / "README.md"
        if readme.exists():
            print(f"    詳細:   {readme.relative_to(PROJECT_ROOT)}")
        print()


def list_auto() -> None:
    """自動ダウンロード可能なモデルの一覧を表示する。"""
    print("\n=== 自動ダウンロード可能なモデル ===\n")
    for name, info in AUTO_DOWNLOADABLE.items():
        dest_key = "dest" if "dest" in info else None
        dests = [info["dest"]] if dest_key else list(info.get("files", {}).values())
        status = "✓" if all((PROJECT_ROOT / d).exists() for d in dests) else " "
        print(f"  [{status}] {name}: {info['description']}")
        print(f"       {info['note']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="モデルチェックポイントの自動ダウンロード"
    )
    parser.add_argument(
        "--model",
        choices=list(AUTO_DOWNLOADABLE.keys()),
        help="ダウンロードするモデル名",
    )
    parser.add_argument("--all", action="store_true", help="全モデルをダウンロード")
    parser.add_argument(
        "--dry-run", action="store_true", help="実際にはダウンロードしない"
    )
    parser.add_argument(
        "--list-manual",
        action="store_true",
        help="手動ダウンロードが必要なモデルの一覧を表示",
    )
    parser.add_argument(
        "--list-auto",
        action="store_true",
        help="自動ダウンロード可能なモデルの一覧を表示",
    )

    args = parser.parse_args()

    if args.list_manual:
        list_manual()
        return

    if args.list_auto:
        list_auto()
        return

    if not args.model and not args.all:
        parser.print_help()
        print("\n自動ダウンロード可能なモデル:")
        for name in AUTO_DOWNLOADABLE:
            print(f"  --model {name}")
        print("\n手動ダウンロードが必要なモデルの一覧は --list-manual で確認できます")
        return

    targets = list(AUTO_DOWNLOADABLE.keys()) if args.all else [args.model]
    results: dict[str, bool] = {}

    for name in targets:
        print(f"\n--- {name}: {AUTO_DOWNLOADABLE[name]['description']} ---")
        func = _DOWNLOAD_FUNCS[name]
        results[name] = func(dry_run=args.dry_run)

    # サマリ
    print("\n=== ダウンロード結果 ===")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    if not all(results.values()):
        sys.exit(1)

    # 手動ダウンロードの案内
    print("\n以下のモデルは手動ダウンロードが必要です:")
    print("詳細は --list-manual で確認してください")
    for name in MANUAL_REQUIRED:
        print(f"  - {name}: {MANUAL_REQUIRED[name]['description']}")


if __name__ == "__main__":
    main()
