#!/usr/bin/env python
"""FlashAvatar 学習パイプライン (DECA / SMIRK 対応) のオーケストレータ。

任意の入力動画から FlashAvatar の 3DGS モデルを学習する 5 ステージパイプライン。
configs/train_flashavatar.yaml でハイパーパラメータを管理し、
CLI 引数で個別上書きができる。

ステージ:
    Step 1: 動画フレーム抽出 + per-frame 特徴抽出 (DECA or SMIRK)
    Step 2: MediaPipe セグメンテーションマスク生成
    Step 3: per-frame .pt → FlashAvatar .frame 変換 (FlameConverter)
    Step 4: FlashAvatar train.py で 3DGS を学習
    Step 5: FlashAvatar test.py で検証動画を生成

推奨ハイパーパラメータ (configs/train_flashavatar.yaml のデフォルト値):
    - iterations: 30000  (RTX 3090 + img_size=512 で約 30 分)
    - img_size:   512
    - target_fps: 25     (25fps 未満の動画は補間、超える動画はサブサンプリング)
    - extractor:  deca   (SMIRK は --extractor smirk で変更)

VRAM の目安:
    - 12 GB: img_size=512, iterations=30000
    - 8 GB:  img_size=256, iterations=15000 (品質低下あり)

Usage:
    # DECA (デフォルト)
    python scripts/train_flashavatar.py \\
        --id_name person01 \\
        --video data/raw/person01.mp4

    # SMIRK (非対称表情向け)
    python scripts/train_flashavatar.py \\
        --id_name person01 \\
        --video data/raw/person01.mp4 \\
        --extractor smirk

    # カスタム設定 + 途中から再開
    python scripts/train_flashavatar.py \\
        --id_name person01 \\
        --config configs/train_flashavatar.yaml \\
        --skip_extract --skip_masks --skip_convert

    # 学習済みモデルで検証動画のみ生成
    python scripts/train_flashavatar.py \\
        --id_name person01 \\
        --test_only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

# FLARE ルートを sys.path に追加してローカルモジュールを使えるようにする
_FLARE_ROOT = Path(__file__).resolve().parent.parent
if str(_FLARE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLARE_ROOT))

from flare.training import TrainFlashAvatarConfig  # noqa: E402

_MIN_RECOMMENDED_FRAMES = 600
_TARGET_FPS_DEFAULT = 25


# ---------------------------------------------------------------------------
# エラーメッセージ
# ---------------------------------------------------------------------------

def _err_video_not_found(video: str) -> str:
    return (
        f"\n[ERROR] 動画ファイルが見つかりません: {video}\n"
        f"  - パスが正しいか確認してください。\n"
        f"  - 相対パスは FLARE ルートから指定してください。\n"
        f"  例: --video data/raw/person01.mp4"
    )


def _err_submodule_not_init(name: str, install_cmd: str) -> str:
    return (
        f"\n[ERROR] {name} サブモジュールが未初期化です。\n"
        f"  以下を実行してください:\n"
        f"    {install_cmd}"
    )


def _err_checkpoint_not_found(name: str, path: str, install_cmd: str) -> str:
    return (
        f"\n[ERROR] {name} チェックポイントが見つかりません: {path}\n"
        f"  以下を実行してダウンロードしてください:\n"
        f"    {install_cmd}"
    )


def _err_flame_asset_not_found(asset: str, fa_dir: str) -> str:
    return (
        f"\n[ERROR] FLAME アセットが見つかりません: {asset}\n"
        f"  https://flame.is.tue.mpg.de/ でダウンロードして\n"
        f"  {fa_dir}/flame/ 以下に配置してください。\n"
        f"  詳細: docs/guide_deca_flashavatar.md"
    )


# ---------------------------------------------------------------------------
# 環境チェック
# ---------------------------------------------------------------------------

def _check_prerequisites(
    cfg: TrainFlashAvatarConfig,
    video: Optional[str],
    skip_extract: bool,
    flare_root: Path,
) -> None:
    """前提条件を確認し、問題があれば sys.exit する。"""
    fa_abs = (flare_root / cfg.flashavatar.repo_dir).resolve()
    ext_cfg = cfg.active_extractor_settings()

    # --- 動画ファイル ---
    if not skip_extract:
        if not video:
            logger.error(
                "--video は --skip_extract なしでは必須です。\n"
                "  --skip_extract を指定した場合、抽出済みデータを使います。"
            )
            sys.exit(1)
        if not Path(video).exists():
            logger.error(_err_video_not_found(video))
            sys.exit(1)

    # --- FlashAvatar サブモジュール ---
    if not (fa_abs / "train.py").exists():
        logger.error(
            _err_submodule_not_init(
                "FlashAvatar", "bash install/setup_flashavatar.sh"
            )
        )
        sys.exit(1)

    # --- Extractor チェックポイント ---
    if not skip_extract:
        model_path = Path(ext_cfg.model_path)
        if not model_path.exists():
            if cfg.pipeline.extractor == "deca":
                install_cmd = "bash install/setup_deca.sh"
            else:
                install_cmd = "bash install/setup_smirk.sh"
            logger.error(
                _err_checkpoint_not_found(
                    cfg.pipeline.extractor.upper(),
                    str(model_path),
                    install_cmd,
                )
            )
            sys.exit(1)

    # --- FLAME アセット (FlashAvatar 用) ---
    flame_model = fa_abs / "flame" / "generic_model.pkl"
    flame_masks = fa_abs / "flame" / "FLAME_masks" / "FLAME_masks.pkl"
    for asset in (flame_model, flame_masks):
        if not asset.exists():
            logger.error(_err_flame_asset_not_found(str(asset), str(fa_abs)))
            sys.exit(1)


# ---------------------------------------------------------------------------
# 動画の FPS / 解像度正規化
# ---------------------------------------------------------------------------

def _probe_video(video: str) -> tuple[float, int, int]:
    """ffprobe で動画の FPS / W / H を取得する。

    Returns:
        (fps, width, height)

    Raises:
        RuntimeError: ffprobe が見つからないか解析に失敗した場合。
    """
    if not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffprobe が見つかりません。ffmpeg をインストールしてください:\n"
            "  sudo apt-get install ffmpeg"
        )
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "csv=p=0",
        video,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    parts = out.split(",")
    width = int(parts[0])
    height = int(parts[1])
    fps_str = parts[2]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)
    return fps, width, height


def _needs_normalization(
    fps: float, width: int, height: int, cfg: "TrainFlashAvatarConfig"
) -> bool:
    """FPS 変換 or 解像度変換が必要かどうかを判定する。"""
    img_size = cfg.video.img_size
    target_fps = cfg.video.target_fps
    if target_fps is not None and abs(fps - target_fps) > 0.1:
        return True
    if cfg.video.center_crop and (width != height):
        return True
    if width != img_size or height != img_size:
        return True
    return False


def normalize_video(
    video: str,
    cfg: "TrainFlashAvatarConfig",
    tmp_dir: Path,
) -> str:
    """入力動画を学習用 FPS / 解像度に正規化し、一時ファイルパスを返す。

    ffmpeg を使って:
        1. FPS を target_fps にリサンプリング (設定 null なら skip)
        2. center_crop=True なら短辺基準クロップ
        3. img_size × img_size にリサイズ

    元動画が既に条件を満たす場合は元のパスをそのまま返す (コピーしない)。

    Args:
        video: 入力動画ファイルパス。
        cfg: 学習設定。
        tmp_dir: 正規化済み動画を書き出す一時ディレクトリ。

    Returns:
        使用すべき動画ファイルパス (正規化済みまたは元ファイル)。
    """
    if not shutil.which("ffmpeg"):
        logger.warning(
            "ffmpeg が見つかりません。FPS / 解像度の自動正規化をスキップします。\n"
            "  sudo apt-get install ffmpeg をインストールすることを推奨します。"
        )
        return video

    try:
        fps, width, height = _probe_video(video)
    except Exception as e:
        logger.warning("動画プロービング失敗 ({}): 正規化をスキップします", e)
        return video

    if not _needs_normalization(fps, width, height, cfg):
        logger.info(
            "動画は既に正規化済み: FPS={:.1f}, {}×{} → スキップ",
            fps, width, height,
        )
        return video

    img_size = cfg.video.img_size
    target_fps = cfg.video.target_fps

    # ffmpeg フィルター構築
    filters: list[str] = []
    if cfg.video.center_crop and (width != height):
        crop_size = min(width, height)
        x_offset = (width - crop_size) // 2
        y_offset = (height - crop_size) // 2
        filters.append(f"crop={crop_size}:{crop_size}:{x_offset}:{y_offset}")

    filters.append(f"scale={img_size}:{img_size}:flags=lanczos")

    if target_fps is not None and abs(fps - target_fps) > 0.1:
        filters.append(f"fps={target_fps}")

    filter_str = ",".join(filters)
    out_path = tmp_dir / "normalized.mp4"

    cmd = [
        "ffmpeg", "-y", "-i", video,
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",  # 音声不要
        str(out_path),
    ]
    logger.info(
        "動画を正規化します: FPS {:.1f}→{}, {}×{}→{}×{}",
        fps, target_fps or fps, width, height, img_size, img_size,
    )
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info("正規化済み動画: {}", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# サブプロセス呼び出しヘルパー
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """サブプロセスを実行し、失敗したら sys.exit する。"""
    logger.debug("実行: {}", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        logger.error("コマンドが失敗しました (returncode={}): {}", result.returncode, " ".join(cmd))
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# 各ステージの実行
# ---------------------------------------------------------------------------

def _run_extract(
    video: str,
    data_dir: Path,
    cfg: TrainFlashAvatarConfig,
    flare_root: Path,
) -> None:
    """Step 1: フレーム抽出 + per-frame 特徴抽出。"""
    ext_cfg = cfg.active_extractor_settings()
    extractor = cfg.pipeline.extractor
    device = cfg.pipeline.device
    img_size = cfg.video.img_size
    max_frames = cfg.video.max_frames

    if extractor == "deca":
        script = flare_root / "scripts" / "extract_deca_frames.py"
        cmd = [
            sys.executable, str(script),
            "--video", video,
            "--out_dir", str(data_dir),
            "--model_path", ext_cfg.model_path,
            "--deca_dir", ext_cfg.repo_dir,
            "--device", device,
            "--img_size", str(img_size),
        ]
    else:
        script = flare_root / "scripts" / "extract_smirk_frames.py"
        cmd = [
            sys.executable, str(script),
            "--video", video,
            "--out_dir", str(data_dir),
            "--model_path", ext_cfg.model_path,
            "--smirk_dir", ext_cfg.repo_dir,
            "--device", device,
            "--img_size", str(img_size),
        ]

    if max_frames is not None:
        cmd += ["--max_frames", str(max_frames)]

    _run(cmd, cwd=flare_root)

    # フレーム数チェック
    pt_subdir = "deca_outputs" if extractor == "deca" else "smirk_outputs"
    pt_dir = data_dir / pt_subdir
    n_frames = len(list(pt_dir.glob("*.pt"))) if pt_dir.exists() else 0
    if n_frames < _MIN_RECOMMENDED_FRAMES:
        logger.warning(
            "フレーム数が少ない可能性があります ({} フレーム)。\n"
            "  FlashAvatar は最低 {} フレーム (target_fps=25 なら {} 秒) を推奨します。",
            n_frames, _MIN_RECOMMENDED_FRAMES,
            _MIN_RECOMMENDED_FRAMES // (cfg.video.target_fps or _TARGET_FPS_DEFAULT),
        )
    logger.info("Step 1 完了: {} フレーム抽出", n_frames)


def _run_masks(data_dir: Path, cfg: TrainFlashAvatarConfig, flare_root: Path) -> None:
    """Step 2: MediaPipe マスク生成。"""
    script = flare_root / "scripts" / "generate_masks_mediapipe.py"
    imgs_dir = data_dir / "imgs"
    cmd = [
        sys.executable, str(script),
        "--imgs_dir", str(imgs_dir),
        "--out_dir", str(data_dir),
        "--img_size", str(cfg.video.img_size),
    ]
    _run(cmd, cwd=flare_root)

    n_neckhead = len(list((data_dir / "parsing").glob("*_neckhead.png"))) if (data_dir / "parsing").exists() else 0
    n_alpha = len(list((data_dir / "alpha").glob("*.jpg"))) if (data_dir / "alpha").exists() else 0
    logger.info("Step 2 完了: neckhead={}, alpha={}", n_neckhead, n_alpha)


def _run_convert(
    data_dir: Path,
    frame_dir: Path,
    cfg: TrainFlashAvatarConfig,
    flare_root: Path,
) -> None:
    """Step 3: per-frame .pt → FlashAvatar .frame 変換。"""
    fa_abs = (flare_root / cfg.flashavatar.repo_dir).resolve()
    extractor = cfg.pipeline.extractor
    pt_subdir = "deca_outputs" if extractor == "deca" else "smirk_outputs"
    pt_dir = data_dir / pt_subdir
    img_size = cfg.video.img_size

    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "utils/flame_converter.py",
        "--tracker", extractor,
        "--input_dir", str(pt_dir),
        "--output_dir", str(frame_dir),
        "--img_size", f"{img_size},{img_size}",
        "--ext", ".pt",
    ]
    _run(cmd, cwd=fa_abs)

    n_frames_out = len(list(frame_dir.glob("*.frame"))) if frame_dir.exists() else 0
    logger.info("Step 3 完了: {} .frame ファイル生成", n_frames_out)


def _run_train(
    id_name: str,
    data_dir: Path,
    cfg: TrainFlashAvatarConfig,
    flare_root: Path,
) -> None:
    """Step 4: FlashAvatar train.py で 3DGS を学習。"""
    fa_abs = (flare_root / cfg.flashavatar.repo_dir).resolve()
    ckpt_dir = data_dir / "log" / "ckpt"
    img_size = cfg.video.img_size
    iterations = cfg.flashavatar.iterations

    cmd = [
        sys.executable, "train.py",
        "--idname", id_name,
        "--iterations", str(iterations),
        "--image_res", str(img_size),
    ]

    if cfg.flashavatar.resume_if_exists and ckpt_dir.exists():
        existing = sorted(ckpt_dir.glob("*.pth"))
        if existing:
            latest_ckpt = str(existing[-1])
            cmd += ["--start_checkpoint", latest_ckpt]
            logger.info("既存チェックポイントで再開: {}", latest_ckpt)

    _run(cmd, cwd=fa_abs)
    logger.info("Step 4 完了: checkpoints は {} 以下", ckpt_dir)


def _run_test(
    id_name: str,
    data_dir: Path,
    cfg: TrainFlashAvatarConfig,
    flare_root: Path,
) -> None:
    """Step 5: FlashAvatar test.py で検証動画を生成。"""
    fa_abs = (flare_root / cfg.flashavatar.repo_dir).resolve()
    ckpt_dir = data_dir / "log" / "ckpt"

    existing = sorted(ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else []
    if not existing:
        logger.warning(
            "チェックポイントが見つかりません。Step 5 をスキップします。\n"
            "  {}", ckpt_dir,
        )
        return

    latest_ckpt = str(existing[-1])
    cmd = [
        sys.executable, "test.py",
        "--idname", id_name,
        "--checkpoint", latest_ckpt,
    ]
    _run(cmd, cwd=fa_abs)

    test_video = data_dir / "log" / "test.avi"
    logger.info("Step 5 完了: {}", test_video)


def _setup_fa_symlink(
    id_name: str,
    data_dir: Path,
    cfg: TrainFlashAvatarConfig,
    flare_root: Path,
) -> None:
    """FlashAvatar dataset/<id_name> → data_dir のシンボリックリンクを作成する。"""
    fa_abs = (flare_root / cfg.flashavatar.repo_dir).resolve()
    dataset_dir = fa_abs / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    link = dataset_dir / id_name

    if link.is_symlink():
        link.unlink()
    if link.is_dir():
        logger.warning(
            "dataset/{} はシンボリックリンクではなくディレクトリです。\n"
            "  手動で確認してください: {}",
            id_name, link,
        )
        return

    link.symlink_to(data_dir)
    logger.info("シンボリックリンク作成: {} → {}", link, data_dir)


def _copy_to_checkpoints(
    id_name: str,
    data_dir: Path,
    flare_root: Path,
) -> None:
    """学習済みモデルを checkpoints/flashavatar/<id_name>/ に (シンボリックリンクで) 配置する。"""
    flare_ckpt = flare_root / "checkpoints" / "flashavatar" / id_name
    point_cloud_src = data_dir / "log" / "point_cloud"

    if not point_cloud_src.exists():
        logger.warning(
            "point_cloud ディレクトリが見つかりません: {}\n"
            "  train.py が正常完了しているか確認してください。",
            point_cloud_src,
        )
        return

    flare_ckpt.mkdir(parents=True, exist_ok=True)
    link = flare_ckpt / "point_cloud"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(point_cloud_src)
    logger.info("checkpoints/flashavatar/{}/point_cloud → {}", id_name, point_cloud_src)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FlashAvatar 学習パイプライン (DECA / SMIRK 対応)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--id_name", required=True, help="学習 ID (データ識別子)")
    p.add_argument("--video", default=None, help="入力動画ファイルパス")
    p.add_argument(
        "--config",
        default=None,
        help="YAML 設定ファイルパス (省略時はデフォルト設定)",
    )
    p.add_argument(
        "--extractor",
        choices=["deca", "smirk"],
        default=None,
        help="特徴抽出器 (YAML の設定を上書き)",
    )
    p.add_argument("--device", default=None, help="CUDA デバイス (YAML 上書き)")
    p.add_argument("--iterations", type=int, default=None, help="学習イテレーション数 (YAML 上書き)")
    p.add_argument("--img_size", type=int, default=None, help="フレーム解像度 px (YAML 上書き)")
    p.add_argument("--target_fps", type=int, default=None, help="FPS 正規化目標 (YAML 上書き)")
    p.add_argument("--skip_extract", action="store_true", help="Step 1 をスキップ")
    p.add_argument("--skip_masks", action="store_true", help="Step 2 をスキップ")
    p.add_argument("--skip_convert", action="store_true", help="Step 3 をスキップ")
    p.add_argument(
        "--test_only",
        action="store_true",
        help="Step 5 のみ実行 (学習済みモデルで推論)",
    )
    return p.parse_args()


def _apply_cli_overrides(cfg: TrainFlashAvatarConfig, args: argparse.Namespace) -> TrainFlashAvatarConfig:
    """CLI 引数で YAML 設定を上書きする (pydantic は immutable なので dict 経由)。"""
    data = cfg.model_dump()
    if args.extractor:
        data["pipeline"]["extractor"] = args.extractor
    if args.device:
        data["pipeline"]["device"] = args.device
    if args.iterations is not None:
        data["flashavatar"]["iterations"] = args.iterations
    if args.img_size is not None:
        data["video"]["img_size"] = args.img_size
    if args.target_fps is not None:
        data["video"]["target_fps"] = args.target_fps
    return TrainFlashAvatarConfig.model_validate(data)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 設定ロード
    if args.config:
        cfg = TrainFlashAvatarConfig.from_yaml(args.config)
        logger.info("設定ファイルを読み込みました: {}", args.config)
    else:
        cfg = TrainFlashAvatarConfig()
        logger.info("デフォルト設定を使用します")

    cfg = _apply_cli_overrides(cfg, args)

    # ステージフラグの解決
    if args.test_only:
        for key in ("extract", "masks", "convert", "train"):
            object.__setattr__(cfg.stages, key, False)
        # pydantic v2 は model_copy で immutable 上書きを推奨するが、
        # ここは stages を dict 経由で再構築する
        data = cfg.model_dump()
        data["stages"] = {"extract": False, "masks": False, "convert": False, "train": False, "test": True}
        cfg = TrainFlashAvatarConfig.model_validate(data)

    if args.skip_extract:
        data = cfg.model_dump(); data["stages"]["extract"] = False
        cfg = TrainFlashAvatarConfig.model_validate(data)
    if args.skip_masks:
        data = cfg.model_dump(); data["stages"]["masks"] = False
        cfg = TrainFlashAvatarConfig.model_validate(data)
    if args.skip_convert:
        data = cfg.model_dump(); data["stages"]["convert"] = False
        cfg = TrainFlashAvatarConfig.model_validate(data)

    flare_root = _FLARE_ROOT

    # ロギング設定
    logger.add(cfg.logging.file, rotation=cfg.logging.rotation, level=cfg.logging.level)

    # パス設定
    data_root = Path(cfg.flashavatar.data_root)
    data_dir = data_root / args.id_name
    fa_abs = (flare_root / cfg.flashavatar.repo_dir).resolve()
    frame_dir = fa_abs / "metrical-tracker" / "output" / args.id_name / "checkpoint"

    logger.info("=" * 70)
    logger.info("  FlashAvatar 学習パイプライン")
    logger.info("=" * 70)
    logger.info("  ID        : {}", args.id_name)
    logger.info("  VIDEO     : {}", args.video or "(スキップ)")
    logger.info("  EXTRACTOR : {}", cfg.pipeline.extractor.upper())
    logger.info("  DEVICE    : {}", cfg.pipeline.device)
    logger.info("  ITERS     : {}", cfg.flashavatar.iterations)
    logger.info("  IMG_SIZE  : {}", cfg.video.img_size)
    logger.info("  FPS 目標  : {}", cfg.video.target_fps or "元動画維持")
    logger.info("  DATA_DIR  : {}", data_dir)
    logger.info("=" * 70)

    # 前提チェック
    _check_prerequisites(cfg, args.video, not cfg.stages.extract, flare_root)

    # Step 1: 抽出
    if cfg.stages.extract:
        logger.info("[Step 1] フレーム抽出 + {} per-frame 特徴抽出...", cfg.pipeline.extractor.upper())
        data_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="flare_train_") as tmp_dir:
            video_path = normalize_video(str(args.video), cfg, Path(tmp_dir))
            _run_extract(video_path, data_dir, cfg, flare_root)
    else:
        logger.info("[Step 1] スキップ")

    # FlashAvatar データ構造のセットアップ
    _setup_fa_symlink(args.id_name, data_dir, cfg, flare_root)

    # Step 2: マスク
    if cfg.stages.masks:
        logger.info("[Step 2] MediaPipe マスク生成...")
        _run_masks(data_dir, cfg, flare_root)
    else:
        logger.info("[Step 2] スキップ")

    # Step 3: .frame 変換
    if cfg.stages.convert:
        logger.info("[Step 3] .pt → .frame 変換...")
        _run_convert(data_dir, frame_dir, cfg, flare_root)
    else:
        logger.info("[Step 3] スキップ")

    # Step 4: 学習
    if cfg.stages.train:
        logger.info(
            "[Step 4] FlashAvatar 学習 (iterations={})...",
            cfg.flashavatar.iterations,
        )
        _run_train(args.id_name, data_dir, cfg, flare_root)
    else:
        logger.info("[Step 4] スキップ")

    # Step 5: 検証動画
    if cfg.stages.test:
        logger.info("[Step 5] 検証動画生成...")
        _run_test(args.id_name, data_dir, cfg, flare_root)
    else:
        logger.info("[Step 5] スキップ")

    # checkpoints/ へのコピー
    _copy_to_checkpoints(args.id_name, data_dir, flare_root)

    logger.info("=" * 70)
    logger.info("  完了")
    logger.info("=" * 70)
    logger.info("  学習済みモデル: checkpoints/flashavatar/{}/point_cloud/", args.id_name)
    logger.info("  検証動画      : {}/log/test.avi", data_dir)
    logger.info("")
    logger.info("  リアルタイム動作:")
    logger.info(
        "    python examples/realtime_extract.py --config configs/realtime_flame.yaml --source 0"
    )


if __name__ == "__main__":
    main()
