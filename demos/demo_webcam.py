#!/usr/bin/env python
"""Webカメラリアルタイム SMIRK + FlashAvatar デモ。

Webカメラまたは動画ファイル (ideal-source モード) から毎フレーム:
    1. MediaPipe で顔検出
    2. SMIRK で FLAME パラメータ抽出
    3. SmirkToFlashAvatarAdapter でパラメータ変換
    4. FlashAvatar で 3DGS レンダリング

3 ペイン表示:
    左: 元カメラ映像
    中: MediaPipe ランドマーク overlay
    右: FlashAvatar レンダリング結果

各ステージのタイマをオーバーレイ表示 (cap / mp / smirk / cvt / render / disp)。
終了時に集計サマリを表示。

V4L2/UVC の落とし穴:
    多くの UVC カメラは USB 2.0 帯域の制約で YUYV では 720p@30fps が出ない。
    本デモは FOURCC=MJPG をデフォルトで設定する。
    BUFFERSIZE/AUTO_EXPOSURE は変更しない (FPS 低下の原因になるため)。

WSL2 環境:
    GPU delegate が使用不可なため起動時に警告し、CPU にフォールバックする。

Controls:
    q: 終了
    s: スナップショットを output/ に保存

Usage:
    python demos/demo_webcam.py \\
        --checkpoint_dir checkpoints/flashavatar/person01

    python demos/demo_webcam.py \\
        --source 0 --mp_delegate gpu \\
        --checkpoint_dir checkpoints/flashavatar/person01

    # ideal-source モード (ファイルから読み込み、FPS 制限なし)
    python demos/demo_webcam.py \\
        --source data/raw/sample.mp4 \\
        --checkpoint_dir checkpoints/flashavatar/person01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

# FLARE ルートを sys.path に追加
_DEMO_DIR = Path(__file__).resolve().parent
_FLARE_ROOT = _DEMO_DIR.parent
if str(_FLARE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLARE_ROOT))

import cv2
import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# WSL2 検出
# ---------------------------------------------------------------------------

def _is_wsl2() -> bool:
    """WSL2 環境かどうかを /proc/version で判定する。"""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webカメラリアルタイム SMIRK + FlashAvatar デモ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        default="0",
        help="フレームソース。整数文字列→カメラインデックス、それ以外→動画ファイルパス",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="./checkpoints/flashavatar/",
        help="FlashAvatar チェックポイントディレクトリ",
    )
    p.add_argument(
        "--smirk_model",
        default="./checkpoints/smirk/SMIRK_em1.pt",
        help="SMIRK チェックポイントパス",
    )
    p.add_argument("--device", default="cuda:0", help="推論デバイス")
    p.add_argument(
        "--mp_delegate",
        choices=["cpu", "gpu"],
        default="cpu",
        help="MediaPipe 推論デバイス (gpu は EGL/NVIDIA 環境のみ有効)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Webカメラキャプチャ幅 (Webカメラ時のみ)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Webカメラキャプチャ高さ (Webカメラ時のみ)",
    )
    p.add_argument(
        "--fourcc",
        default="MJPG",
        help="Webカメラ FOURCC ピクセルフォーマット。デフォルト MJPG が USB 2.0 帯域問題を回避する",
    )
    p.add_argument(
        "--display_size",
        type=int,
        default=480,
        help="表示ペインの高さ (px)",
    )
    p.add_argument(
        "--no_render",
        action="store_true",
        help="FlashAvatar レンダリングをスキップ (SMIRK のみベンチマーク)",
    )
    p.add_argument(
        "--snapshot_dir",
        default="output",
        help="スナップショット保存ディレクトリ",
    )
    p.add_argument(
        "--window",
        default="FLARE Webcam (SMIRK)",
        help="OpenCV ウィンドウタイトル",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# モデルローダ
# ---------------------------------------------------------------------------

def _load_smirk(model_path: str, device: str):
    from flare.extractors.smirk import SMIRKExtractor
    return SMIRKExtractor(
        model_path=model_path,
        device=device,
        smirk_dir=str(_FLARE_ROOT / "third_party" / "smirk"),
    )


def _load_flashavatar(checkpoint_dir: str, device: str):
    from flare.renderers.flashavatar import FlashAvatarRenderer
    renderer = FlashAvatarRenderer(model_path=checkpoint_dir, device=device)
    renderer.setup()
    return renderer


# ---------------------------------------------------------------------------
# オーバーレイ描画
# ---------------------------------------------------------------------------

def _draw_overlay(
    canvas: np.ndarray,
    fps: float,
    timings: dict[str, float],
    face_ok: bool,
    mp_delegate: str,
    device_str: str,
) -> None:
    """タイミング情報をキャンバスにオーバーレイする。"""
    pad = 8

    def t(k: str) -> str:
        v = timings.get(k)
        return "  --" if v is None else f"{v:5.1f}"

    total = sum(v for v in timings.values())
    lines = [
        f"FPS:{fps:5.1f}  total:{total:5.1f}ms",
        f"cap:{t('cap')}  mp:{t('mp')}  smirk:{t('smirk')}",
        f"cvt:{t('cvt')}  render:{t('render')}",
        f"disp:{t('disp')}",
        f"mp:{mp_delegate}  dev:{device_str}",
        f"face:{'OK' if face_ok else '---'}",
    ]
    for i, line in enumerate(lines):
        y = pad + 18 * (i + 1)
        cv2.putText(canvas, line, (pad + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 230, 30), 1, cv2.LINE_AA)


def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """MediaPipe ランドマーク (468 点) を frame 上に描画して返す。"""
    out = frame.copy()
    h, w = out.shape[:2]
    for pt in landmarks:
        x, y = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
    return out


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # WSL2 チェック
    if _is_wsl2() and args.mp_delegate == "gpu":
        logger.warning(
            "WSL2 環境では MediaPipe GPU delegate は使用できません。\n"
            "  CPU に自動フォールバックします。"
        )
        args.mp_delegate = "cpu"

    # デバイス確定
    if not torch.cuda.is_available() and args.device != "cpu":
        logger.warning("CUDA が利用できません。CPU に切り替えます。")
        args.device = "cpu"
    device = args.device

    # モデルロード
    logger.info("SMIRK ロード中: {}", args.smirk_model)
    if not Path(args.smirk_model).exists():
        logger.error(
            "SMIRK チェックポイントが見つかりません: {}\n"
            "  bash install/setup_smirk.sh を実行してください。",
            args.smirk_model,
        )
        sys.exit(1)
    smirk = _load_smirk(args.smirk_model, device)

    from flare.converters.smirk_to_flashavatar import SmirkToFlashAvatarAdapter
    adapter = SmirkToFlashAvatarAdapter()

    renderer = None
    if not args.no_render:
        if not Path(args.checkpoint_dir).exists():
            logger.error(
                "FlashAvatar チェックポイントが見つかりません: {}\n"
                "  scripts/train_flashavatar.py で学習してから実行してください。",
                args.checkpoint_dir,
            )
            sys.exit(1)
        logger.info("FlashAvatar ロード中: {}", args.checkpoint_dir)
        renderer = _load_flashavatar(args.checkpoint_dir, device)

    from flare.utils.face_detect import FaceDetector
    face_detector = FaceDetector()

    # ソース解決
    try:
        source_handle: object = int(args.source)
        is_webcam = True
    except ValueError:
        source_handle = args.source
        is_webcam = False

    cap = cv2.VideoCapture(source_handle)
    if not cap.isOpened():
        logger.error("ソースを開けません: {}", args.source)
        sys.exit(1)

    if is_webcam:
        if args.fourcc:
            fourcc_val = cv2.VideoWriter_fourcc(*args.fourcc.upper())
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_val)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = (
        "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        if actual_fourcc else "?"
    )
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "ソース: {}  mode: {}  fourcc: {}  size: {}x{}  reported_fps: {:.1f}",
        args.source,
        "webcam" if is_webcam else "ideal-video",
        fourcc_str, actual_w, actual_h, actual_fps,
    )

    os.makedirs(args.snapshot_dir, exist_ok=True)
    ds = args.display_size
    fps_window: deque[float] = deque(maxlen=30)
    frame_idx = 0

    agg: dict[str, float] = {k: 0.0 for k in ("cap", "mp", "smirk", "cvt", "render", "disp")}
    agg_face = 0
    first_frame_sec: Optional[float] = None
    t_run0 = time.perf_counter()

    logger.info("q: 終了  s: スナップショット")

    last_render: Optional[np.ndarray] = None
    last_landmarks: Optional[np.ndarray] = None

    try:
        while True:
            t_frame0 = time.perf_counter()
            timings: dict[str, float] = {}

            # --- Capture ---
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.info("フレーム読み込み終了")
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)
            timings["cap"] = (time.perf_counter() - t0) * 1000.0

            face_ok = False
            render_img: Optional[np.ndarray] = None
            landmarks_norm: Optional[np.ndarray] = None

            # --- MediaPipe ---
            t0 = time.perf_counter()
            try:
                bbox = face_detector.detect(frame)
                face_ok = bbox is not None
            except Exception:
                face_ok = False
            timings["mp"] = (time.perf_counter() - t0) * 1000.0

            if face_ok:
                # MediaPipe landmark overlay 用の正規化ランドマーク
                # (FaceDetector は bbox のみ返すので簡易的に bbox からランドマーク推定)
                last_landmarks = None

                # --- SMIRK ---
                t0 = time.perf_counter()
                try:
                    cropped = face_detector.crop_and_align(
                        frame, bbox, size=224, margin_scale=1.25
                    )
                    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    tensor = (
                        torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                        .unsqueeze(0).to(device)
                    )
                    with torch.no_grad():
                        params = smirk.extract(tensor)
                    timings["smirk"] = (time.perf_counter() - t0) * 1000.0

                    # --- 変換 ---
                    t0 = time.perf_counter()
                    flash_params = adapter.convert(params)
                    timings["cvt"] = (time.perf_counter() - t0) * 1000.0

                    # --- レンダリング ---
                    if renderer is not None:
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            rendered = renderer.render(flash_params)
                        rendered_rgb = (
                            rendered[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
                        ).astype(np.uint8)
                        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                        last_render = cv2.resize(rendered_bgr, (ds, ds))
                        timings["render"] = (time.perf_counter() - t0) * 1000.0

                    agg_face += 1

                except Exception as e:
                    logger.debug("frame {}: 処理失敗 ({})", frame_idx, e)
                    face_ok = False

            # --- 表示 ---
            t0 = time.perf_counter()
            orig_h, orig_w = frame.shape[:2]
            scale = ds / orig_h
            left_pane = cv2.resize(frame, (int(orig_w * scale), ds))

            # 中ペイン: bbox を描画した入力フレーム
            # FaceDetector.detect() は (x1, y1, x2, y2) を返す
            mid_pane_src = frame.copy()
            if face_ok and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(mid_pane_src, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mid_pane = cv2.resize(mid_pane_src, (ds, ds))

            right_pane = last_render if last_render is not None else np.zeros((ds, ds, 3), np.uint8)

            canvas = np.hstack([left_pane, mid_pane, right_pane])
            timings["disp"] = (time.perf_counter() - t0) * 1000.0

            t_frame = time.perf_counter() - t_frame0
            fps_window.append(t_frame)
            fps = len(fps_window) / max(sum(fps_window), 1e-6)

            _draw_overlay(canvas, fps, timings, face_ok, args.mp_delegate, device)

            cv2.imshow(args.window, canvas)
            key = cv2.waitKey(1) & 0xFF

            if first_frame_sec is None:
                first_frame_sec = t_frame
            else:
                for k, v in timings.items():
                    agg[k] += v / 1000.0

            if key == ord("q"):
                break
            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                snap = os.path.join(args.snapshot_dir, f"snapshot_{ts}.png")
                cv2.imwrite(snap, canvas)
                logger.info("スナップショット保存: {}", snap)

            frame_idx += 1

    finally:
        cap.release()
        face_detector.release()
        cv2.destroyAllWindows()

        t_run = time.perf_counter() - t_run0
        e2e_fps = frame_idx / t_run if t_run > 0 else 0.0
        bench = max(0, frame_idx - 1)

        def _avg(k: str) -> float:
            return 1000.0 * agg[k] / bench if bench else 0.0

        stages = ("cap", "mp", "smirk", "cvt", "render", "disp")
        first_ms = (first_frame_sec * 1000.0) if first_frame_sec is not None else 0.0
        logger.info(
            "=== サマリ: {} フレーム / {:.1f}s  e2e={:.1f} fps  face_ok={}",
            frame_idx, t_run, e2e_fps, agg_face,
        )
        logger.info("first-frame: {:.1f}ms (集計から除外)", first_ms)
        logger.info(
            "平均: " + "  ".join(f"{k}={_avg(k):.1f}ms" for k in stages)
        )
        logger.info("mp_delegate={}  device={}", args.mp_delegate, device)


if __name__ == "__main__":
    main()
