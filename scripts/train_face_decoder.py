#!/usr/bin/env python
"""顔画像デコーダの学習スクリプト。

対象人物の動画から per-frame に DECA パラメータを抽出し、
(FLAME パラメータ, 顔画像) のペアで FaceDecoderNet を学習する。

学習済みモデルを使えば、LHG パイプラインで抽出した npz 特徴量から
フォトリアルな顔画像を再構成できる。

使用方法
--------

**ステップ 1: 学習データの準備**

  対象人物が映っている動画を 1 本用意する (数分〜数十分)。
  DECA チェックポイントと FLAME モデルが必要。

**ステップ 2: 学習の実行**

.. code-block:: bash

    python scripts/train_face_decoder.py \\
        --video ./data/target_person.mp4 \\
        --deca-path ./checkpoints/deca_model.tar \\
        --output-dir ./checkpoints/face_decoder/person01/ \\
        --config configs/train_face_decoder.yaml \\
        --device cuda:0

**ステップ 3: 学習済みモデルの利用**

  → ``scripts/demo_visualize.py`` で npz から可視化

処理フロー
----------
1. 動画を読み込み、MediaPipe で顔検出 → crop (224×224)
2. DECA で per-frame 3DMM パラメータ抽出
3. ソース画像を 1 枚選定（シーケンス中央付近の正面顔）
4. (source_image, FLAME_params, target_image) トリプレットで学習
5. L1 + Perceptual Loss (VGG-16) で最適化
6. チェックポイントを保存

入出力
------
入力:
    - 動画ファイル (mp4/avi)
    - DECA チェックポイント
    - 学習設定 YAML (オプション)

出力:
    - ``output_dir/face_decoder.pth`` — 学習済みモデル
    - ``output_dir/source_image.png`` — ソース画像
    - ``output_dir/train_config.yaml`` — 学習時の設定記録
    - ``output_dir/samples/`` — epoch ごとのサンプル画像
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VideoFrameDataset(Dataset):
    """動画フレーム + DECA パラメータのペアデータセット。

    ``prepare()`` でデータを事前抽出してメモリに保持する。

    Attributes:
        frames: 正規化済み顔画像 ``(N, 3, 256, 256)`` float32。
        conditions: FLAME 条件ベクトル ``(N, cond_dim)`` float32。
        source_image: ソース画像テンソル ``(3, 256, 256)``。
    """

    def __init__(self) -> None:
        self.frames: Optional[torch.Tensor] = None
        self.conditions: Optional[torch.Tensor] = None
        self.source_image: Optional[torch.Tensor] = None
        self._cond_dim: int = 0

    def prepare(
        self,
        video_path: str,
        deca_model_path: str,
        device: str = "cuda:0",
        max_frames: int = 10000,
        crop_size: int = 224,
        target_size: int = 256,
    ) -> None:
        """動画から学習データを抽出する。

        Args:
            video_path: 入力動画パス。
            deca_model_path: DECA チェックポイントパス。
            device: DECA 推論デバイス。
            max_frames: 最大フレーム数。
            crop_size: 顔クロップサイズ (DECA 入力)。
            target_size: デコーダ入出力画像サイズ。
        """
        from flare.extractors.deca import DECAExtractor
        from flare.utils.face_detect import FaceDetector

        print(f"[Prepare] Loading DECA from {deca_model_path} ...")
        extractor = DECAExtractor(model_path=deca_model_path, device=device)
        face_detector = FaceDetector()

        print(f"[Prepare] Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        all_frames: list[np.ndarray] = []
        all_conditions: list[np.ndarray] = []
        frame_idx = 0
        dev = torch.device(device)

        while len(all_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 顔検出
            try:
                bbox = face_detector.detect(frame)
            except Exception:
                continue
            if bbox is None:
                continue

            # Crop for DECA (224x224)
            try:
                cropped = face_detector.crop_and_align(
                    frame, bbox, size=crop_size
                )
            except Exception:
                continue

            # DECA 推論
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            )
            tensor = tensor.unsqueeze(0).to(dev)

            try:
                with torch.no_grad():
                    params = extractor.extract(tensor)
            except Exception:
                continue

            # 条件ベクトル: exp(50) + global_pose(3) + jaw_pose(3) = 56
            exp = params["exp"][0].detach().cpu().numpy()       # (50,)
            pose = params["pose"][0].detach().cpu().numpy()     # (6,)
            global_pose = pose[:3]
            jaw_pose = pose[3:6]
            cond = np.concatenate([exp, global_pose, jaw_pose])  # (56,)

            # デコーダ用にリサイズ (256x256)
            target_img = cv2.resize(
                cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )
            target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            target_float = target_rgb.astype(np.float32) / 255.0

            all_frames.append(target_float)
            all_conditions.append(cond.astype(np.float32))

            if len(all_frames) % 500 == 0:
                print(
                    f"  Extracted {len(all_frames)} frames "
                    f"(scanned {frame_idx})"
                )

        cap.release()

        if len(all_frames) == 0:
            raise RuntimeError("No valid frames extracted from video")

        print(f"[Prepare] Total: {len(all_frames)} frames from {frame_idx} scanned")

        # ソース画像: シーケンス中央付近を選択
        mid = len(all_frames) // 2
        self.source_image = (
            torch.from_numpy(all_frames[mid]).permute(2, 0, 1)  # (3, 256, 256)
        )

        # テンソル化
        frames_arr = np.stack(all_frames, axis=0)  # (N, 256, 256, 3)
        conds_arr = np.stack(all_conditions, axis=0)  # (N, 56)

        self.frames = torch.from_numpy(frames_arr).permute(0, 3, 1, 2)  # (N, 3, H, W)
        self.conditions = torch.from_numpy(conds_arr)
        self._cond_dim = conds_arr.shape[1]

        print(
            f"[Prepare] Dataset ready: frames={self.frames.shape}, "
            f"cond_dim={self._cond_dim}"
        )

    def prepare_from_extracted(
        self,
        npz_dir: str,
        video_path: str,
        crop_size: int = 224,
        target_size: int = 256,
    ) -> None:
        """抽出済み npz + 元動画から学習データを構築する。

        lhg-extract で抽出済みの npz がある場合に使用。
        npz のデノーマライズ済みパラメータと動画フレームをペアリングする。

        Args:
            npz_dir: 抽出済み npz ディレクトリ (例: movements/data001/comp/)。
            video_path: 元動画のパス。
            crop_size: 顔クロップサイズ。
            target_size: 出力画像サイズ。
        """
        from flare.utils.face_detect import FaceDetector

        face_detector = FaceDetector()

        # npz ファイル読み込み
        npz_path = Path(npz_dir)
        npz_files = sorted(npz_path.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No npz files in {npz_dir}")

        # 全シーケンスのパラメータと区間を収集
        segments: list[dict[str, np.ndarray]] = []
        for f in npz_files:
            d = np.load(str(f))
            section = d["section"]  # [start, end]
            # デノーマライズ
            angle = d["angle"] * d["angle_std"] + d["angle_mean"]
            exp = d["expression"] * d["expression_std"] + d["expression_mean"]
            segments.append({
                "start": int(section[0]),
                "end": int(section[1]),
                "angle": angle,
                "expression": exp,
                "jaw_pose": d["jaw_pose"] if "jaw_pose" in d else None,
            })

        # 動画フレームを走査
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        all_frames: list[np.ndarray] = []
        all_conditions: list[np.ndarray] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # このフレームがどのセグメントに属するか
            for seg in segments:
                if seg["start"] <= frame_idx <= seg["end"]:
                    local_idx = frame_idx - seg["start"]
                    if local_idx >= len(seg["expression"]):
                        continue

                    # 顔検出 + crop
                    try:
                        bbox = face_detector.detect(frame)
                    except Exception:
                        break
                    if bbox is None:
                        break

                    try:
                        cropped = face_detector.crop_and_align(
                            frame, bbox, size=crop_size
                        )
                    except Exception:
                        break

                    target_img = cv2.resize(
                        cropped, (target_size, target_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

                    exp_vec = seg["expression"][local_idx]
                    angle_vec = seg["angle"][local_idx]
                    parts = [exp_vec, angle_vec]
                    if seg["jaw_pose"] is not None:
                        parts.append(seg["jaw_pose"][local_idx])
                    cond = np.concatenate(parts).astype(np.float32)

                    all_frames.append(target_rgb.astype(np.float32) / 255.0)
                    all_conditions.append(cond)
                    break

            frame_idx += 1

        cap.release()

        if not all_frames:
            raise RuntimeError("No frames matched with npz segments")

        mid = len(all_frames) // 2
        self.source_image = (
            torch.from_numpy(all_frames[mid]).permute(2, 0, 1)
        )
        self.frames = (
            torch.from_numpy(np.stack(all_frames)).permute(0, 3, 1, 2)
        )
        self.conditions = torch.from_numpy(np.stack(all_conditions))
        self._cond_dim = self.conditions.shape[1]

        print(
            f"[Prepare] From npz: {len(all_frames)} frames, "
            f"cond_dim={self._cond_dim}"
        )

    @property
    def cond_dim(self) -> int:
        return self._cond_dim

    def __len__(self) -> int:
        return 0 if self.frames is None else self.frames.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        assert self.frames is not None and self.conditions is not None
        return {
            "target_image": self.frames[idx],
            "condition": self.conditions[idx],
        }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train(
    dataset: VideoFrameDataset,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    perceptual_weight: float = 0.5,
    device: str = "cuda:0",
    save_interval: int = 10,
    sample_interval: int = 5,
) -> None:
    """FaceDecoderNet の学習を実行する。

    Args:
        dataset: 準備済みの VideoFrameDataset。
        output_dir: 出力ディレクトリ。
        epochs: エポック数。
        batch_size: バッチサイズ。
        lr: 学習率。
        perceptual_weight: perceptual loss の重み。0 なら L1 のみ。
        device: 学習デバイス。
        save_interval: チェックポイント保存間隔 (エポック)。
        sample_interval: サンプル画像保存間隔 (エポック)。
    """
    from flare.decoders.face_decoder_net import FaceDecoderNet, PerceptualLoss

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    samples_dir = out_path / "samples"
    samples_dir.mkdir(exist_ok=True)

    # ソース画像を保存
    assert dataset.source_image is not None
    src_img = (dataset.source_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(
        str(out_path / "source_image.png"),
        cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR),
    )

    dev = torch.device(device)
    cond_dim = dataset.cond_dim

    # モデル構築
    model = FaceDecoderNet(
        cond_dim=cond_dim,
        style_dim=512,
        pretrained_encoder=True,
    ).to(dev)

    # 損失関数
    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn: Optional[PerceptualLoss] = None
    if perceptual_weight > 0:
        try:
            perceptual_loss_fn = PerceptualLoss().to(dev)
            perceptual_loss_fn.eval()
            print("[Train] Perceptual loss enabled (VGG-16)")
        except ImportError:
            print("[Train] torchvision not available, using L1 only")
            perceptual_weight = 0.0

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    source_img_tensor = dataset.source_image.unsqueeze(0).to(dev)  # (1, 3, 256, 256)

    print(f"[Train] Start: {len(dataset)} samples, {epochs} epochs, "
          f"batch={batch_size}, lr={lr}")
    print(f"[Train] Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            target = batch["target_image"].to(dev)
            cond = batch["condition"].to(dev)
            bs = target.shape[0]

            # ソース画像をバッチ分複製
            source_batch = source_img_tensor.expand(bs, -1, -1, -1)

            pred = model(source_batch, cond)

            loss = l1_loss_fn(pred, target)
            if perceptual_loss_fn is not None and perceptual_weight > 0:
                loss = loss + perceptual_weight * perceptual_loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:04d}/{epochs} | "
            f"loss={avg_loss:.6f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        # サンプル画像保存
        if epoch % sample_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample_cond = dataset.conditions[:4].to(dev)
                sample_target = dataset.frames[:4].to(dev)
                sample_src = source_img_tensor.expand(sample_cond.shape[0], -1, -1, -1)
                sample_pred = model(sample_src, sample_cond)

                # 横に並べて保存: source | pred | target
                rows = []
                for i in range(min(4, sample_cond.shape[0])):
                    src_np = (source_img_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pred_np = (sample_pred[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    tgt_np = (sample_target[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    row = np.concatenate([src_np, pred_np, tgt_np], axis=1)
                    rows.append(row)

                grid = np.concatenate(rows, axis=0)
                grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(samples_dir / f"epoch_{epoch:04d}.png"), grid_bgr)

        # チェックポイント保存
        if epoch % save_interval == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "cond_dim": cond_dim,
            }
            torch.save(ckpt, str(out_path / f"checkpoint_epoch{epoch:04d}.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cond_dim": cond_dim,
                    "best_loss": best_loss,
                    "epoch": epoch,
                },
                str(out_path / "face_decoder.pth"),
            )

    # 学習設定の記録
    train_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "perceptual_weight": perceptual_weight,
        "cond_dim": cond_dim,
        "num_samples": len(dataset),
        "best_loss": float(best_loss),
    }
    with open(out_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"[Train] Done. Best loss: {best_loss:.6f}")
    print(f"[Train] Model saved to: {out_path / 'face_decoder.pth'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceDecoderNet 学習スクリプト — DECA パラメータからの顔画像復元"
    )
    parser.add_argument(
        "--video", required=True, help="学習用動画ファイルパス"
    )
    parser.add_argument(
        "--deca-path",
        default="./checkpoints/deca_model.tar",
        help="DECA チェックポイントパス",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/face_decoder/",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--npz-dir",
        default=None,
        help="抽出済み npz ディレクトリ。指定時は動画 + npz からデータ構築",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="学習設定 YAML ファイルパス (オプション)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--perceptual-weight", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # YAML config による上書き
    if args.config is not None and _HAS_YAML:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, val in cfg.get("training", {}).items():
            key_underscore = key.replace("-", "_")
            if hasattr(args, key_underscore) and val is not None:
                setattr(args, key_underscore, val)

    dataset = VideoFrameDataset()

    if args.npz_dir is not None:
        dataset.prepare_from_extracted(
            npz_dir=args.npz_dir,
            video_path=args.video,
        )
    else:
        dataset.prepare(
            video_path=args.video,
            deca_model_path=args.deca_path,
            device=args.device,
            max_frames=args.max_frames,
        )

    train(
        dataset=dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        perceptual_weight=args.perceptual_weight,
        device=args.device,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
    )


if __name__ == "__main__":
    main()
