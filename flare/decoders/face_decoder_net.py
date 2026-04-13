"""学習ベースの顔画像デコーダネットワーク。

FLAME 3DMM パラメータ (expression + pose) と話者のソース画像から、
対象フレームの顔画像を生成する軽量ニューラルレンダラ。

アーキテクチャ概要:

    ┌─────────────────┐
    │ Source Image     │
    │ (3, 256, 256)   │
    └───────┬─────────┘
            │ Encoder (ResNet-18 features)
            ▼
    ┌───────────────────┐      ┌──────────────────┐
    │ Image Features    │      │ FLAME Params     │
    │ (512, 8, 8)      │      │ (D_cond,)        │
    └───────┬───────────┘      └─────────┬────────┘
            │                            │ MLP
            │                     ┌──────▼────────┐
            │                     │ Style Vector  │
            │                     │ (512,)        │
            │                     └──────┬────────┘
            │                            │
            ▼                            ▼
    ┌───────────────────────────────────────────┐
    │           AdaIN Decoder Blocks            │
    │ (512,8,8)→(256,16,16)→(128,32,32)→...    │
    │           → (3, 256, 256)                 │
    └───────────────────────────────────────────┘

学習方針:
    - 対象人物の動画から frame + DECA params のペアを作成
    - L1 再構成損失 + perceptual 損失 (VGG-16 feature matching)
    - 1 人あたり数千フレーム × 数時間の学習で実用品質
    - 推論速度: ~200 FPS @ 256×256 on RTX 3090

Note:
    - FlashAvatar (3DGS) や PIRender (flow-based) に比べ品質は劣るが、
      外部リポジトリ不要で自己完結した可視化が可能
    - LHG 研究のプロトタイピングや抽出結果の定性評価に最適
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (Huang & Belongie, 2017)。

    スタイルベクトルからチャンネルごとの mean / std を予測し、
    コンテンツ特徴量を正規化 → 再スケーリングする。

    Args:
        n_channels: 入力特徴マップのチャンネル数。
        style_dim: スタイルベクトルの次元数。
    """

    def __init__(self, n_channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channels, affine=False)
        self.fc = nn.Linear(style_dim, n_channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """AdaIN forward。

        Args:
            x: コンテンツ特徴マップ ``(B, C, H, W)``。
            style: スタイルベクトル ``(B, style_dim)``。

        Returns:
            スタイル適用済み特徴マップ ``(B, C, H, W)``。
        """
        params = self.fc(style)  # (B, C*2)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        out = self.norm(x)
        return (1 + gamma) * out + beta


class DecoderBlock(nn.Module):
    """AdaIN 付きアップサンプリングブロック。

    Bilinear upsample → Conv → AdaIN → LeakyReLU → Conv → AdaIN → LeakyReLU

    Args:
        in_ch: 入力チャンネル数。
        out_ch: 出力チャンネル数。
        style_dim: スタイルベクトル次元数。
    """

    def __init__(self, in_ch: int, out_ch: int, style_dim: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adain1 = AdaIN(out_ch, style_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adain2 = AdaIN(out_ch, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv1(x)
        x = self.adain1(x, style)
        x = self.act(x)
        x = self.conv2(x)
        x = self.adain2(x, style)
        x = self.act(x)
        return x


class SourceEncoder(nn.Module):
    """ソース画像エンコーダ（ResNet-18 ベース）。

    ImageNet pretrained ResNet-18 の前半を特徴抽出器として使用。
    出力は ``(B, 512, 8, 8)`` の特徴マップ。

    Args:
        pretrained: ImageNet pretrained weights を使用するか。
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights

            if pretrained:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                backbone = resnet18(weights=None)
        except ImportError:
            # torchvision が無い場合はフォールバック
            backbone = self._build_simple_encoder()
            self.layers = backbone
            self._use_simple = True
            return

        # conv1 ~ layer4 (avgpool / fc を除く)
        self.layers = nn.Sequential(
            backbone.conv1,   # → (64, 128, 128)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # → (64, 64, 64)
            backbone.layer1,   # → (64, 64, 64)
            backbone.layer2,   # → (128, 32, 32)
            backbone.layer3,   # → (256, 16, 16)
            backbone.layer4,   # → (512, 8, 8)
        )
        self._use_simple = False

    @staticmethod
    def _build_simple_encoder() -> nn.Sequential:
        """torchvision 不要のシンプルなエンコーダ。"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ソース画像をエンコードする。

        Args:
            x: RGB 画像 ``(B, 3, 256, 256)``、値域 ``[0, 1]``。

        Returns:
            特徴マップ ``(B, 512, 8, 8)``。
        """
        return self.layers(x)


class ConditionEncoder(nn.Module):
    """FLAME パラメータ → スタイルベクトル変換 MLP。

    Args:
        cond_dim: 入力条件ベクトル次元数 (expression + pose 等)。
        style_dim: 出力スタイルベクトル次元数。
    """

    def __init__(self, cond_dim: int, style_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, style_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """条件ベクトルをスタイルベクトルに変換。

        Args:
            cond: FLAME パラメータ ``(B, cond_dim)``。

        Returns:
            スタイルベクトル ``(B, style_dim)``。
        """
        return self.net(cond)


class FaceDecoderNet(nn.Module):
    """FLAME パラメータ条件付き顔画像デコーダ。

    ソース画像から外見特徴を抽出し、FLAME パラメータ（表情・姿勢）で
    条件付けして対象フレームの顔画像を生成する。

    入力:
        - source_image: ソース (アイデンティティ) 画像 ``(B, 3, 256, 256)``
        - condition: FLAME パラメータ ``(B, cond_dim)``

    出力:
        - 生成画像 ``(B, 3, 256, 256)``、値域 ``[0, 1]``

    Args:
        cond_dim: FLAME 条件ベクトル次元数。DECA なら exp(50)+pose(6)=56、
            DECA+jaw なら exp(50)+global(3)+jaw(3)=56。
        style_dim: 内部スタイルベクトル次元数。
        pretrained_encoder: ソースエンコーダに pretrained weights を使用するか。
    """

    def __init__(
        self,
        cond_dim: int = 56,
        style_dim: int = 512,
        pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.source_encoder = SourceEncoder(pretrained=pretrained_encoder)
        self.cond_encoder = ConditionEncoder(cond_dim, style_dim)

        # Decoder: (512, 8, 8) → (3, 256, 256)
        # 8→16→32→64→128→256 = 5 upsampling blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(512, 256, style_dim),   # 8 → 16
                DecoderBlock(256, 128, style_dim),   # 16 → 32
                DecoderBlock(128, 64, style_dim),    # 32 → 64
                DecoderBlock(64, 32, style_dim),     # 64 → 128
                DecoderBlock(32, 16, style_dim),     # 128 → 256
            ]
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        source_image: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """条件付き顔画像を生成する。

        Args:
            source_image: ソース画像 ``(B, 3, 256, 256)``、値域 ``[0, 1]``。
            condition: FLAME パラメータ ``(B, cond_dim)``。

        Returns:
            生成画像 ``(B, 3, 256, 256)``、値域 ``[0, 1]``。
        """
        features = self.source_encoder(source_image)  # (B, 512, 8, 8)
        style = self.cond_encoder(condition)           # (B, style_dim)

        x = features
        for block in self.decoder_blocks:
            x = block(x, style)

        return self.to_rgb(x)

    @staticmethod
    def build_condition_vector(
        expression: torch.Tensor,
        global_pose: torch.Tensor,
        jaw_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DECA 出力から条件ベクトルを構築するユーティリティ。

        Args:
            expression: 表情係数 ``(B, E)``。
            global_pose: グローバル回転 ``(B, 3)``。
            jaw_pose: 顎回転 ``(B, 3)``。None の場合は含めない。

        Returns:
            条件ベクトル ``(B, E+3[+3])``。
        """
        parts = [expression, global_pose]
        if jaw_pose is not None:
            parts.append(jaw_pose)
        return torch.cat(parts, dim=-1)


class PerceptualLoss(nn.Module):
    """VGG-16 ベースの perceptual loss。

    VGG-16 の各層 (relu1_2, relu2_2, relu3_3, relu4_3) の特徴マップ間の
    L1 距離を計算する。画像の高周波構造やテクスチャの一致度を測る。

    Args:
        layers: 使用する VGG 層インデックスのリスト。
        weights: 各層の損失重み。None なら均等。
    """

    _VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    _VGG_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    def __init__(
        self,
        layers: Optional[list[int]] = None,
        weights: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self._layers = layers or [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        self._weights = weights or [1.0] * len(self._layers)

        try:
            from torchvision.models import vgg16, VGG16_Weights

            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        except ImportError:
            raise ImportError(
                "torchvision is required for PerceptualLoss. "
                "Install with: pip install torchvision"
            )

        # 必要な層までのサブネットワークを構築
        self.slices = nn.ModuleList()
        prev = 0
        for idx in sorted(self._layers):
            self.slices.append(nn.Sequential(*list(vgg.children())[prev : idx + 1]))
            prev = idx + 1

        # VGG は固定
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """perceptual loss を計算する。

        Args:
            pred: 予測画像 ``(B, 3, H, W)``、値域 ``[0, 1]``。
            target: 正解画像 ``(B, 3, H, W)``、値域 ``[0, 1]``。

        Returns:
            スカラ損失テンソル。
        """
        mean = self._VGG_MEAN.to(pred.device)
        std = self._VGG_STD.to(pred.device)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred_norm
        x_target = target_norm

        for i, slc in enumerate(self.slices):
            x_pred = slc(x_pred)
            with torch.no_grad():
                x_target = slc(x_target)
            loss = loss + self._weights[i] * F.l1_loss(x_pred, x_target)

        return loss
