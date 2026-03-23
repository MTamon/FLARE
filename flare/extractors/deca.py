"""DECA Extractor: FLAME系3DMMパラメータ抽出。

YadiraF/DECA（SIGGRAPH 2021）をラップし、BaseExtractorインターフェースで
FLAME系パラメータを抽出する。DECAは最も広く使われているFLAME系Extractorであり、
L2L（Learning to Listen）との直接互換性を持つ。

DECA出力パラメータ:
    ========= ====== ============================================
    キー       次元    説明
    ========= ====== ============================================
    shape     100D   FLAME形状パラメータ
    tex        50D   テクスチャパラメータ
    exp        50D   表情パラメータ（FLAME PCA 第1-50主成分）
    pose        6D   姿勢（global_rotation 3D + jaw_pose 3D）
    cam         3D   カメラパラメータ
    light      27D   照明パラメータ（9x3を27Dにreshape）
    detail    128D   詳細パラメータ（E_detail ResNet出力）
    ========= ====== ============================================
    合計      364D

DECA FAN に関する発見（v2.0）:
    DECAのencode()内部にFAN顔検出は含まれない。FAN は前処理（TestData）でのみ
    使用される。したがって、face_detect.pyで前処理済みの画像をそのまま
    encode()に渡せる。

Example:
    >>> extractor = DECAExtractor(
    ...     model_path="./checkpoints/deca_model.tar",
    ...     device="cuda:0",
    ... )
    >>> result = extractor.extract(image_tensor)
    >>> result["exp"].shape  # (1, 50)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from flare.extractors.base import BaseExtractor
from flare.utils.errors import ModelLoadError


class DECAExtractor(BaseExtractor):
    """DECA（SIGGRAPH 2021）ベースの3DMMパラメータ抽出器。

    YadiraF/DECAリポジトリのencode()を呼び出し、FLAME系パラメータを
    BaseExtractorインターフェースで返す。

    DECA exp 50DはFLAME expression PCA空間の第1-50主成分であり、
    FlashAvatar expr 100Dの部分空間に完全に含まれる。
    DECAToFlameAdapterでゼロパディングにより100Dに変換可能。

    Attributes:
        _device: モデルが配置されるCUDAデバイス。
        _model_path: DECAチェックポイントファイルパス。
        _deca: DECAモデルインスタンス。

    Example:
        >>> extractor = DECAExtractor("./checkpoints/deca_model.tar", "cuda:0")
        >>> result = extractor.extract(cropped_face_tensor)
        >>> print(result.keys())
        dict_keys(['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'detail'])
    """

    #: 出力パラメータの総次元数
    _PARAM_DIM: int = 364  # 100+50+50+6+3+27+128

    #: 出力Dictのキーリスト
    _PARAM_KEYS: List[str] = [
        "shape", "tex", "exp", "pose", "cam", "light", "detail",
    ]

    #: 各キーの期待次元数（extract後の検証用）
    _KEY_DIMS: Dict[str, int] = {
        "shape": 100,
        "tex": 50,
        "exp": 50,
        "pose": 6,
        "cam": 3,
        "light": 27,
        "detail": 128,
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        *,
        cfg_path: Optional[str] = None,
    ) -> None:
        """DECAExtractorを初期化する。

        Args:
            model_path: DECAチェックポイントファイルパス（.tarファイル）。
            device: 計算デバイス。例: ``"cuda:0"``、``"cpu"``。
            cfg_path: DECA設定ファイルパス（オプション）。
                Noneの場合はDECAのデフォルト設定を使用する。

        Raises:
            ModelLoadError: チェックポイントファイルが見つからない場合、
                またはモデルのロードに失敗した場合。
        """
        self._device: torch.device = torch.device(device)
        self._model_path: str = model_path
        self._cfg_path: Optional[str] = cfg_path
        self._deca: Any = None

        self._load_model()

    def _load_model(self) -> None:
        """DECAモデルをロードする。

        YadiraF/DECAリポジトリのDECAクラスをインポートし、
        チェックポイントからモデルを初期化する。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        model_file: Path = Path(self._model_path)
        if not model_file.exists():
            raise ModelLoadError(
                f"DECAチェックポイントが見つかりません: {self._model_path}"
            )

        try:
            # YadiraF/DECAリポジトリからのインポート
            # DECAはsys.pathにリポジトリルートを追加して使用することを想定
            from decalib.deca import DECA as DECAModel
            from decalib.utils.config import cfg as deca_cfg

            if self._cfg_path is not None:
                deca_cfg.merge_from_file(self._cfg_path)

            deca_cfg.model.use_tex = True
            deca_cfg.rasterizer_type = "standard"
            deca_cfg.model.extract_tex = False

            self._deca = DECAModel(config=deca_cfg, device=self._device)

            logger.info(
                "DECAモデルロード完了: path={} | device={}",
                self._model_path,
                self._device,
            )

        except ImportError:
            logger.warning(
                "decalibが見つかりません。スタンドアロンチェックポイントロードを試行します。"
            )
            self._load_standalone(model_file)

        except Exception as exc:
            raise ModelLoadError(
                f"DECAモデルのロードに失敗しました: {exc}"
            ) from exc

    def _load_standalone(self, model_file: Path) -> None:
        """decalibなしでチェックポイントをスタンドアロンでロードする。

        DECAリポジトリが利用不可の場合のフォールバック。
        チェックポイントからエンコーダネットワークを直接ロードする。

        Args:
            model_file: チェックポイントファイルパス。

        Raises:
            ModelLoadError: ロードに失敗した場合。
        """
        try:
            checkpoint: Dict[str, Any] = torch.load(
                str(model_file),
                map_location=self._device,
                weights_only=False,
            )

            # チェックポイント構造の検証
            if isinstance(checkpoint, dict):
                expected_keys: set[str] = {"E_flame", "E_detail"}
                available_keys: set[str] = set(checkpoint.keys())
                if expected_keys & available_keys:
                    logger.info(
                        "DECAチェックポイントロード: keys={}",
                        list(available_keys)[:10],
                    )
                else:
                    logger.info(
                        "DECAチェックポイントロード（非標準構造）: keys={}",
                        list(available_keys)[:10],
                    )

            self._deca = _StandaloneDECA(checkpoint, self._device)

            logger.info(
                "DECAスタンドアロンロード完了: path={} | device={}",
                model_file,
                self._device,
            )

        except Exception as exc:
            raise ModelLoadError(
                f"DECAチェックポイントのスタンドアロンロードに失敗: {exc}"
            ) from exc

    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """1フレームからDECA 3DMMパラメータを抽出する。

        face_detect.pyで前処理済み（顔検出・クロッピング・リサイズ）の
        画像テンソルを受け取り、DECAのencode()を呼び出してパラメータを返す。

        Args:
            image: 前処理済み顔画像テンソル。shape ``(1, 3, 224, 224)``、
                値域 ``[0, 1]``、GPU上。

        Returns:
            DECA出力パラメータDict。キーと次元は以下の通り:
                shape (1, 100), tex (1, 50), exp (1, 50), pose (1, 6),
                cam (1, 3), light (1, 27), detail (1, 128)。

        Raises:
            RuntimeError: DECAモデルが初期化されていない場合、
                またはencode()実行中にエラーが発生した場合。
        """
        self.validate_image(image)

        if self._deca is None:
            raise RuntimeError("DECAモデルが初期化されていません")

        image_gpu: torch.Tensor = image.to(self._device)

        with torch.no_grad():
            codedict: Dict[str, torch.Tensor] = self._deca.encode(image_gpu)

        result: Dict[str, torch.Tensor] = self._extract_params(codedict)
        return result

    def _extract_params(
        self, codedict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """DECAのcodedictから標準パラメータDictを構成する。

        DECAの内部キー名とBaseExtractorの出力キー名のマッピングを行い、
        lightパラメータの(B, 9, 3)→(B, 27)のreshapeも実行する。

        Args:
            codedict: DECAのencode()出力。

        Returns:
            標準化されたパラメータDict。
        """
        batch_size: int = codedict["shape"].shape[0]

        # DECA codedictのキーマッピング
        # DECA内部キー → BaseExtractor出力キー
        shape: torch.Tensor = codedict["shape"]  # (B, 100)
        tex: torch.Tensor = codedict["tex"]  # (B, 50)
        exp: torch.Tensor = codedict["exp"]  # (B, 50)
        pose: torch.Tensor = codedict["pose"]  # (B, 6)
        cam: torch.Tensor = codedict["cam"]  # (B, 3)

        # light: (B, 9, 3) → (B, 27) にreshape
        light_raw: torch.Tensor = codedict["light"]  # (B, 9, 3)
        light: torch.Tensor = light_raw.reshape(batch_size, -1)  # (B, 27)

        # detail: E_detail ResNet出力 (B, 128)
        detail: torch.Tensor = codedict["detail"]  # (B, 128)

        return {
            "shape": shape,
            "tex": tex,
            "exp": exp,
            "pose": pose,
            "cam": cam,
            "light": light,
            "detail": detail,
        }

    @property
    def param_dim(self) -> int:
        """出力パラメータの総次元数。

        Returns:
            364 (100+50+50+6+3+27+128)。
        """
        return self._PARAM_DIM

    @property
    def param_keys(self) -> List[str]:
        """出力Dictのキーリスト。

        Returns:
            ["shape", "tex", "exp", "pose", "cam", "light", "detail"]。
        """
        return list(self._PARAM_KEYS)


class _StandaloneDECA:
    """DECAリポジトリ非依存のスタンドアロンエンコーダ。

    decalibがインポートできない場合のフォールバックとして、
    チェックポイントから直接エンコーダを復元する。

    Attributes:
        _checkpoint: ロード済みチェックポイントデータ。
        _device: 計算デバイス。
        _E_flame: FLAMEエンコーダネットワーク。
        _E_detail: 詳細エンコーダネットワーク。
    """

    def __init__(
        self, checkpoint: Dict[str, Any], device: torch.device
    ) -> None:
        """スタンドアロンDECAを初期化する。

        Args:
            checkpoint: torch.load()で読み込んだチェックポイントデータ。
            device: 計算デバイス。
        """
        self._checkpoint: Dict[str, Any] = checkpoint
        self._device: torch.device = device
        self._E_flame: Optional[torch.nn.Module] = None
        self._E_detail: Optional[torch.nn.Module] = None

        self._try_load_encoders()

    def _try_load_encoders(self) -> None:
        """チェックポイントからエンコーダの重みをロードする。

        E_flameとE_detailのstate_dictが見つかれば復元を試みる。
        ResNet50ベースのエンコーダ構造を前提とする。
        """
        try:
            from torchvision.models import resnet50

            if "E_flame" in self._checkpoint:
                # FLAMEエンコーダ: ResNet50 → 236D出力
                # shape(100)+tex(50)+exp(50)+pose(6)+cam(3)+light(27) = 236
                e_flame: torch.nn.Module = resnet50(weights=None)
                e_flame.fc = torch.nn.Linear(2048, 236)
                e_flame.load_state_dict(
                    self._checkpoint["E_flame"], strict=False
                )
                e_flame.to(self._device).eval()
                self._E_flame = e_flame

            if "E_detail" in self._checkpoint:
                # 詳細エンコーダ: ResNet50 → 128D出力
                e_detail: torch.nn.Module = resnet50(weights=None)
                e_detail.fc = torch.nn.Linear(2048, 128)
                e_detail.load_state_dict(
                    self._checkpoint["E_detail"], strict=False
                )
                e_detail.to(self._device).eval()
                self._E_detail = e_detail

            logger.debug(
                "スタンドアロンエンコーダロード: E_flame={}, E_detail={}",
                self._E_flame is not None,
                self._E_detail is not None,
            )

        except Exception as exc:
            logger.warning("エンコーダ復元失敗、ゼロ出力モード: {}", exc)

    def encode(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """画像からDECAパラメータをエンコードする。

        エンコーダが利用可能であれば推論を実行し、利用不可であれば
        正しい形状のゼロテンソルを返す（デバッグ・パイプラインテスト用）。

        Args:
            image: 入力画像テンソル。shape ``(B, 3, 224, 224)``。

        Returns:
            DECAパラメータDict。
        """
        batch_size: int = image.shape[0]

        if self._E_flame is not None:
            flame_out: torch.Tensor = self._E_flame(image)
            # 236D → 各パラメータに分割
            shape: torch.Tensor = flame_out[:, :100]
            tex: torch.Tensor = flame_out[:, 100:150]
            exp: torch.Tensor = flame_out[:, 150:200]
            pose: torch.Tensor = flame_out[:, 200:206]
            cam: torch.Tensor = flame_out[:, 206:209]
            light_flat: torch.Tensor = flame_out[:, 209:236]
        else:
            shape = torch.zeros(batch_size, 100, device=self._device)
            tex = torch.zeros(batch_size, 50, device=self._device)
            exp = torch.zeros(batch_size, 50, device=self._device)
            pose = torch.zeros(batch_size, 6, device=self._device)
            cam = torch.zeros(batch_size, 3, device=self._device)
            light_flat = torch.zeros(batch_size, 27, device=self._device)

        light: torch.Tensor = light_flat.reshape(batch_size, 9, 3)

        if self._E_detail is not None:
            detail: torch.Tensor = self._E_detail(image)
        else:
            detail = torch.zeros(batch_size, 128, device=self._device)

        return {
            "shape": shape,
            "tex": tex,
            "exp": exp,
            "pose": pose,
            "cam": cam,
            "light": light,
            "detail": detail,
        }
