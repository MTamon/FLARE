"""MediaPipe FaceMesh + cv2.solvePnP によるリアルタイム頭部姿勢・位置推定。

このモジュールは DECAExtractor / SMIRKExtractor と組み合わせて使用し、
弱透視カメラ近似（DECA の cam = [scale, tx, ty]）の代わりに
真の world-space カメラ外部パラメータ（K, R, t）を供給する。

出力の (K, R, t) は FlameConverter.convert() の camera_K / camera_R /
camera_t 引数に直接渡せる。

    R : 頭部の回転（姿勢 = Pose）  → カメラ座標系における頭部の向き
    t : 頭部の並進（位置 = Position）  → カメラ座標系における頭部の 3D 座標 (x, y, z)

リアルタイム性能（参考値）:
    - MediaPipe FaceMesh（GPU）: 200+ FPS
    - cv2.solvePnP: <1 ms / frame
    - 合計: 25 FPS 以上の real-time 要件を満たす

座標系規則:
    OpenCV 右手系を採用する。
        x: 画像右方向（被写体から見て自身の左）
        y: 画像下方向
        z: カメラ奥方向（正面を向いた顔から奥へ）

    カメラ行列 K は以下の場合に順に優先して使用される:
        1. コンストラクタに camera_matrix= として渡した実測値
        2. track() に img_wh= として渡した画像サイズから自動推定
           （仮定: 正方形ピクセル、焦点距離 = max(W, H) px、主点 = 中心）

Example::

    tracker = MediaPipePnPTracker()

    # 1 フレーム処理
    result = tracker.track(frame_bgr)   # full-resolution BGR frame
    if result is not None:
        K, R, t = result["K"], result["R"], result["t"]
        frame_dict = flame_converter.convert(
            deca_output,
            camera_K=K, camera_R=R, camera_t=t,
        )

    # リソース解放
    tracker.release()
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

try:
    import mediapipe as mp  # type: ignore[import-untyped]

    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False


# ---------------------------------------------------------------------------
# 6 点顔モデル定数（OpenCV 座標系、単位: mm）
# ---------------------------------------------------------------------------

# MediaPipe FaceMesh 468 点モデルにおける 6 点のランドマーク番号
# 注: MediaPipe は被写体の視点で「左」「右」を定義する
_MP_LANDMARK_IDX: list[int] = [
    4,    # 鼻先 (Nose tip)
    152,  # 顎先 (Chin)
    263,  # 被写体左目の外角 (viewer の右側に映る)
    33,   # 被写体右目の外角 (viewer の左側に映る)
    61,   # 被写体左口角
    291,  # 被写体右口角
]

# 対応する 3D 正準位置 (mm)
# 系: x = 画像右 (+) / 左 (-), y = 画像下 (+) / 上 (-), z = 奥へ向かう (+)
# 被写体左目の外角は画像右に映る → +x
_FACE_MODEL_3D = np.array(
    [
        [  0.0,    0.0,    0.0],  # 鼻先（原点）
        [  0.0,   66.0,   -6.0],  # 顎先
        [ 45.0,  -34.0,  -14.0],  # 被写体左目外角  (+x = 画像右)
        [-45.0,  -34.0,  -14.0],  # 被写体右目外角  (-x = 画像左)
        [ 30.0,   30.0,  -12.0],  # 被写体左口角
        [-30.0,   30.0,  -12.0],  # 被写体右口角
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# MediaPipePnPTracker
# ---------------------------------------------------------------------------


class MediaPipePnPTracker:
    """MediaPipe FaceMesh + cv2.solvePnP によるリアルタイム頭部姿勢・位置推定。

    BaseExtractor を継承しない（入力がクロップ済みテンソルではなくフル解像度
    BGR フレームであるため）。DECA / SMIRK と並列に呼び出し、FlameConverter に
    camera_K / camera_R / camera_t を供給する補助トラッカとして使用する。

    Args:
        camera_matrix: カメラ固有行列 (3, 3)。None の場合はフレームサイズから
            自動推定する（焦点距離 = max(W, H)、主点 = 中心）。
        dist_coeffs: レンズ歪み係数。None の場合は歪みなしとする。
    """

    def __init__(
        self,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        if not _HAS_MEDIAPIPE:
            raise ImportError(
                "mediapipe をインストールしてください: pip install mediapipe"
            )

        self._camera_matrix = camera_matrix
        self._dist_coeffs = (
            dist_coeffs
            if dist_coeffs is not None
            else np.zeros(4, dtype=np.float64)
        )

        _mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
        self._face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def track(
        self,
        frame_bgr: np.ndarray,
        img_wh: tuple[int, int] | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor] | None:
        """1 フレームから頭部姿勢・位置を推定する。

        Args:
            frame_bgr: フル解像度の BGR 画像。形状は ``(H, W, 3)``、dtype は
                ``uint8``。クロップ前のカメラ生フレームを渡すこと。
            img_wh: ``(width, height)``。None の場合は frame_bgr の形状から
                自動取得する。camera_matrix が None の場合のみ使用される。
            device: 出力テンソルのデバイス。

        Returns:
            成功した場合は以下のキーを持つ辞書:

            - ``"K"``: (3, 3) カメラ固有行列 (``torch.float32``)
            - ``"R"``: (3, 3) 回転行列（頭部の向き = 姿勢）(``torch.float32``)
            - ``"t"``: (3,) 並進ベクトル（頭部の 3D 位置）(``torch.float32``)
            - ``"rvec"``: (3,) 回転ベクトル（axis-angle） (``torch.float32``)
            - ``"success"``: True

            失敗した場合は None。

        Note:
            返り値の ``K``, ``R``, ``t`` は FlameConverter.convert() の
            ``camera_K``, ``camera_R``, ``camera_t`` 引数にそのまま渡せる。
            FlameConverter は (B, 3, 3) を期待するため、必要に応じて
            ``.unsqueeze(0)`` すること。
        """
        h, w = frame_bgr.shape[:2]
        if img_wh is None:
            img_wh = (w, h)

        # ----- カメラ固有行列 -----
        if self._camera_matrix is not None:
            K_np = self._camera_matrix.astype(np.float64)
        else:
            fx = float(max(img_wh))
            cx = img_wh[0] / 2.0
            cy = img_wh[1] / 2.0
            K_np = np.array(
                [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

        # ----- MediaPipe FaceMesh -----
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        n_landmarks = len(face_landmarks.landmark)

        # 6 点の 2D 画像座標を取得
        pts_2d = np.zeros((len(_MP_LANDMARK_IDX), 2), dtype=np.float64)
        for i, idx in enumerate(_MP_LANDMARK_IDX):
            if idx >= n_landmarks:
                return None
            lm = face_landmarks.landmark[idx]
            pts_2d[i, 0] = lm.x * w
            pts_2d[i, 1] = lm.y * h

        # ----- cv2.solvePnP -----
        success, rvec, tvec = cv2.solvePnP(
            _FACE_MODEL_3D,
            pts_2d,
            K_np,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        R_np, _ = cv2.Rodrigues(rvec)

        # ----- numpy → torch -----
        K_t = torch.from_numpy(K_np.astype(np.float32)).to(device)
        R_t = torch.from_numpy(R_np.astype(np.float32)).to(device)
        t_t = torch.from_numpy(tvec.flatten().astype(np.float32)).to(device)
        rvec_t = torch.from_numpy(rvec.flatten().astype(np.float32)).to(device)

        return {
            "K": K_t,
            "R": R_t,
            "t": t_t,
            "rvec": rvec_t,
            "success": True,
        }

    def release(self) -> None:
        """MediaPipe リソースを解放する。使用後に呼ぶこと。"""
        self._face_mesh.close()

    def __enter__(self) -> "MediaPipePnPTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # class method: K をカメラキャリブレーションファイルから読み込む
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(
        cls,
        calib_file: str,
        dist_coeffs: np.ndarray | None = None,
    ) -> "MediaPipePnPTracker":
        """OpenCV 形式のキャリブレーションファイルから K を読み込む。

        Args:
            calib_file: ``cv2.FileStorage`` で保存された YAML / XML ファイル。
                ``camera_matrix`` キーに 3×3 行列が格納されていること。
            dist_coeffs: レンズ歪み係数。None の場合は歪みなしとする。

        Returns:
            初期化済み ``MediaPipePnPTracker``。
        """
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        if K is None or K.shape != (3, 3):
            raise ValueError(
                f"キャリブレーションファイルに camera_matrix (3×3) が見つかりません: {calib_file}"
            )
        if dist_coeffs is None:
            node = fs.getNode("dist_coeffs")
            if not node.empty():
                dist_coeffs = node.mat().flatten()
        fs.release()
        return cls(camera_matrix=K, dist_coeffs=dist_coeffs)
