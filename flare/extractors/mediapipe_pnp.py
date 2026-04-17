"""MediaPipe FaceMesh + cv2.solvePnP によるリアルタイム頭部姿勢・位置推定。

このモジュールは DECAExtractor / SMIRKExtractor と組み合わせて使用し、
弱透視カメラ近似（DECA の cam = [scale, tx, ty]）の代わりに
真の world-space カメラ外部パラメータ（K, R, t）を供給する。

出力の (K, R, t) は FlameConverter.convert() の camera_K / camera_R /
camera_t 引数に直接渡せる。

    R : 頭部の回転（姿勢 = Pose）  → カメラ座標系における頭部の向き
    t : 頭部の並進（位置 = Position）  → カメラ座標系における頭部の 3D 座標 (x, y, z)

バックエンド選択（``backend`` パラメータ）:

    "solutions" (既定)
        ``mp.solutions.face_mesh.FaceMesh`` を使用。CPU 推論 (TFLite XNNPACK)。
        追加モデルファイル不要。720p で 60-100 FPS。
        注: Google は 2023-03 に deprecated 宣言しているが現時点では機能する。

    "tasks"
        ``mediapipe.tasks.python.vision.FaceLandmarker`` を使用。
        ``gpu=True`` にすると TFLite GPU delegate（OpenGL ES + EGL 経由）を使用。
        CUDA 12.8 の NVIDIA ドライバに付属する EGL を用いてヘッドレス環境でも動作する。
        環境変数の設定が必要な場合::

            export __EGL_VENDOR_LIBRARY_FILENAMES=\\
                /usr/share/glvnd/egl_vendor.d/10_nvidia.json

        GPU delegate での実測 FPS（RTX 3090）: ~150-200 FPS（CPU比 1.5-1.8 倍）。
        ``model_path`` に ``face_landmarker.task`` ファイルのパスを指定すること。

リアルタイム性能まとめ:
    +---------------+------+----------+-----------+
    | backend       | gpu  | FPS 目安 | 要件      |
    +===============+======+==========+===========+
    | solutions     | N/A  | 80-120   | -         |
    | tasks         | False| 80-120   | .task ファイル |
    | tasks         | True | 150-200  | .task + EGL|
    +---------------+------+----------+-----------+
    FLARE の 25 FPS 目標はどの設定でも満たす。

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

    # CPU 推論（solutions API、デフォルト）
    tracker = MediaPipePnPTracker()

    # Tasks API + GPU delegate
    tracker = MediaPipePnPTracker(
        backend="tasks",
        model_path="./checkpoints/mediapipe/face_landmarker.task",
        gpu=True,
    )

    # YAML 設定ファイルから生成
    import yaml
    cfg = yaml.safe_load(open("configs/realtime_flame.yaml"))
    tracker = MediaPipePnPTracker.from_config(cfg["mediapipe_pnp"])

    # 1 フレーム処理
    result = tracker.track(frame_bgr)   # full-resolution BGR frame
    if result is not None:
        K, R, t = result["K"], result["R"], result["t"]
        frame_dict = flame_converter.convert(
            deca_output,
            camera_K=K, camera_R=R, camera_t=t,
        )

    # リソース解放（context manager も使用可）
    tracker.release()

YAML 設定例 (configs/realtime_flame.yaml 内)::

    mediapipe_pnp:
      backend: tasks                              # "solutions" または "tasks"
      model_path: ./checkpoints/mediapipe/face_landmarker.task  # tasks 時のみ必須
      gpu: true                                   # tasks 時のみ有効
      calib_file: null                            # OpenCV YAML キャリブレーションファイル
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
# 内部バックエンド（ランドマーク検出のみ担当）
# ---------------------------------------------------------------------------

class _SolutionsBackend:
    """mp.solutions.face_mesh を使うバックエンド（CPU、deprecated だが機能する）。"""

    def __init__(self) -> None:
        _mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
        self._face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # 虹彩 10 点は PnP で未使用
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(
        self, frame_rgb: np.ndarray, w: int, h: int
    ) -> np.ndarray | None:
        """ランドマーク 6 点の画素座標 (6, 2) を返す。失敗時は None。"""
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        face_lm = results.multi_face_landmarks[0]
        if len(face_lm.landmark) < max(_MP_LANDMARK_IDX) + 1:
            return None
        pts = np.zeros((len(_MP_LANDMARK_IDX), 2), dtype=np.float64)
        for i, idx in enumerate(_MP_LANDMARK_IDX):
            lm = face_lm.landmark[idx]
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
        return pts

    def close(self) -> None:
        self._face_mesh.close()


class _TasksBackend:
    """mediapipe.tasks FaceLandmarker を使うバックエンド（CPU / GPU 選択可能）。

    Args:
        model_path: ``face_landmarker.task`` バンドルファイルのパス。
        gpu: True のとき TFLite GPU delegate を使用する（OpenGL ES + EGL 経由）。
    """

    def __init__(self, model_path: str, gpu: bool = False) -> None:
        try:
            from mediapipe.tasks import python as _mp_tasks  # type: ignore[import-untyped]
            from mediapipe.tasks.python import vision as _mp_vision  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "mediapipe Tasks API が見つかりません。"
                "`pip install mediapipe>=0.10` を確認してください。"
            ) from e

        delegate = (
            _mp_tasks.BaseOptions.Delegate.GPU
            if gpu
            else _mp_tasks.BaseOptions.Delegate.CPU
        )
        base_options = _mp_tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate,
        )
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = _mp_vision.FaceLandmarker.create_from_options(options)

    def detect(
        self, frame_rgb: np.ndarray, w: int, h: int
    ) -> np.ndarray | None:
        """ランドマーク 6 点の画素座標 (6, 2) を返す。失敗時は None。"""
        mp_image = mp.Image(  # type: ignore[attr-defined]
            image_format=mp.ImageFormat.SRGB,  # type: ignore[attr-defined]
            data=frame_rgb,
        )
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        if len(landmarks) < max(_MP_LANDMARK_IDX) + 1:
            return None
        pts = np.zeros((len(_MP_LANDMARK_IDX), 2), dtype=np.float64)
        for i, idx in enumerate(_MP_LANDMARK_IDX):
            lm = landmarks[idx]
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
        return pts

    def close(self) -> None:
        self._landmarker.close()


# ---------------------------------------------------------------------------
# MediaPipePnPTracker（公開クラス）
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
        backend: ``"solutions"``（既定）または ``"tasks"``。
            ``"solutions"`` は deprecated な ``mp.solutions.face_mesh`` を使用。
            ``"tasks"`` は ``mediapipe.tasks`` FaceLandmarker を使用（GPU 対応）。
        model_path: ``backend="tasks"`` の場合に必須。
            ``face_landmarker.task`` ファイルのパス。
        gpu: ``backend="tasks"`` かつ ``True`` の場合に GPU delegate を使用。
            Linux + NVIDIA EGL ドライバが必要。
    """

    def __init__(
        self,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        backend: str = "solutions",
        model_path: str | None = None,
        gpu: bool = False,
    ) -> None:
        if not _HAS_MEDIAPIPE:
            raise ImportError(
                "mediapipe をインストールしてください: pip install mediapipe"
            )

        self._camera_matrix = camera_matrix
        self._dist_coeffs = (
            dist_coeffs if dist_coeffs is not None
            else np.zeros(4, dtype=np.float64)
        )
        self._backend_name = backend

        if backend == "solutions":
            self._backend: _SolutionsBackend | _TasksBackend = _SolutionsBackend()
        elif backend == "tasks":
            if not model_path:
                raise ValueError(
                    "backend='tasks' には model_path（face_landmarker.task のパス）が必要です。"
                )
            self._backend = _TasksBackend(model_path=model_path, gpu=gpu)
        else:
            raise ValueError(
                f"不明な backend: {backend!r}。'solutions' または 'tasks' を指定してください。"
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
            FlameConverter は内部で単発 ``(3, 3)`` / ``(3,)`` をバッチ次元
            付きに自動 unsqueeze する（``flame_converter.py`` L252-257 参照）
            ので、1 フレーム処理ではそのまま渡せばよい。
        """
        if self._backend is None:
            raise RuntimeError(
                "MediaPipePnPTracker は既に release() されています。再生成してください。"
            )

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

        # ----- ランドマーク検出（バックエンド委譲） -----
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pts_2d = self._backend.detect(frame_rgb, w, h)
        if pts_2d is None:
            return None

        # ----- cv2.solvePnP -----
        # SQPNP (OpenCV ≥ 4.5) は 6 点用途で ITERATIVE より堅牢で初期値不要。
        pnp_flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
        success, rvec, tvec = cv2.solvePnP(
            _FACE_MODEL_3D,
            pts_2d,
            K_np,
            self._dist_coeffs,
            flags=pnp_flag,
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

    @property
    def backend(self) -> str:
        """現在のバックエンド名（"solutions" または "tasks"）。"""
        return self._backend_name

    def release(self) -> None:
        """MediaPipe リソースを解放する。使用後に呼ぶこと。二重呼び出し安全。"""
        if self._backend is not None:
            self._backend.close()
            self._backend = None  # type: ignore[assignment]

    def __enter__(self) -> "MediaPipePnPTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "MediaPipePnPTracker":
        """辞書（YAML 等）から生成する。

        Args:
            config: 以下のキーを持つ辞書::

                backend:    "solutions" | "tasks"   (既定: "solutions")
                model_path: str | null               (tasks 時に必須)
                gpu:        bool                     (tasks 時のみ有効, 既定: false)
                calib_file: str | null               (既定: null = 自動推定)

        Returns:
            初期化済み ``MediaPipePnPTracker``。

        Example::

            # configs/realtime_flame.yaml 内:
            #   mediapipe_pnp:
            #     backend: tasks
            #     model_path: ./checkpoints/mediapipe/face_landmarker.task
            #     gpu: true
            #     calib_file: null

            import yaml
            cfg = yaml.safe_load(open("configs/realtime_flame.yaml"))
            tracker = MediaPipePnPTracker.from_config(cfg["mediapipe_pnp"])
        """
        backend = config.get("backend", "solutions")
        model_path = config.get("model_path") or None
        gpu = bool(config.get("gpu", False))
        calib_file = config.get("calib_file") or None

        camera_matrix: np.ndarray | None = None
        dist_coeffs: np.ndarray | None = None

        if calib_file:
            fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
            node_K = fs.getNode("camera_matrix")
            if node_K.empty():
                raise ValueError(
                    f"キャリブレーションファイルに camera_matrix が見つかりません: {calib_file}"
                )
            camera_matrix = node_K.mat()
            node_d = fs.getNode("dist_coeffs")
            if not node_d.empty():
                dist_coeffs = node_d.mat().flatten()
            fs.release()

        return cls(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            backend=backend,
            model_path=model_path,
            gpu=gpu,
        )

    @classmethod
    def from_calibration(
        cls,
        calib_file: str,
        dist_coeffs: np.ndarray | None = None,
        backend: str = "solutions",
        model_path: str | None = None,
        gpu: bool = False,
    ) -> "MediaPipePnPTracker":
        """OpenCV 形式のキャリブレーションファイルから K を読み込む。

        Args:
            calib_file: ``cv2.FileStorage`` で保存された YAML / XML ファイル。
                ``camera_matrix`` キーに 3×3 行列が格納されていること。
            dist_coeffs: レンズ歪み係数。None の場合はファイルから読む（なければ歪みなし）。
            backend: ``"solutions"`` または ``"tasks"``。
            model_path: ``backend="tasks"`` の場合に必須。
            gpu: ``backend="tasks"`` の場合に有効。

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
        return cls(
            camera_matrix=K,
            dist_coeffs=dist_coeffs,
            backend=backend,
            model_path=model_path,
            gpu=gpu,
        )
