"""FLAME パラメトリックモデルによるメッシュ可視化。

FLAME (Faces Learned with an Articulated Model and Expressions) の
forward pass を純 PyTorch で実装し、3DMM パラメータからメッシュ頂点を生成、
簡易正射影でワイヤフレーム / ソリッド描画を行う。

**学習不要** — FLAME の ``generic_model.pkl`` を読み込むだけで即座に可視化可能。
DECA / SMIRK の抽出済みパラメータのサニティチェック用途に最適。

FLAME モデル構造 (generic_model.pkl):
    - ``v_template``:  (5023, 3)       テンプレートメッシュ頂点
    - ``shapedirs``:   (5023, 3, 400)  shape(300) + expression(100) blend shapes
    - ``J_regressor``: sparse (5, 5023) 関節位置リグレッサ
    - ``posedirs``:    (36, 15069)      ポーズ補正 blend shapes
    - ``lbs_weights``: (5023, 5)        LBS ウェイト
    - ``f``:           (9976, 3)        三角形面インデックス
    - ``kintree_table``: (2, 5)         運動学ツリー

入手方法:
    https://flame.is.tue.mpg.de/ で academic license に同意してダウンロード。
    DECA チェックポイントに同梱されていることもある (data/ 以下)。

Example:
    基本的な使用::

        renderer = FLAMEMeshRenderer("./checkpoints/flame/generic_model.pkl")
        image = renderer.render(
            shape=np.zeros(100),
            expression=np.zeros(50),
            global_pose=np.zeros(3),
            jaw_pose=np.zeros(3),
        )
        cv2.imshow("FLAME", image)

    npz ファイルからの可視化::

        data = np.load("movements/data001/comp/deca_comp_00000_04499.npz")
        # デノーマライズ
        angle = data["angle"] * data["angle_std"] + data["angle_mean"]
        exp = data["expression"] * data["expression_std"] + data["expression_mean"]
        image = renderer.render(
            shape=data["shape"],
            expression=exp[0],
            global_pose=angle[0],
            jaw_pose=data["jaw_pose"][0] if "jaw_pose" in data else None,
        )
"""

from __future__ import annotations

import pickle
import struct as _struct
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class _Struct:
    """pickle 内の辞書を attribute access に変換するヘルパー。"""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _to_tensor(
    x: Any, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> torch.Tensor:
    """任意入力を torch.Tensor に変換する。"""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    arr = np.asarray(x)
    return torch.tensor(arr, dtype=dtype, device=device)


def _rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues 回転公式。axis-angle (B, 3) → 回転行列 (B, 3, 3)。

    Args:
        axis_angle: 軸角表現 ``(B, 3)``。

    Returns:
        回転行列 ``(B, 3, 3)``。
    """
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    safe_theta = torch.clamp(theta, min=1e-8)
    k = axis_angle / safe_theta.squeeze(-1)  # (B, 3)

    K = torch.zeros(*k.shape[:-1], 3, 3, device=k.device, dtype=k.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]

    eye = torch.eye(3, device=k.device, dtype=k.dtype).expand_as(K)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

    near_zero = (theta.squeeze(-1).squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)
    return torch.where(near_zero, eye, R)


class FLAMEMeshRenderer:
    """FLAME パラメトリックモデルのメッシュ可視化器。

    学習不要。``generic_model.pkl`` を読み込むだけで、3DMM パラメータから
    メッシュを生成し、簡易正射影でレンダリングする。

    Attributes:
        _v_template: テンプレートメッシュ頂点 ``(5023, 3)``。
        _shapedirs: shape + expression blend shapes ``(5023, 3, n_total)``。
        _posedirs: ポーズ補正 ``(N_pose, 5023*3)``。
        _J_regressor: 関節リグレッサ ``(5, 5023)``。
        _lbs_weights: LBS ウェイト ``(5023, 5)``。
        _faces: 三角形面 ``(F, 3)``。
        _kintree_table: 運動学ツリー ``(2, 5)``。
        _n_shape: shape パラメータ次元数 (通常 300)。
        _n_exp: expression パラメータ次元数 (通常 100)。
        _device: 計算デバイス。
    """

    def __init__(
        self,
        flame_model_path: Union[str, Path],
        n_shape: int = 300,
        n_exp: int = 100,
        device: str = "cpu",
    ) -> None:
        """FLAME モデルをロードする。

        Args:
            flame_model_path: ``generic_model.pkl`` のパス。
            n_shape: shape blend shapes の使用次元数。DECA では 100 次元分
                のみ使用するが、モデルには 300 次元分格納されている。
            n_exp: expression blend shapes の使用次元数。DECA は 50。
            device: 計算デバイス。
        """
        self._device = device
        self._n_shape = n_shape
        self._n_exp = n_exp

        path = Path(flame_model_path)
        if not path.exists():
            raise FileNotFoundError(f"FLAME model not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f, encoding="latin1")  # noqa: S301

        if isinstance(model_data, dict):
            model = _Struct(**model_data)
        else:
            model = model_data

        self._v_template = _to_tensor(
            np.asarray(model.v_template), device=device
        )  # (5023, 3)
        self._shapedirs = _to_tensor(
            np.asarray(model.shapedirs), device=device
        )  # (5023, 3, 400)
        self._J_regressor = _to_tensor(
            np.asarray(model.J_regressor.toarray()), device=device
        )  # (5, 5023)
        self._lbs_weights = _to_tensor(
            np.asarray(model.weights), device=device
        )  # (5023, 5)

        posedirs = np.asarray(model.posedirs)
        # posedirs は (5023*3, 36) or (36, 5023*3) の場合がある
        if posedirs.shape[0] == 36:
            posedirs = posedirs.T  # → (5023*3, 36)
        self._posedirs = _to_tensor(posedirs, device=device)

        self._faces = np.asarray(model.f, dtype=np.int32)  # numpy のまま
        self._kintree_table = np.asarray(model.kintree_table, dtype=np.int32)

    def forward(
        self,
        shape: Optional[np.ndarray] = None,
        expression: Optional[np.ndarray] = None,
        global_pose: Optional[np.ndarray] = None,
        jaw_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """FLAME forward pass: パラメータ → メッシュ頂点。

        Args:
            shape: shape 係数 ``(S,)``。None → ゼロ。
            expression: expression 係数 ``(E,)``。None → ゼロ。
            global_pose: グローバル回転 axis-angle ``(3,)``。None → ゼロ。
            jaw_pose: 顎回転 axis-angle ``(3,)``。None → ゼロ。

        Returns:
            メッシュ頂点 ``(5023, 3)`` numpy 配列。
        """
        dev = self._device

        # --- blend shapes ---
        # shape
        n_s = min(self._n_shape, self._shapedirs.shape[2])
        if shape is not None:
            s = _to_tensor(shape[:n_s], device=dev).unsqueeze(0)  # (1, S)
        else:
            s = torch.zeros(1, n_s, device=dev)

        # expression — FLAME shapedirs[:, :, 300:300+n_exp]
        n_e = min(self._n_exp, self._shapedirs.shape[2] - 300)
        if expression is not None:
            e_dim = min(len(expression), n_e)
            e = torch.zeros(1, n_e, device=dev)
            e[0, :e_dim] = _to_tensor(expression[:e_dim], device=dev)
        else:
            e = torch.zeros(1, n_e, device=dev)

        # v_shaped = v_template + shape_offsets + exp_offsets
        shape_offsets = torch.einsum(
            "vci,bi->bvc", self._shapedirs[:, :, :n_s], s
        )  # (1, 5023, 3)
        exp_offsets = torch.einsum(
            "vci,bi->bvc", self._shapedirs[:, :, 300 : 300 + n_e], e
        )  # (1, 5023, 3)
        v_shaped = self._v_template.unsqueeze(0) + shape_offsets + exp_offsets

        # --- 関節位置 ---
        J = torch.einsum("jv,bvc->bjc", self._J_regressor, v_shaped)  # (1, 5, 3)

        # --- ポーズ (5 joints: neck, head, jaw, left_eye, right_eye) ---
        # global rotation
        if global_pose is not None:
            gp = _to_tensor(global_pose, device=dev).reshape(1, 3)
        else:
            gp = torch.zeros(1, 3, device=dev)

        # jaw rotation
        if jaw_pose is not None:
            jp = _to_tensor(jaw_pose, device=dev).reshape(1, 3)
        else:
            jp = torch.zeros(1, 3, device=dev)

        # 残り 3 joints (neck, left_eye, right_eye) は静止
        rest = torch.zeros(1, 9, device=dev)

        # 全ポーズ: [global(3), neck(3), jaw(3), left_eye(3), right_eye(3)]
        full_pose = torch.cat([gp, rest[:, :3], jp, rest[:, 3:9]], dim=1)  # (1, 15)
        rot_mats = _rodrigues(full_pose.reshape(-1, 3))  # (5, 3, 3)
        rot_mats = rot_mats.unsqueeze(0)  # (1, 5, 3, 3)

        # --- ポーズ補正 blend shapes ---
        ident = torch.eye(3, device=dev, dtype=torch.float32)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(1, -1)  # (1, 36)
        n_posedirs = min(pose_feature.shape[1], self._posedirs.shape[1])
        pose_offsets = (
            self._posedirs[:, :n_posedirs] @ pose_feature[0, :n_posedirs]
        )  # (5023*3,)
        pose_offsets = pose_offsets.reshape(1, -1, 3)  # (1, 5023, 3)
        v_posed = v_shaped + pose_offsets

        # --- LBS (Linear Blend Skinning) ---
        n_joints = rot_mats.shape[1]
        # 同次変換行列
        transforms = torch.zeros(1, n_joints, 4, 4, device=dev)
        transforms[:, :, :3, :3] = rot_mats[:, :, :3, :3]
        transforms[:, :, :3, 3] = J[:, :n_joints]

        # 親ノードを遡って連鎖変換
        # kintree: joint 0 = root, joint i の parent = kintree[0, i]
        for i in range(1, n_joints):
            parent = int(self._kintree_table[0, i])
            transforms[:, i] = transforms[:, parent] @ transforms[:, i]

        # 関節位置のオフセットを差し引く（rest pose 基準化）
        rest_J = torch.zeros(1, n_joints, 4, 1, device=dev)
        rest_J[:, :, :3, 0] = J[:, :n_joints]
        rest_J[:, :, 3, 0] = 1.0
        posed_J = transforms @ rest_J
        transforms[:, :, :3, 3] = transforms[:, :, :3, 3] - posed_J[:, :, :3, 0]

        # ウェイテッド変換
        W = self._lbs_weights[:, :n_joints]  # (5023, 5)
        T = torch.einsum("vj,bjmn->bvmn", W, transforms)  # (1, 5023, 4, 4)

        v_homo = torch.cat(
            [v_posed, torch.ones(*v_posed.shape[:-1], 1, device=dev)], dim=-1
        )  # (1, 5023, 4)
        v_final = torch.einsum("bvmn,bvn->bvm", T, v_homo)[:, :, :3]  # (1, 5023, 3)

        return v_final[0].detach().cpu().numpy()

    def render(
        self,
        shape: Optional[np.ndarray] = None,
        expression: Optional[np.ndarray] = None,
        global_pose: Optional[np.ndarray] = None,
        jaw_pose: Optional[np.ndarray] = None,
        image_size: int = 512,
        draw_wireframe: bool = True,
        bg_color: tuple[int, int, int] = (32, 32, 32),
        mesh_color: tuple[int, int, int] = (200, 220, 255),
        wire_color: tuple[int, int, int] = (100, 140, 200),
    ) -> np.ndarray:
        """3DMM パラメータからメッシュ画像をレンダリングする。

        正射影 + OpenCV 描画による軽量レンダリング。
        外部レンダラ (pytorch3d 等) 不要。

        Args:
            shape: shape 係数 ``(S,)``。
            expression: expression 係数 ``(E,)``。
            global_pose: グローバル回転 ``(3,)``。
            jaw_pose: 顎回転 ``(3,)``。
            image_size: 出力画像サイズ (正方形)。
            draw_wireframe: True ならワイヤフレームも描画。
            bg_color: 背景色 (B, G, R)。
            mesh_color: ソリッド面色 (B, G, R)。
            wire_color: ワイヤフレーム色 (B, G, R)。

        Returns:
            BGR 画像 ``(H, W, 3)`` uint8。
        """
        vertices = self.forward(shape, expression, global_pose, jaw_pose)
        return self._render_mesh(
            vertices,
            image_size=image_size,
            draw_wireframe=draw_wireframe,
            bg_color=bg_color,
            mesh_color=mesh_color,
            wire_color=wire_color,
        )

    def _render_mesh(
        self,
        vertices: np.ndarray,
        image_size: int = 512,
        draw_wireframe: bool = True,
        bg_color: tuple[int, int, int] = (32, 32, 32),
        mesh_color: tuple[int, int, int] = (200, 220, 255),
        wire_color: tuple[int, int, int] = (100, 140, 200),
    ) -> np.ndarray:
        """メッシュ頂点と面から画像を描画する。

        正射影 + Z ソートによる面の前後判定 + OpenCV fillPoly/polylines。

        Args:
            vertices: メッシュ頂点 ``(V, 3)``。
            image_size: 出力画像サイズ。
            draw_wireframe: ワイヤフレーム描画の有無。
            bg_color: 背景色。
            mesh_color: 面色。
            wire_color: ワイヤフレーム色。

        Returns:
            BGR 画像。
        """
        img = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)

        # --- 正射影 ---
        verts = vertices.copy()
        # Y 軸反転（画像座標系）
        verts[:, 1] = -verts[:, 1]

        # スケーリング: メッシュの範囲を画像サイズに合わせる
        vmin = verts[:, :2].min(axis=0)
        vmax = verts[:, :2].max(axis=0)
        center = (vmin + vmax) / 2.0
        extent = (vmax - vmin).max()
        margin = 0.1
        scale = image_size * (1.0 - 2 * margin) / max(extent, 1e-8)
        offset = np.array([image_size / 2.0, image_size / 2.0])

        proj = (verts[:, :2] - center) * scale + offset  # (V, 2)
        depth = verts[:, 2]

        # --- 面描画 (Z ソート: 奥→手前) ---
        faces = self._faces
        face_depth = depth[faces].mean(axis=1)
        order = np.argsort(-face_depth)  # 遠い面から描画

        for fi in order:
            tri = faces[fi]
            pts = proj[tri].astype(np.int32).reshape((-1, 1, 2))

            # 簡易シェーディング: 面の法線で明るさを変える
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            n_len = np.linalg.norm(normal)
            if n_len > 1e-8:
                normal /= n_len
            light_dir = np.array([0.0, 0.0, 1.0])
            shade = max(0.3, abs(float(np.dot(normal, light_dir))))

            color = tuple(int(c * shade) for c in mesh_color)
            cv2.fillPoly(img, [pts], color)

        if draw_wireframe:
            for fi in order:
                tri = faces[fi]
                pts = proj[tri].astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=wire_color, thickness=1)

        return img

    @property
    def faces(self) -> np.ndarray:
        """三角形面インデックス ``(F, 3)``。"""
        return self._faces.copy()

    @property
    def num_vertices(self) -> int:
        """テンプレートメッシュの頂点数 (通常 5023)。"""
        return int(self._v_template.shape[0])
