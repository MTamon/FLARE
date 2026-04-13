"""SO(3) 回転補間ユーティリティモジュール。

頭部グローバル回転（軸角表現 3D）に対する数学的に正しい補間手法を提供する。
旧バージョンの素朴な軸角線形補間では、回転空間が曲がった多様体 SO(3) であるため
測地線上の中点を与えない。本モジュールでは四元数 SLERP (Shoemake 1985) を
実装し、pytorch3d 等の外部依存なしで純 NumPy による SO(3) 補間を可能にする。

実装方針:
    - 入力・出力はすべて NumPy ``float32`` の軸角ベクトル ``(T, 3)`` 形式
    - 内部では単位四元数 ``(T, 4)`` = ``[w, x, y, z]`` レイアウトに変換して演算
    - 二重被覆 S^3 → SO(3) に対応するため、内積符号で short arc を選択
    - 退化ケース（ほぼ同一回転）では線形補間にフォールバック

詳細な数学的根拠は ``docs/design/rotation_interpolation.md`` を参照のこと。

Example:
    ギャップを含む軸角シーケンスの補間::

        import numpy as np
        from flare.utils.rotation_interp import interp_rotation_slerp

        rotations = np.array([
            [0.0, 0.1, 0.0],
            [np.nan, np.nan, np.nan],   # 欠損
            [np.nan, np.nan, np.nan],   # 欠損
            [0.0, 0.4, 0.0],
        ], dtype=np.float32)
        mask = np.array([True, False, False, True])
        filled = interp_rotation_slerp(rotations, mask)
"""

from __future__ import annotations

import numpy as np

_EPS: float = 1e-8
"""float: 数値的な零除算を避けるための微小値。"""

_SLERP_DEGENERATE_THRESHOLD: float = 1e-6
"""float: SLERP の退化判定閾値。これ未満の sin(omega) では線形補間にフォールバック。"""


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """軸角表現を単位四元数に変換する。

    Rodrigues' formula による標準的な変換を行う。入力軸角の大きさが角度
    （ラジアン）、方向が回転軸を表す。出力四元数は ``[w, x, y, z]`` レイアウト
    （Hamilton 規約）。

    Args:
        axis_angle: 軸角ベクトル。任意の末尾形状 ``(..., 3)``。

    Returns:
        単位四元数。形状 ``(..., 4)``。dtype は入力と同じ。
    """
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe_angle = np.maximum(angle, _EPS)
    axis = axis_angle / safe_angle
    half = angle * 0.5
    w = np.cos(half)
    xyz = axis * np.sin(half)
    quat: np.ndarray = np.concatenate([w, xyz], axis=-1)
    return quat


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """単位四元数を軸角表現に変換する。

    入力は正規化されていることを期待するが、内部で再正規化を行う。
    退化ケース（回転角がほぼゼロ）では軸を（0, 0, 0）として返す。

    Args:
        quaternion: 単位四元数。形状 ``(..., 4)``、レイアウト ``[w, x, y, z]``。

    Returns:
        軸角ベクトル。形状 ``(..., 3)``。dtype は入力と同じ。
    """
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    q = quaternion / np.maximum(norm, _EPS)
    w = np.clip(q[..., :1], -1.0, 1.0)
    xyz = q[..., 1:]
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = np.where(sin_half > _EPS, xyz / np.maximum(sin_half, _EPS), np.zeros_like(xyz))
    result: np.ndarray = axis * angle
    return result


def slerp_quaternion(
    q0: np.ndarray,
    q1: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """2 つの単位四元数間で球面線形補間 (SLERP) を行う。

    Shoemake (1985) の SLERP を実装する。二重被覆 S^3 → SO(3) に対応するため、
    内積が負の場合は一方を反転して short arc を選択する。退化ケース（2 つが
    ほぼ同一）ではベクトル線形補間にフォールバックする。

    数式::

        cos(omega) = q0 · q1  (short arc 選択後)
        SLERP(q0, q1; t) = sin((1-t)*omega)/sin(omega) * q0
                         + sin(t*omega)/sin(omega) * q1

    Args:
        q0: 始点四元数。形状 ``(..., 4)``、``[w, x, y, z]``。
        q1: 終点四元数。形状 ``(..., 4)``、``[w, x, y, z]``。
        t: 補間パラメータ 0 ≤ t ≤ 1。形状 ``(..., 1)``。

    Returns:
        補間された単位四元数。形状 ``(..., 4)``。
    """
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1_aligned = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), a_min=None, a_max=1.0 - 1e-7)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    safe_sin = np.where(sin_omega < _SLERP_DEGENERATE_THRESHOLD, 1.0, sin_omega)
    w0 = np.sin((1.0 - t) * omega) / safe_sin
    w1 = np.sin(t * omega) / safe_sin
    spherical = w0 * q0 + w1 * q1_aligned

    linear = (1.0 - t) * q0 + t * q1_aligned
    result: np.ndarray = np.where(
        sin_omega < _SLERP_DEGENERATE_THRESHOLD, linear, spherical
    )
    # 正規化（線形フォールバックで単位長から外れる可能性があるため）
    result = result / np.maximum(
        np.linalg.norm(result, axis=-1, keepdims=True), _EPS
    )
    return result


def slerp_axis_angle(
    aa0: np.ndarray,
    aa1: np.ndarray,
    t: float,
) -> np.ndarray:
    """2 つの軸角表現間で SLERP を行うユーティリティ。

    内部で四元数に変換して ``slerp_quaternion`` を呼び出し、軸角に戻す。

    Args:
        aa0: 始点軸角。形状 ``(..., 3)``。
        aa1: 終点軸角。形状 ``(..., 3)``。
        t: 補間パラメータ 0 ≤ t ≤ 1。スカラ。

    Returns:
        補間された軸角。形状 ``(..., 3)``。
    """
    q0 = axis_angle_to_quaternion(aa0)
    q1 = axis_angle_to_quaternion(aa1)
    t_arr = np.full(q0.shape[:-1] + (1,), float(t), dtype=q0.dtype)
    q_interp = slerp_quaternion(q0, q1, t_arr)
    result: np.ndarray = quaternion_to_axis_angle(q_interp)
    return result


def _find_valid_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """有効フレームの連続区間を列挙する。

    Args:
        mask: 有効フレームを ``True``、欠損フレームを ``False`` とするブールマスク。
            形状 ``(T,)``。

    Returns:
        ``(start, end)`` のタプルのリスト。``end`` は inclusive の末尾インデックス。
        有効区間が存在しない場合は空リスト。
    """
    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i, m in enumerate(mask):
        if m and not in_run:
            run_start = i
            in_run = True
        elif not m and in_run:
            runs.append((run_start, i - 1))
            in_run = False
    if in_run:
        runs.append((run_start, len(mask) - 1))
    return runs


def _find_gap_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """欠損フレームの連続区間を列挙する。

    Args:
        mask: 有効フレームを ``True``、欠損フレームを ``False`` とするブールマスク。

    Returns:
        ``(start, end)`` のタプルのリスト。``end`` は inclusive の末尾インデックス。
    """
    inverted = ~mask
    return _find_valid_runs(inverted)


def interp_rotation_slerp(
    rotations: np.ndarray,
    mask: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """欠損を含む軸角シーケンスに対して SLERP ベースのギャップ補間を行う。

    以下の方針で補間する:

    1. 長さ ``max_gap`` 以下のギャップ:
        両端の既知値を用いて SLERP で補間、マスクは ``True`` に更新
    2. 長さが ``max_gap`` を超えるギャップ:
        補間せず、マスクも ``False`` のまま維持
    3. シーケンス先頭・末尾のギャップ:
        片側の既知値しかないため補間不可、マスクは ``False`` のまま維持
        （後段の分割処理で境界として扱われる）

    Args:
        rotations: 軸角シーケンス。形状 ``(T, 3)``。欠損位置の値は任意（未使用）。
        mask: 有効フレームのマスク。形状 ``(T,)``、``True`` が有効。
        max_gap: 補間を許容するギャップの最大フレーム数。

    Returns:
        2 要素のタプル:
            - 補間済み軸角シーケンス ``(T, 3)``、未補間のフレームは元の値のまま
            - 更新後のマスク ``(T,)``

    Raises:
        ValueError: ``rotations`` の形状が ``(T, 3)`` でない場合。
    """
    if rotations.ndim != 2 or rotations.shape[1] != 3:
        raise ValueError(
            f"rotations must have shape (T, 3), got {rotations.shape}"
        )

    filled = rotations.astype(np.float32, copy=True)
    new_mask = mask.copy()

    if new_mask.sum() < 2:
        return filled, new_mask

    gap_runs = _find_gap_runs(new_mask)
    for gap_start, gap_end in gap_runs:
        gap_length = gap_end - gap_start + 1
        # 先頭・末尾ギャップはスキップ
        if gap_start == 0 or gap_end == len(new_mask) - 1:
            continue
        if gap_length > max_gap:
            continue

        left_idx = gap_start - 1
        right_idx = gap_end + 1
        aa_left = filled[left_idx]
        aa_right = filled[right_idx]

        for k in range(gap_length):
            t = (k + 1) / (gap_length + 1)
            interpolated = slerp_axis_angle(aa_left, aa_right, t)
            filled[gap_start + k] = interpolated
            new_mask[gap_start + k] = True

    return filled, new_mask


def interp_rotation_linear(
    rotations: np.ndarray,
    mask: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """欠損を含む軸角シーケンスに対してベクトル線形補間を行う。

    旧 ``extract_angle_cent.py`` 互換モード。軸角ベクトルを単純に線形補間する。
    回転差が小さい（< 30°）場合は SLERP とほぼ同じ結果になる。

    Args:
        rotations: 軸角シーケンス。形状 ``(T, 3)``。
        mask: 有効フレームのマスク。形状 ``(T,)``。
        max_gap: 補間を許容するギャップの最大フレーム数。

    Returns:
        2 要素のタプル:
            - 補間済み軸角シーケンス ``(T, 3)``
            - 更新後のマスク ``(T,)``

    Raises:
        ValueError: ``rotations`` の形状が ``(T, 3)`` でない場合。
    """
    if rotations.ndim != 2 or rotations.shape[1] != 3:
        raise ValueError(
            f"rotations must have shape (T, 3), got {rotations.shape}"
        )

    filled = rotations.astype(np.float32, copy=True)
    new_mask = mask.copy()

    if new_mask.sum() < 2:
        return filled, new_mask

    gap_runs = _find_gap_runs(new_mask)
    for gap_start, gap_end in gap_runs:
        gap_length = gap_end - gap_start + 1
        if gap_start == 0 or gap_end == len(new_mask) - 1:
            continue
        if gap_length > max_gap:
            continue

        left_idx = gap_start - 1
        right_idx = gap_end + 1
        aa_left = filled[left_idx]
        aa_right = filled[right_idx]

        for k in range(gap_length):
            t = (k + 1) / (gap_length + 1)
            filled[gap_start + k] = (1.0 - t) * aa_left + t * aa_right
            new_mask[gap_start + k] = True

    return filled, new_mask


def interp_rotation(
    rotations: np.ndarray,
    mask: np.ndarray,
    max_gap: int,
    method: str = "slerp",
) -> tuple[np.ndarray, np.ndarray]:
    """回転補間のディスパッチャ関数。

    Args:
        rotations: 軸角シーケンス ``(T, 3)``。
        mask: 有効フレームのマスク ``(T,)``。
        max_gap: 補間を許容するギャップの最大フレーム数。
        method: 補間手法。``"slerp"`` または ``"linear"``。

    Returns:
        2 要素のタプル: 補間済み軸角シーケンスと更新後マスク。

    Raises:
        ValueError: ``method`` が認識されない場合。
    """
    if method == "slerp":
        return interp_rotation_slerp(rotations, mask, max_gap)
    if method == "linear":
        return interp_rotation_linear(rotations, mask, max_gap)
    raise ValueError(
        f"Unknown rotation interpolation method: {method!r}. "
        f"Supported: 'slerp', 'linear'."
    )
