"""線形空間特徴量の補間ユーティリティモジュール。

DECA / BFM の PCA 係数（expression, identity）やユークリッド空間の特徴量
（centroid, face_size 等）に対するギャップ補間ロジックを提供する。

本モジュールは**回転特徴量を扱わない**。回転補間は ``rotation_interp.py`` 側で
SLERP ベースの実装を提供する。

サポート手法:
    - ``linear``: 線形補間（旧版互換、デフォルト）
    - ``pchip``: Piecewise Cubic Hermite (Fritsch-Carlson)、``scipy`` が
        インストールされている場合のみ有効。未インストール時は linear にフォールバック

設計根拠の詳細は ``docs/design/interpolation.md`` を参照。

Example:
    ギャップを含む表情係数シーケンスの補間::

        import numpy as np
        from flare.utils.interp import interp_linear, find_gap_runs, find_valid_runs

        values = np.random.randn(30, 50).astype(np.float32)
        mask = np.ones(30, dtype=bool)
        mask[10:13] = False  # 3 フレーム欠損
        filled, new_mask = interp_linear(values, mask, max_gap=12, order="linear")
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator  # type: ignore[import-not-found]

    _HAS_SCIPY: bool = True
except ImportError:
    PchipInterpolator = None  # type: ignore[assignment, misc]
    _HAS_SCIPY = False


def find_valid_runs(mask: np.ndarray) -> list[tuple[int, int]]:
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


def find_gap_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """欠損フレームの連続区間を列挙する。

    Args:
        mask: 有効フレームを ``True``、欠損フレームを ``False`` とするブールマスク。
            形状 ``(T,)``。

    Returns:
        ``(start, end)`` のタプルのリスト。``end`` は inclusive の末尾インデックス。
    """
    inverted = ~mask
    return find_valid_runs(inverted)


def _interp_linear_impl(
    values: np.ndarray,
    mask: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """線形補間の内部実装。

    Args:
        values: 特徴量配列。形状 ``(T, D)`` または ``(T,)``。
        mask: 有効マスク。形状 ``(T,)``。
        max_gap: 補間を許容する最大ギャップフレーム数。

    Returns:
        2 要素のタプル: 補間済み値と更新後マスク。
    """
    if values.ndim == 1:
        values_2d = values[:, None]
        was_1d = True
    else:
        values_2d = values
        was_1d = False

    filled = values_2d.astype(np.float32, copy=True)
    new_mask = mask.copy()

    if new_mask.sum() < 2:
        return (filled[:, 0] if was_1d else filled), new_mask

    gap_runs = find_gap_runs(new_mask)
    for gap_start, gap_end in gap_runs:
        gap_length = gap_end - gap_start + 1
        if gap_start == 0 or gap_end == len(new_mask) - 1:
            continue
        if gap_length > max_gap:
            continue

        left_idx = gap_start - 1
        right_idx = gap_end + 1
        y_left = filled[left_idx]
        y_right = filled[right_idx]

        for k in range(gap_length):
            t = (k + 1) / (gap_length + 1)
            filled[gap_start + k] = (1.0 - t) * y_left + t * y_right
            new_mask[gap_start + k] = True

    out: np.ndarray = filled[:, 0] if was_1d else filled
    return out, new_mask


def _interp_pchip_impl(
    values: np.ndarray,
    mask: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """PCHIP 補間の内部実装。

    scipy の ``PchipInterpolator`` を用いた単調性保存 3 次 Hermite 補間。
    各ギャップごとに、周辺の有効点（最大片側 4 点）を使って PCHIP を構築し、
    ギャップ内の位置で評価する。

    Args:
        values: 特徴量配列。形状 ``(T, D)`` または ``(T,)``。
        mask: 有効マスク。形状 ``(T,)``。
        max_gap: 補間を許容する最大ギャップフレーム数。

    Returns:
        2 要素のタプル: 補間済み値と更新後マスク。
    """
    if not _HAS_SCIPY:
        return _interp_linear_impl(values, mask, max_gap)

    if values.ndim == 1:
        values_2d = values[:, None]
        was_1d = True
    else:
        values_2d = values
        was_1d = False

    filled = values_2d.astype(np.float32, copy=True)
    new_mask = mask.copy()

    if new_mask.sum() < 2:
        return (filled[:, 0] if was_1d else filled), new_mask

    gap_runs = find_gap_runs(new_mask)
    window_neighbors = 4

    for gap_start, gap_end in gap_runs:
        gap_length = gap_end - gap_start + 1
        if gap_start == 0 or gap_end == len(new_mask) - 1:
            continue
        if gap_length > max_gap:
            continue

        valid_idx = np.where(new_mask)[0]
        left_valid = valid_idx[valid_idx < gap_start]
        right_valid = valid_idx[valid_idx > gap_end]

        left_take = left_valid[-window_neighbors:]
        right_take = right_valid[:window_neighbors]
        support_idx = np.concatenate([left_take, right_take])

        if len(support_idx) < 2:
            continue
        if len(support_idx) < 4:
            # PCHIP requires ≥2 points; with 2–3 it degenerates to linear.
            for k in range(gap_length):
                t = (k + 1) / (gap_length + 1)
                left_val = filled[gap_start - 1]
                right_val = filled[gap_end + 1]
                filled[gap_start + k] = (1.0 - t) * left_val + t * right_val
                new_mask[gap_start + k] = True
            continue

        x_support = support_idx.astype(np.float64)
        y_support = filled[support_idx].astype(np.float64)
        x_query = np.arange(gap_start, gap_end + 1, dtype=np.float64)

        interpolator = PchipInterpolator(x_support, y_support, axis=0, extrapolate=False)
        y_query = interpolator(x_query)
        # PchipInterpolator with extrapolate=False returns NaN outside support.
        # Fallback to linear for any NaN positions.
        nan_rows = np.any(np.isnan(y_query), axis=1) if y_query.ndim > 1 else np.isnan(y_query)
        if np.any(nan_rows):
            left_val = filled[gap_start - 1]
            right_val = filled[gap_end + 1]
            for k in range(gap_length):
                if nan_rows[k]:
                    t = (k + 1) / (gap_length + 1)
                    y_query[k] = (1.0 - t) * left_val + t * right_val

        filled[gap_start : gap_end + 1] = y_query.astype(np.float32)
        for k in range(gap_length):
            new_mask[gap_start + k] = True

    out: np.ndarray = filled[:, 0] if was_1d else filled
    return out, new_mask


def interp_linear(
    values: np.ndarray,
    mask: np.ndarray,
    max_gap: int,
    order: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """線形空間特徴量のギャップ補間ディスパッチャ。

    以下の方針で補間する:

    1. 長さ ``max_gap`` 以下のギャップ: 指定手法で補間、マスクを ``True`` に更新
    2. 長さが ``max_gap`` を超えるギャップ: 補間せず、マスクは ``False`` のまま
    3. シーケンス先頭・末尾のギャップ: 片側情報しかないため補間不可、
       マスクは ``False`` のまま

    Args:
        values: 特徴量配列。形状 ``(T, D)`` または ``(T,)``。
        mask: 有効マスク。形状 ``(T,)``、``True`` が有効。
        max_gap: 補間を許容する最大ギャップフレーム数。
        order: 補間手法。``"linear"`` または ``"pchip"``。scipy が無い場合
            ``"pchip"`` は自動的に ``"linear"`` へフォールバックする。

    Returns:
        2 要素のタプル:
            - 補間済み値配列（入力と同じ形状）
            - 更新後のマスク ``(T,)``

    Raises:
        ValueError: ``order`` が未知の場合、または ``values`` と ``mask`` の
            長さが一致しない場合。
    """
    if values.shape[0] != mask.shape[0]:
        raise ValueError(
            f"values length {values.shape[0]} does not match mask length {mask.shape[0]}"
        )

    if order == "linear":
        return _interp_linear_impl(values, mask, max_gap)
    if order == "pchip":
        return _interp_pchip_impl(values, mask, max_gap)
    raise ValueError(
        f"Unknown interpolation order: {order!r}. Supported: 'linear', 'pchip'."
    )


def split_on_long_gaps(
    mask: np.ndarray,
    min_length: int = 1,
) -> list[tuple[int, int]]:
    """マスクの ``True`` 連続領域をシーケンスとして列挙する。

    補間後のマスクに対して呼び出し、残存する ``False`` 位置で分割する。

    Args:
        mask: 有効マスク。形状 ``(T,)``。
        min_length: 採用する最小シーケンス長。これ未満の区間は破棄される。

    Returns:
        ``(start, end)`` のタプルのリスト。``end`` は inclusive。
        ``min_length`` 以上の区間のみを含む。
    """
    runs = find_valid_runs(mask)
    return [(s, e) for s, e in runs if (e - s + 1) >= min_length]


def compute_stats(
    values: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """特徴量の平均・標準偏差を算出する。

    有効マスクが与えられた場合は有効フレームのみを用いて統計量を計算する。
    標準偏差が極小値（数値誤差による 0 割り）にならないよう下限でクリップする。

    Args:
        values: 特徴量配列。形状 ``(T, D)`` または ``(T,)``。
        mask: 有効フレームのマスク。``None`` の場合は全フレームを使用。

    Returns:
        2 要素のタプル: ``(mean, std)``。形状は ``(D,)`` または ``()``。
        dtype は ``float32``。
    """
    if mask is not None:
        selected = values[mask]
    else:
        selected = values

    if selected.shape[0] == 0:
        shape = values.shape[1:] if values.ndim > 1 else ()
        zeros = np.zeros(shape, dtype=np.float32)
        ones = np.ones(shape, dtype=np.float32)
        return zeros, ones

    mean = selected.mean(axis=0).astype(np.float32)
    std = selected.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    return mean, std


def normalize(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """統計量を用いて特徴量を正規化する。

    Args:
        values: 特徴量配列。
        mean: 平均。
        std: 標準偏差（ゼロ除算を避けるため事前クリップ済みを想定）。

    Returns:
        正規化済み特徴量（``(values - mean) / std``、dtype は ``float32``）。
    """
    out: np.ndarray = ((values - mean) / std).astype(np.float32)
    return out
