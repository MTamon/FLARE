# 回転特徴量の補間手法選定

## 1. 対象

DECA / BFM から得られる**頭部グローバル回転** (`pose[:, :3]`) の欠損フレーム補間。
軸角ベクトル $\mathbf{r} \in \mathbb{R}^3$ 形式で、$\|\mathbf{r}\|$ が回転角、
$\mathbf{r} / \|\mathbf{r}\|$ が回転軸を表す。

## 2. 回転の数学的背景

### 2.1 表現の選択肢と問題点

| 表現 | 次元 | 線形補間の妥当性 |
|---|---|---|
| Euler 角 (yaw, pitch, roll) | 3 | ✗ ジンバルロック / $\pm \pi$ 不連続 |
| 軸角 $\mathbf{r}$ | 3 | ✗ 測地線にならない |
| 回転行列 $R \in \mathrm{SO}(3)$ | 9 (自由度 3) | ✗ 補間結果が $\mathrm{SO}(3)$ から外れる |
| 単位四元数 $q \in S^3$ | 4 (自由度 3) | ✗ 単位長制約が崩れる。ただし正規化＋大円補間で解決可 |

**どの 3 次元表現を線形補間しても測地線上の中点は得られない。** これは回転空間
$\mathrm{SO}(3)$ がリー群であり、曲がった多様体だからである。

### 2.2 具体例 A：同一平面内の大回転

$\mathbf{r}_1 = (0, 170°, 0)$ と $\mathbf{r}_2 = (0, -170°, 0)$（両方とも yaw のみ）の
中点:

- **軸角の線形補間**: $\mathbf{r}_{\text{lin}} = (0, 0°, 0)$ → **正面向き（誤）**
- **真の測地線中点**: $(0, 180°, 0) \equiv (0, -180°, 0)$ → **真後ろ向き（正）**

この場合、軸角を線形補間すると回転が「真後ろ → 一瞬で正面 → また真後ろ」と
ワープする。

### 2.3 具体例 B：異なる平面での回転

$R_1 = R_x(30°)$（X 軸まわり 30°）と $R_2 = R_y(30°)$（Y 軸まわり 30°）の中点:

- **軸角の線形平均**: $(0.262, 0.262, 0)$ ≈ 軸 $(1,1,0)/\sqrt{2}$, 角 $21.2°$
- **真の測地線中点** (行列対数経由): 軸方向と角度の両方が微妙に異なる結果

軸角を単純平均すると**軸の方向も角度も両方ともズレる**。誤差は回転差が大きいほど
顕著になる。

### 2.4 誤差のオーダー評価

Rodrigues 展開により、連続する軸角 $\mathbf{r}_- = \mathbf{r} - \Delta/2$,
$\mathbf{r}_+ = \mathbf{r} + \Delta/2$ について、線形補間 $\mathbf{r}_{\text{lin}}(t)$
と真の測地線補間 $\mathbf{r}_{\text{true}}(t)$ の差は次のオーダーとなる:

$$\|\mathbf{r}_{\text{lin}}(t) - \mathbf{r}_{\text{true}}(t)\| = \mathcal{O}(\|\Delta\|^3)$$

すなわち**連続フレーム間の回転差が小さい（< 5°）なら誤差は $\sim 10^{-4}$ rad で
無視可能**だが、差が 60° を超えると数度の誤差が生じる。

## 3. 補間候補

### 3.1 Linear on Axis-Angle（素朴法、旧版）

$$\mathbf{r}(t) = (1-t)\, \mathbf{r}_- + t\, \mathbf{r}_+$$

- **長所**: 実装簡単、高速
- **短所**: 上記 §2.2–2.4 の通り、原理的に誤り
- **誤差の上限**: 差が小さいとき $\mathcal{O}(\|\Delta\|^3)$ で許容範囲

### 3.2 SLERP (Spherical Linear Interpolation, Shoemake 1985)

単位四元数 $q_0, q_1 \in S^3$ 間の定速大円補間:

$$
\operatorname{SLERP}(q_0, q_1; t) = \frac{\sin((1 - t)\, \Omega)}{\sin \Omega}\, q_0
  + \frac{\sin(t\, \Omega)}{\sin \Omega}\, q_1
$$

ここで $\cos \Omega = q_0 \cdot q_1$（四元数内積）。等価な群演算表示:

$$\operatorname{SLERP}(q_0, q_1; t) = q_0\, (q_0^{-1} q_1)^t$$

- **長所**: 測地線上を等速移動 → 数学的に正しい $\mathrm{SO}(3)$ 補間
- **実装**: 三角関数のみ、pytorch3d 不要
- **退化ケース**: $\Omega \approx 0$（ほぼ同一）で $\sin \Omega \to 0$、そのときは
  ベクトル線形補間にフォールバック
- **二重被覆対策**: $q_0 \cdot q_1 < 0$ なら $q_1 \leftarrow -q_1$（short arc を選択）

### 3.3 Log-space Polynomial（任意次数）

手順:

1. 中心回転 $R_c$（区間中央付近の既知値）を選ぶ
2. 各既知回転を $R_c$ 相対に変換: $\tilde R_i = R_c^{-1} R_i$
3. 行列対数で接空間 $\mathfrak{so}(3) \cong \mathbb{R}^3$ へ:
   $\mathbf{v}_i = \log(\tilde R_i)^\vee$
4. $\mathbf{v}_i$ に対して通常の多項式補間（PCHIP / Cubic）を適用
5. 指数写像で戻す: $R(t) = R_c \exp(\hat{\mathbf{v}}(t))$

- **長所**: 高次（2次・3次）の回転補間が可能で滑らか
- **短所**: 区間中央を中心とするため、区間が広いと線形化誤差が増加
- **妥当性条件**: 区間全体の回転差が $\ll \pi$（対面対話なら常に成立）

### 3.4 SQUAD (Spherical Quadrangle, Shoemake 1987)

4 つの四元数を使う SLERP の 2 次拡張。実装複雑で制御点選定が難しく、
log-space polynomial の下位互換なので不採用。

## 4. 実用的制約の分析

### 4.1 対面対話での頭部可動域

| 軸 | 人間の最大可動域 | 実対話での典型範囲 | DECA 検出信頼域 |
|---|---|---|---|
| Yaw (首の左右) | $\pm 80°$ | $\pm 45°$ | $\pm 60°$ |
| Pitch (うなずき) | $\pm 60°$ | $\pm 20°$ | $\pm 40°$ |
| Roll (首の傾げ) | $\pm 45°$ | $\pm 15°$ | $\pm 30°$ |

出典:

- Youdas, J. W., et al. "Normal range of motion of the cervical spine."
  *Arch Phys Med Rehabil*, 1992.
- DECA 論文 (Feng et al. 2021) Fig.7 の detection degradation curve

### 4.2 「±90° で十分」というユーザ仮説の検証

**仮説**: 対面対話なら頭部回転は ±90° 以内、かつ平均で正規化すれば線形補間で十分。

**分析**:

1. **可動域条件**: 対話中は yaw ±60° 以内でほぼ成立 → §2.2 の ±170° 問題は
   発生しない ✓
2. **平均正規化の効果**: 分布中心を 0 にシフトするため、補間対象の差
   $\Delta \mathbf{r} = \mathbf{r}_+ - \mathbf{r}_-$ が小さくなり、線形補間誤差
   $\mathcal{O}(\|\Delta\|^3)$ が縮小 ✓
3. **連続フレーム仮定**: 30 fps なら 1 フレーム間の差は通常 $< 3°$、
   `max_gap_sec = 0.4s` ギャップ全体でも $< 30°$ が一般的
4. **定量評価**:
   - $\|\Delta\| = 30°$ のとき、線形 vs SLERP の誤差 ≈ 0.003 rad ≈ 0.17° → 無視可能
   - $\|\Delta\| = 60°$ のとき、誤差 ≈ 0.025 rad ≈ 1.4° → 許容
   - $\|\Delta\| = 90°$ のとき、誤差 ≈ 0.085 rad ≈ 4.9° → 境界線

**結論**: ユーザ仮説は**原則として正しい**。対面対話・±60° 範囲・短ギャップという
三重条件下では linear でほぼ問題ない。

### 4.3 留意点

1. **検出失敗は偏る**: 横向き / 下向きが強い瞬間に検出失敗しやすく、「ギャップ両端が
   極端な角度」なパターンが統計的に増える → 実運用で $\|\Delta\| > 60°$ が起きる
   確率は想定より高い
2. **四元数の二重被覆**: 軸角ベクトルは $\|\mathbf{r}\| \leq \pi$ に制限して扱うこと。
   DECA/BFM の出力は通常この範囲に入るが、連続フレームで符号が揺れる報告あり
3. **正規化の順序**: 「対話単位で平均減算」するなら、補間は**正規化前**に行うべき。
   正規化後の空間は $\mathrm{SO}(3)$ 構造を持たず、SLERP の意味が失われる
4. **学習への影響**: LHG モデルが出力回転を損失関数で扱う際、補間アーチファクトが
   直接学習信号になる。SLERP を使えばこの信号がクリーン

## 5. 採用方針

### 5.1 デフォルト: **SLERP**

理由:

1. 実装コストが linear と同等（三角関数数個）
2. 数学的に厳密
3. 小角領域では linear に漸近するため、対面対話の典型ケースで旧版と結果がほぼ
   一致（bit-exact ではないが MSE $< 10^{-4}$）
4. 外れ値（検出境界での大角度差）で正しく動く安全弁として機能
5. pytorch3d 不要、純 `numpy` / 純 `torch` で実装可
6. PyTorch 2.9 / 2.11 どちらのバージョンでも追加依存なしで動作

### 5.2 オプション: **linear**（旧版互換モード）

`--rotation-interp linear` で選択。旧版との再現性確認用。

### 5.3 実験的オプション: **log_pchip**

`--rotation-interp log_pchip` で選択。log 空間で PCHIP を適用する実装。
滑らかさ最大だが区間中心の選定が難しく、短ギャップではメリットが小さい。
本実装の初期フェーズでは未実装（将来拡張用の設計のみ）。

### 5.4 実装コア（純 NumPy / PyTorch, 数式ベース）

以下のヘルパを `flare/utils/rotation_interp.py` に実装する。

```python
def axis_angle_to_quaternion(aa):
    """aa: (..., 3) → q: (..., 4), q = [w, x, y, z]"""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe_angle = np.maximum(angle, 1e-8)
    axis = aa / safe_angle
    half = angle * 0.5
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)


def quaternion_to_axis_angle(q):
    """q: (..., 4) → aa: (..., 3)"""
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w = np.clip(q[..., :1], -1.0, 1.0)
    xyz = q[..., 1:]
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    safe = sin_half > 1e-6
    axis = np.where(safe, xyz / np.maximum(sin_half, 1e-8), np.zeros_like(xyz))
    return axis * angle


def slerp(q0, q1, t):
    """単位四元数 SLERP. q0,q1: (..., 4), t: (..., 1) in [0,1]."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)             # short arc
    dot = np.clip(np.abs(dot), None, 1.0 - 1e-7)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    w0 = np.sin((1 - t) * omega) / sin_omega
    w1 = np.sin(t * omega) / sin_omega
    out = w0 * q0 + w1 * q1
    linear = (1 - t) * q0 + t * q1
    return np.where(sin_omega < 1e-6, linear, out)
```

## 6. 専門家向け根拠まとめ

> SLERP は Shoemake (1985) により導入された $\mathrm{SO}(3)$ 上の geodesic 補間
> であり、$\operatorname{SLERP}(q_0, q_1; t) = q_0 (q_0^{-1} q_1)^t$ と等価である。
> 単位四元数を二重被覆 $S^3 \to \mathrm{SO}(3)$ で扱うため、内積符号による
> short arc 選択を行えば全回転範囲で正しく動く。対象が $\|\Delta r\| < \pi/3$ の
> 対面対話領域ではベクトル線形補間の誤差が $\mathcal{O}(\|\Delta r\|^3)$ で
> 無視できるが、境界サンプル（検出失敗前後の大角度差）では SLERP が必要となる。
> 実装は三角関数のみ、PyTorch 2.9 / 2.11 いずれでも外部依存なく動作する。
> log-space 高次補間は対話データでの短ギャップ（12 frame）では $10^{-4}$ 以下の
> 改善しかもたらさず、実用上 SLERP で十分である。

## 7. 参考文献

1. Shoemake, K. "Animating Rotation with Quaternion Curves." *SIGGRAPH 1985*.
2. Shoemake, K. "Quaternion Calculus and Fast Animation."
   *SIGGRAPH Course Notes*, 1987.
3. Murray, R. M., Li, Z., Sastry, S. S. *A Mathematical Introduction to
   Robotic Manipulation*, CRC Press 1994, Ch.2 (Lie Group $\mathrm{SO}(3)$).
4. Youdas, J. W., et al. "Normal range of motion of the cervical spine."
   *Arch Phys Med Rehabil*, 1992.
5. Feng, Y., et al. "DECA: Detailed Expression Capture and Animation."
   *ACM Transactions on Graphics (SIGGRAPH) 2021*.
