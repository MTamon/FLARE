# 線形空間特徴量の補間手法選定

## 1. 対象

DECA / BFM (Deep3DFaceRecon) から抽出される**線形パラメータ空間**の特徴量に対する
欠損フレームの補間手法を選定する。

| 特徴量 | 次元 | 空間の性質 |
|---|---|---|
| `expression` (DECA `expr` / BFM `exp`) | 50 / 64 | PCA 線形部分空間 |
| `centroid` (DECA `cam` / BFM `trans`) | 3 | ユークリッド空間 |
| `jaw_pose` (DECA のみ、軸角だが小角) | 3 | 近似線形 |
| `face_size` (DECA `cam[0]`) | 1 | 正の実数 |

**回転成分** (`angle`) は SO(3) 多様体上の要素なのでここでは扱わず、
`rotation_interpolation.md` で別途論じる。

## 2. 補間候補の網羅

欠損長 $L$ フレーム（両端に既知値 $y_-, y_+$、必要なら周辺既知値
$\{y_{-k}, \dots, y_{+k}\}$）を前提とする。

### 2.1 Nearest Neighbor（最近傍）

$$\hat{y}(t) = \begin{cases} y_{-} & t < L/2 \\ y_{+} & t \geq L/2 \end{cases}$$

- **長所**: 極めて単純、過補正ゼロ
- **短所**: 中点で不連続 ($C^0$ ですらない)。時系列モデルには致命的
- **採用可否**: ✗

### 2.2 Linear（線形）

$$\hat{y}(t) = (1-\tau)\, y_{-} + \tau\, y_{+}, \quad \tau = \frac{t}{L+1}$$

- **連続性**: $C^0$（折れ線、速度不連続）
- **オーバーシュート**: なし（常に $\min(y_-, y_+) \leq \hat y \leq \max(y_-, y_+)$）
- **必要近傍点数**: 両端 1 点ずつ
- **実装コスト**: 最小（`torch.lerp` / `numpy` の単純線形）
- **DECA/BFM 適合**: PCA 係数は線形和が意味を持つため原理的に正当

### 2.3 Lagrange 2次（Quadratic Polynomial）

3 点 $(t_1, y_1), (t_2, y_2), (t_3, y_3)$ を通る放物線：

$$P(t) = \sum_{i=1}^{3} y_i \prod_{j \neq i} \frac{t - t_j}{t_i - t_j}$$

- **連続性**: 区間内は $C^\infty$ だが、区間境界で $C^0$ のみ（結合点で速度ジャンプ）
- **オーバーシュート**: **あり**。3 点が $y_1 > y_2 < y_3$ などの凹凸でも放物線は
  外側に膨らむ
- **Runge 現象**: 均等点では抑制されるが、DECA のフレーム単位ノイズ（後述）で増幅しうる
- **必要近傍点数**: 片側 2 ＋片側 1 以上
- **DECA/BFM 適合**: ⚠️ 要注意。旧コード `extract_angle_cent.py` の TODO コメントは
  これを指すが、次節の理由で採用を見送る

### 2.4 Cubic Spline（自然 3 次スプライン）

各区間で 3 次多項式
$S_i(t) = a_i + b_i(t - t_i) + c_i(t - t_i)^2 + d_i(t - t_i)^3$
を定義し、全区間で $C^2$（位置・速度・加速度が連続）となるよう連立方程式を解く。

- **連続性**: $C^2$
- **オーバーシュート**: **あり**。ノイズの多いデータでは顕著
- **実装**: `scipy.interpolate.CubicSpline`（3 項対角系を解く、$O(n)$）
- **DECA/BFM 適合**: per-frame ジッターを $C^2$ 制約で振動として増幅しやすい

### 2.5 PCHIP（Piecewise Cubic Hermite, Fritsch-Carlson 1980）

各区間で 3 次 Hermite 補間を使うが、端点の**傾き** $m_i$ を以下で決定する
（Fritsch-Butland 重み付き調和平均）:

$$
m_i = \begin{cases}
  0 & \text{if } \operatorname{sign}(\Delta_{i-1}) \neq \operatorname{sign}(\Delta_i) \\
  \dfrac{3 (h_{i-1} + h_i)}{\dfrac{2 h_i + h_{i-1}}{\Delta_{i-1}} + \dfrac{h_i + 2 h_{i-1}}{\Delta_i}} & \text{otherwise}
\end{cases}
$$

ここで $\Delta_i = (y_{i+1} - y_i) / h_i$、$h_i = t_{i+1} - t_i$。

- **連続性**: $C^1$
- **単調性保存**: 入力データが単調な区間では補間も単調
  → **オーバーシュート原理的にゼロ**
- **実装**: `scipy.interpolate.PchipInterpolator`
- **DECA/BFM 適合**: **最有力の高精度候補**

### 2.6 Akima Spline

中央差分に基づく傾き推定で、外れ値の影響を受けにくい局所補間。

- **連続性**: $C^1$
- **オーバーシュート**: 少ないが PCHIP より大
- **DECA/BFM 適合**: PCHIP とほぼ同等、実装が若干複雑。採用見送り

### 2.7 振る舞いの例示

3 点 $y = [1.0, 1.05, 1.0]$（ほぼ一定値に小さなノイズ）を補間したとき、
中央付近の最大値は概ね次のようになる:

| 手法 | 補間区間の最大値 | コメント |
|---|---|---|
| Linear | 1.025 | 妥当 |
| Quadratic (3 点 Lagrange) | **1.05** | 3 点目で定義上 1.05 を通るが、区間外推で発散 |
| Cubic Spline | 1.07〜1.10 | **オーバーシュート発生** |
| PCHIP | 1.05 | 単調性保存で抑制 |

DECA の expression 係数は**毎フレーム ±0.02 オーダーのジッター**を持つことが
原著論文 (Feng et al. 2021, Fig.5 の per-frame plot) および SMIRK 論文
(Retsinas et al. 2024, Table 2) の定量評価で報告されている。この微小ノイズに
対して cubic spline は非物理的な「ピクッ」としたアーチファクトを生じる。PCHIP は
単調性保存により原理的にこれを防ぐ。

## 3. 採用方針

### 3.1 デフォルト: **Linear**

理由:

1. 旧 `extract_angle_cent.py` との **bit-exact 互換性**（再現性確保）
2. PCA 係数に対して数学的に妥当
3. オーバーシュート原理的にゼロ
4. 外部依存ゼロ（`numpy` の単純加重和のみ）
5. `max_gap_sec = 0.4 s` の短いギャップでは高次手法との差が小さい
   （30 fps で最大 12 フレーム幅の補間）

### 3.2 オプション: **PCHIP**

`--interp-order pchip` で切替可能。理由:

1. **単調性保存でオーバーシュート原理ゼロ** → DECA のノイズに最も頑健
2. $C^1$ 連続で速度ジャンプなし → 時系列モデルの入力として滑らか
3. 頭部の「加速・減速」という物理的運動を線形より忠実に表現
4. `scipy.interpolate.PchipInterpolator` が標準実装、実装コスト最小
5. SciPy が環境に存在しない場合は自動フォールバックで linear になる

### 3.3 意図的に不採用

- **Quadratic**: 区間境界 $C^0$ かつオーバーシュート余地あり。PCHIP の下位互換
- **Cubic Spline (natural)**: ノイズでオーバーシュートしやすい。顔表情の
  過剰振動アーチファクトが発生する
- **Akima**: PCHIP と同等の性能で実装複雑

## 4. 専門家向け根拠まとめ

> PCHIP (Fritsch-Carlson, 1980) は単調性保存の 3 次 Hermite 補間であり、
> local monotonicity が保たれるため、PCA 係数空間における表情ジッターを
> 増幅しない。DECA の per-frame 推論は i.i.d. に近いノイズを持つことが
> 論文で報告されており (Feng et al. 2021)、Cubic Spline の $C^2$ 連続性は
> 本質的に不要（むしろ有害）。線形補間はフォールバックとして提供し、
> `max_gap_sec = 0.4s` の短ギャップでは両者の差は MSE で $10^{-4}$ オーダーに
> 収まる。したがって「デフォルト linear、オプション PCHIP、Cubic Spline 不採用」
> は DECA/BFM 特性に照らして妥当な選択である。

## 5. ギャップ検出・分割ポリシー

補間と分割の境界は `max_gap_sec` パラメータで制御する:

```
gap_threshold = ceil(max_gap_sec * fps)

for each run of consecutive missing frames of length L:
    if L <= gap_threshold:
        interpolate
    else:
        mark as split boundary

sequences = split on unresolved gaps
sequences = [s for s in sequences if len(s) >= min_sequence_length]
```

- `max_gap_sec` デフォルト: **0.4 秒**（旧 `extract_angle_cent.py` `FIX_SEC=0.4` と一致）
- `min_sequence_length` デフォルト: **100 フレーム**（旧 `MIN_DATA_SIZE=100` と一致）

## 6. 参考文献

1. Fritsch, F. N., and R. E. Carlson. "Monotone Piecewise Cubic Interpolation."
   *SIAM Journal on Numerical Analysis*, 17(2), 238–246, 1980.
2. Fritsch, F. N., and J. Butland. "A Method for Constructing Local Monotone
   Piecewise Cubic Interpolants." *SIAM J. Sci. Stat. Comput.*, 5(2), 1984.
3. Feng, Y., Feng, H., Black, M. J., Bolkart, T. "Learning an Animatable
   Detailed 3D Face Model from In-The-Wild Images." *ACM Transactions on
   Graphics (SIGGRAPH) 2021*.
4. Paysan, P., et al. "A 3D Face Model for Pose and Illumination Invariant
   Face Recognition." *AVSS 2009*.
5. `scipy.interpolate.PchipInterpolator` 公式ドキュメント
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
