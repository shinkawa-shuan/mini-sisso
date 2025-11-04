# mini-sisso

[![PyPI version](https://badge.fury.io/py/mini-sisso.svg)](https://pypi.org/project/mini-sisso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/mini-sisso.svg)](https://pypi.org/project/mini-sisso/)

**`mini-sisso` は、シンボリック回帰アルゴリズム SISSOをPythonで実装した、軽量で手軽なライブラリです。scikit-learnエコシステムと完全に互換性があり、データから人間が解釈可能な数式モデルを発見します。**

C++/Fortranベースのオリジナル実装が持つ高度な探索能力を、よりモダンで使いやすい形で提供します。

-   **🚀 手軽な導入**: `pip install` で簡単インストール。CPU版はNumPy/SciPyのみに依存し、軽量です。
-   **🧠 メモリ効率と高速な探索**:
    -   「レシピ化」アーキテクチャにより、特徴拡張時のメモリ消費を劇的に削減。
    -   「レベルワイズSIS」機能（オン/オフ可能）により、無駄な計算を省き、探索を高速化。
-   **🤝 `scikit-learn`完全互換**: `fit`/`predict`インターフェースはもちろん、`GridSearchCV`や`Pipeline`とシームレスに連携。
-   **⚡ オプションのGPUサポート**: `pip install "mini-sisso[gpu]"`でPyTorchを導入すれば、GPUアクセラレーションによるさらなる高速化が可能です。

## 📥 インストール

### CPU版 (デフォルト・推奨)

PyPIから軽量なCPU版をインストールします。NumPy/SciPyのみに依存します。

```bash
pip install mini-sisso
```

### GPU版 (オプション)

PyTorchを利用したGPUアクセラレーションを有効にするには、`[gpu]`オプションを付けてインストールします。

```bash
pip install "mini-sisso[gpu]"
```

## 🚀 クイックスタート

わずか数行のコードで、データから数式モデルを学習できます。

```python
import pandas as pd
import numpy as np
from mini_sisso.model import MiniSisso

# 1. データの準備
np.random.seed(42) # 再現性のための乱数シード固定
# 特徴量データを作成 (X)
X_df = pd.DataFrame(np.random.rand(100, 2) *, columns=["feature_A", "feature_B"])
# 真の式 y = 2*sin(feature_A) + feature_B^2 にノイズを加えてターゲットデータを作成 (y)
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1)

# 2. モデルのインスタンス化
# MiniSissoの全ハイパーパラメータを設定できます。
# 使わないものはコメントアウトしたり、デフォルト値のままにします。
model = MiniSisso(
    # --- 探索空間の制御 ---
    n_expansion=2,          # 特徴拡張のレベル (深くするほど複雑な式を発見できるが計算時間増)
    operators=["+", "sin", "pow2"], # 特徴拡張に使う演算子リスト
    
    # --- モデルの複雑さの制御 ---
    n_term=2,               # 発見する数式の最大項数 (exhaustiveメソッド用)
    
    # --- 探索戦略の選択 ---
    so_method="exhaustive", # モデル探索戦略 ('exhaustive' or 'lasso')
    # alpha=0.01,           # so_method='lasso' の場合に使う正則化パラメータ
    
    # --- 計算効率の制御 ---
    use_levelwise_sis=True, # 段階的に特徴を枝刈りする高速化機能 (Trueを強く推奨)
    k_per_level=50,         # use_levelwise_sis=True の場合、各レベルで残す有望な特徴の数
    k=10,                   # 最終的なモデル構築の際、各項の候補となる特徴の数
    
    # --- 実行環境の選択 ---
    # device="cuda",          # GPUを使う場合は 'cuda' を指定 (別途PyTorchが必要)
)

# 3. モデルの学習
# scikit-learnと同じく fit(X, y) で学習
model.fit(X_df, y_series)

# 4. 学習結果の確認
# 学習済みの属性 (末尾にアンダースコアが付く) にアクセス
print(f"発見された数式: {model.equation_}")
print(f"訓練RMSE: {model.rmse_:.4f}")
print(f"訓練R2スコア: {model.r2_:.4f}")

# 5. 新しいデータで予測
# scikit-learnと同じく predict(X) で予測
X_test_df = pd.DataFrame(np.array([, ]), columns=["feature_A", "feature_B"])
predictions = model.predict(X_test_df)
print(f"\n新しいデータに対する予測結果: {predictions}")
```

**出力例**:
```
Using NumPy/SciPy backend for CPU execution.
*** Starting Level-wise Recipe Generation (Level-wise SIS: ON, k_per_level=50) ***
Level 1: Generated 5, selected top 5. Total promising: 7. Time: 0.00s
Level 2: Generated 30, selected top 30. Total promising: 37. Time: 0.00s
***************** Starting SISSO Regressor (NumPy/SciPy Backend, Method: exhaustive) *****************

===== Searching for 1-term models =====
SIS selected 10 new features. Pool size: 10
--- Running SO for 1-term models. Total combinations: 10 ---
Best 1-term model: RMSE=0.228209, Eq: +0.980302 * (feature_A + ^2(feature_B)) +0.477770
Time: 0.00 seconds

===== Searching for 2-term models =====
SIS selected 10 new features. Pool size: 20
--- Running SO for 2-term models. Total combinations: 190 ---
Best 2-term model: RMSE=0.092124, Eq: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610
Time: 0.01 seconds

==================================================
SISSO fitting finished. Total time: 0.02s
==================================================

Best Model Found (2 terms):
  RMSE: 0.092124
  R2:   0.998806
  Equation: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610

発見された数式: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610
訓練RMSE: 0.0921
訓練R2スコア: 0.9988

新しいデータに対する予測結果: [2.0016012 5.6796584]
```

## 🛠️ 使い方ガイド

### `use_levelwise_sis`: 特徴生成戦略の切り替え

`mini-sisso`の高速化の鍵である「レベルワイズSIS」機能のオン/オフを切り替えます。

#### `True` (デフォルト)
特徴拡張をレベル（段階）ごとに行い、各レベルの直後にスクリーニング（SIS）を実行します。有望な特徴だけを次のレベルの生成に使用するため、計算時間とメモリ使用量を大幅に削減できます。**通常はこちらの使用を推奨します。**

```python
# k_per_levelで各レベルで残す特徴数を制御できる
model_fast = MiniSisso(use_levelwise_sis=True, k_per_level=100)
```

#### `False`
特徴拡張の全レベルで考えられるすべての特徴（レシピ）を一度に生成してから、最終的なSIS/SOステップを実行します。
-   **長所**: 探索空間が広がり、思わぬ特徴の組み合わせが見つかる可能性があります。
-   **短所**: **メモリ使用量と計算時間が爆発的に増加します。** `n_expansion`が大きい場合や、ベース特徴量の数が多い場合は、メモリ不足でプログラムがクラッシュするリスクがあります。

```python
# n_expansionは小さく設定することを推奨
model_full_search = MiniSisso(use_levelwise_sis=False, n_expansion=2)
```

### `so_method`: モデル探索戦略の選択

#### `exhaustive` (デフォルト)
候補となる特徴のすべての組み合わせをテストする**総当たり探索**。シンプルで解釈しやすいモデルが見つかりやすいですが、計算時間は組み合わせ的に増加します。`n_term`で使用する項数を指定します。

```python
# 3項までのモデルを総当たりで探索
model_exhaustive = MiniSisso(
    so_method="exhaustive", 
    n_term=3,
    operators=["+", "-", "*", "sqrt"]
)
```

#### `lasso`
**Lasso回帰**を用いて、多数の候補から重要な特徴を高速に選択します。`exhaustive`では現実的でない大規模な探索空間で有効です。`alpha`パラメータで正則化の強さを調整します。

```python
# Lassoで高速に特徴を選択
# alphaが小さいほど、多くの特徴が選択される傾向にある
model_lasso = MiniSisso(
    so_method="lasso",
    alpha=0.01,
    operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "pow2", "pow3"]
)
```
**`alpha`の調整**: `alpha`は試行錯誤が必要です。`LASSO selected 0 features.`と表示されたら`alpha`を小さく、特徴を選びすぎる場合は大きくしてみてください。

### 利用可能な演算子

`operators`引数に文字列のリストとして指定します。

| 演算子   | 説明              |
| :------- | :---------------- |
| `'+'`    | 加算 (a + b)      |
| `'-'`    | 減算 (a - b)      |
| `'*'`    | 乗算 (a * b)      |
| `'/'`    | 除算 (a / b)      |
| `'sin'`  | サイン (sin(a))   |
| `'cos'`  | コサイン (cos(a)) |
| `'exp'`  | 指数関数 (e^a)    |
| `'log'`  | 自然対数 (ln(a))  |
| `'sqrt'` | 平方根 (sqrt(     | a | )) *負の値でもエラーにならない* |
| `'pow2'` | 2乗 (a^2)         |
| `'pow3'` | 3乗 (a^3)         |
| `'inv'`  | 逆数 (1/a)        |

## 🤝 `scikit-learn`エコシステムとの連携

`mini-sisso`は`scikit-learn`の`BaseEstimator`と`RegressorMixin`を継承しているため、`scikit-learn`が提供する強力なツール群とシームレスに連携できます。

### `Pipeline`による前処理との連結

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mini_sisso.model import MiniSisso

# データ準備
X_df = pd.DataFrame(np.random.rand(100, 2) *, columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1)

# Pipelineの定義
# 注意: MiniSissoは入力特徴量のスケールに敏感なため、StandardScalerのような前処理は推奨されません。
pipeline = Pipeline([
    # ('scaler', StandardScaler()), # MiniSissoでは通常不要/非推奨
    ('sisso', MiniSisso(n_expansion=2, n_term=2, operators=["+", "sin", "pow2"]))
])

# Pipeline全体を学習
pipeline.fit(X_df, y_series)

# 予測
predictions = pipeline.predict(X_df)
print(f"Pipelineによる予測 (一部): {predictions[:5]}")
```

### `GridSearchCV`によるハイパーパラメータチューニング

```python
from sklearn.model_selection import GridSearchCV
from mini_sisso.model import MiniSisso

# データ準備
X_df = pd.DataFrame(np.random.rand(100, 2) *, columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1)

# チューニングしたいハイパーパラメータのグリッドを定義
param_grid = {
    'n_expansion':,
    'n_term':,
    'k':,
    'use_levelwise_sis': [True], # 通常はTrueに固定
    # 'alpha': [0.001, 0.01, 0.1] # lassoを使う場合
}

# GridSearchCVのインスタンスを作成
grid_search = GridSearchCV(
    MiniSisso(operators=["+", "sin", "pow2"], so_method="exhaustive"),
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
)

# ハイパーパラメータの探索を実行
grid_search.fit(X_df, y_series)

print(f"\n最適なハイパーパラメータ: {grid_search.best_params_}")
print(f"最良モデルのRMSE (交差検証): {-grid_search.best_score_:.4f}")
print(f"最良モデルの数式: {grid_search.best_estimator_.equation_}")
```

## ⚙️ APIリファレンス

### `MiniSisso`

```python
class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, n_term: int = 2, k: int = 10, 
                 k_per_level: int = 50, use_levelwise_sis: bool = True,
                 operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01,
                 device: str = "cpu"):
```

#### パラメータ
-   `n_expansion` (int, default=2): 特徴拡張の最大レベル。
-   `n_term` (int, default=2): 見つける数式モデルの最大項数 (`exhaustive`サーチ用)。
-   `k` (int, default=10): SISステップで、各反復で選択する有望な特徴の数。
-   `k_per_level` (int, default=50): `use_levelwise_sis=True`の場合、各拡張レベルで次のステップに引き継ぐ有望なレシピの数。
-   `use_levelwise_sis` (bool, default=True): レベルワイズSIS機能のオン/オフを切り替えます。
-   `device` (str, default="cpu"): 計算に使用するデバイス。`"cuda"`または`"cpu"`。
-   `operators` (list[str], required): 特徴拡張に使用する演算子のリスト。
-   `so_method` (str, default="exhaustive"): モデル探索戦略。`"exhaustive"`または`"lasso"`を選択。
-   `alpha` (float, default=0.01): `so_method="lasso"`の場合に使用する正則化パラメータ。

---

### `fit(X, y)`

モデルを学習させます。

#### パラメータ
-   `X` (array-like or pd.DataFrame): 特徴量データ。形状 `(n_samples, n_features)`。
-   `y` (array-like or pd.Series): ターゲット変数データ。形状 `(n_samples,)`。

#### 戻り値
-   `self`: 学習済みの`MiniSisso`インスタンス。

---

### `predict(X)`

学習済みのモデルを使って予測を行います。

#### パラメータ
-   `X` (array-like or pd.DataFrame): 予測したいデータ。特徴量データの形状 `(n_samples, n_features)`。

#### 戻り値
-   `np.ndarray`: 予測結果のNumPy配列。

---

### `score(X, y)`

モデルの性能を評価します。デフォルトでは決定係数（R²スコア）を返します。

#### パラメータ
-   `X` (array-like or pd.DataFrame): 特徴量データ。
-   `y` (array-like or pd.Series): 真のターゲット変数データ。

#### 戻り値
-   `float`: R²スコア。

---

### 学習済み属性

`fit()`の後に、以下の属性にアクセスできます。

-   `model.equation_` (str): 見つかった最良の数式モデル。
-   `model.rmse_` (float): 最良モデルの訓練データに対するRMSE。
-   `model.r2_` (float): 最良モデルの訓練データに対するR2スコア。
-   `model.coef_` (np.ndarray): 最良モデルの各項の係数。
-   `model.intercept_` (float): 最良モデルの切片。

## 📜 ライセンス
このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞
このライブラリは、オリジナルのSISSOアルゴリズムの論文に大きなインスピレーションを受けています。また、NumPy, SciPy, Pandas, scikit-learn, PyTorchといった素晴らしいオープンソースプロジェクトの上に成り立っています。