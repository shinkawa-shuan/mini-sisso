# mini-sisso

[![PyPI version](https://badge.fury.io/py/mini-sisso.svg)](https://badge.fury.io/py/mini-sisso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/mini-sisso.svg)](https://pypi.org/project/mini-sisso/)

**`mini-sisso` は、シンボリック回帰アルゴリズム SISSO (Sure Independence Screening and Sparsifying Operator) をPythonで実装した、軽量で手軽なライブラリです。scikit-learnエコシステムとの完全な互換性を持ち、データから解釈可能な数式モデルを発見します。**

C++/Fortranベースのオリジナル実装が持つ高度な探索能力を受け継ぎつつ、以下の特徴を提供します。

-   **🚀 手軽な導入と利用**:
    -   `pip install` で簡単インストール。PyTorchのような巨大な依存関係がなく、CPU環境で動くため、環境構築の手間を最小限に抑えます。
    -   `scikit-learn`ライクな`fit()` / `predict()` インターフェースにより、直感的にモデルを構築・評価できます。
-   **🧠 メモリ効率と高速な探索**:
    -   「レシピ化」アーキテクチャにより、特徴拡張（Feature Expansion）時のメモリ消費を劇的に削減します。
    -   「レベルワイズSIS」機能（オン/オフ可能）により、無駄な計算を削減し、探索を高速化します。
    -   `NumPy`/`SciPy`ベースの計算エンジンにより、数値的な安定性と効率性を両立します。
-   **🤝 `scikit-learn`エコシステムとの完全な互換性**:
    -   `GridSearchCV`によるハイパーパラメータチューニング、`Pipeline`による前処理との連携など、`scikit-learn`の強力なツール群とシームレスに統合できます。

このライブラリは、物理法則の発見、材料科学における物性予測、金融モデリングなど、データ背後に潜むメカニズムを解釈可能な数式の形で明らかにしたいあらゆる分野で活用できます。

## 📥 インストール

### GitHubから直接インストール (最新版)

常に最新の開発版をインストールするには、`pip`を使ってこのGitHubリポジトリから直接インストールしてください。

```bash
pip install git+https://github.com/shinkawa-shuan/mini-sisso.git
```

ライブラリをアップデートする場合は、`--upgrade`オプションを使用します。
```bash
pip install --upgrade git+https://github.com/shinkawa-shuan/mini-sisso.git
```

### PyPIからインストール (安定版)

**（注：現在、PyPIには登録されていません。将来的には以下のコマンドでインストール可能になる予定です。）**

安定版をPyPIからインストールするには、以下のコマンドを実行します。

```bash
pip install mini-sisso
```

## 🚀 クイックスタート

わずか数行のコードで、データから数式モデルを学習し、予測を行うことができます。

```python
import pandas as pd
import numpy as np
from mini_sisso.model import MiniSisso # パッケージ名とクラス名に注意

# 1. ベンチマークデータの準備 (scikit-learnのfit/predictに合わせX, yを分離)
np.random.seed(42)
X_df = pd.DataFrame(np.random.rand(100, 2) * [2, 3], columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1, name="target_value")

# 2. MiniSissoモデルのインスタンスを作成
model = MiniSisso(
    n_expansion=2,          # 特徴拡張のレベル
    n_term=2,               # 探すモデルの最大項数
    operators=["+", "sin", "pow2"], # 使用する演算子
    use_levelwise_sis=True, # メモリ効率の良いレベルワイズSISを有効に (デフォルト)
)

# 3. モデルの学習
model.fit(X_df, y_series) # Xとyを分離して渡す

# 4. 学習結果の確認
print(f"発見された数式: {model.equation_}")
print(f"訓練RMSE: {model.rmse_:.4f}")
print(f"訓練R2スコア: {model.r2_:.4f}")

# 5. 新しいデータで予測 (scikit-learnのpredictに合わせXのみを渡す)
X_test_df = pd.DataFrame(np.array([[0.5, 1.0], [1.0, 2.0]]), columns=["feature_A", "feature_B"])
predictions = model.predict(X_test_df)
print(f"\n予測結果: {predictions}")
```

**出力例**:
```
... (学習ログ) ...
Best Model Found (2 terms):
  RMSE: 0.088097
  R2:   0.998927
  Equation: +0.997666 * ^2(feature_B) +2.014116 * sin(feature_A) +0.001270

発見された数式: +0.997666 * ^2(feature_B) +2.014116 * sin(feature_A) +0.001270
訓練RMSE: 0.0881
訓練R2スコア: 0.9989

予測結果: [3.0481014 5.6796246]
```

## 🛠️ 使い方ガイド

### `use_levelwise_sis`: 特徴生成戦略の切り替え

`mini-sisso`の高速化の鍵である「レベルワイズSIS」機能のオン/オフを切り替えます。

#### `True` (デフォルト)
特徴拡張をレベル（段階）ごとに行い、各レベルの直後にスクリーニング（SIS）を実行します。有望な特徴だけを次のレベルの生成に使用するため、計算時間とメモリ使用量を大幅に削減できます。**通常はこちらの使用を推奨します。**

```python
model = MiniSisso(
    use_levelwise_sis=True, # デフォルトなので省略可
    k_per_level=100,        # 各レベルで有望な特徴を100個残す
    # ... other parameters
)
```

#### `False`
特徴拡張の全レベルで考えられるすべての特徴（レシピ）を一度に生成してから、最終的なSIS/SOステップを実行します。
-   **長所**: 探索空間が広がり、思わぬ特徴の組み合わせが見つかる可能性があります。
-   **短所**: **メモリ使用量と計算時間が爆発的に増加します。** `n_expansion`が大きい場合や、ベース特徴量の数が多い場合は、メモリ不足でプログラムがクラッシュするリスクがあります。

```python
model = MiniSisso(
    use_levelwise_sis=False,
    n_expansion=2, # メモリ消費を抑えるため、小さい値に設定することを推奨
    # ... other parameters
)
```

### `so_method`: モデル探索戦略の選択

`MiniSisso`は2つのモデル探索戦略（Sparsifying Operator）を提供します。

#### 1. `exhaustive` (デフォルト)
候補となる特徴のすべての組み合わせをテストする総当たり探索です。
-   **長所**: 最適な組み合わせを見つけられる可能性が最も高い。シンプルで解釈しやすいモデルが見つかりやすい。
-   **短所**: 候補特徴や項数が増えると、計算時間が爆発的に増加する。

```python
model = MiniSisso(
    n_term=3,
    so_method="exhaustive", # デフォルトなので省略可
    operators=["+", "-", "*", "sqrt"]
)
```

#### 2. `lasso`
Lasso回帰を用いて、多数の候補から重要な特徴を高速に選択します。
-   **長所**: 非常に高速。`exhaustive`では現実的でない大規模な探索空間でも有効なモデルを見つけられる可能性がある。
-   **短所**: `alpha`パラメータの調整が必要。見つかるモデルが複雑になりがちで、必ずしも最適解とは限らない。

```python
model = MiniSisso(
    so_method="lasso",
    alpha=0.01, # Lassoの正則化パラメータ。小さいほど多くの特徴を選択する。
    operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "pow2", "pow3"]
)
model.fit(X_df, y_series) # Xとyを分離して渡す
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
from mini_sisso.model import MiniSisso # パッケージ名とクラス名に注意

# データ準備
X_df = pd.DataFrame(np.random.rand(100, 2) * [2, 3], columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1, name="target_value")

# Pipelineの定義
# 注意: MiniSissoは入力特徴量のスケールに敏感なため、StandardScalerのような前処理は推奨されません。
# ここではPipelineが動作することを示すための例として、StandardScalerは含めません。
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
from mini_sisso.model import MiniSisso # パッケージ名とクラス名に注意

# データ準備 (上記と同じ)
X_df = pd.DataFrame(np.random.rand(100, 2) * [2, 3], columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1, name="target_value")

# チューニングしたいハイパーパラメータのグリッドを定義
param_grid = {
    'n_expansion': [1, 2],
    'n_term': [1, 2],
    'k': [10, 20],
    'use_levelwise_sis': [True], # 通常はTrueに固定
    # 'alpha': [0.001, 0.01, 0.1] # lassoを使う場合
}

# GridSearchCVのインスタンスを作成
# scoring='neg_root_mean_squared_error' でRMSEを評価
grid_search = GridSearchCV(
    MiniSisso(operators=["+", "sin", "pow2"], so_method="exhaustive"), # MiniSissoの共通パラメータを設定
    param_grid,
    cv=3, # 3分割交差検証
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, # 利用可能なCPUコアをすべて使う
    verbose=1, # ログを詳細に出力
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
                 operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01):
```

#### パラメータ
-   `n_expansion` (int, default=2): 特徴拡張の最大レベル。
-   `n_term` (int, default=2): 見つける数式モデルの最大項数 (`exhaustive`サーチ用)。
-   `k` (int, default=10): SISステップで、各反復で選択する有望な特徴の数。
-   `k_per_level` (int, default=50): `use_levelwise_sis=True`の場合、各拡張レベルで次のステップに引き継ぐ有望なレシピの数。
-   `use_levelwise_sis` (bool, default=True): レベルワイズSIS機能のオン/オフを切り替えます。
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
-   `X` (array-like or pd.DataFrame): 予測したいデータ。特徴量データの形状 `(n_samples, n_features)`。**学習時と同じ特徴量名と順序**を持つことを推奨します（`pd.DataFrame`の場合）。

#### 戻り値
-   `np.ndarray`: 予測結果のNumPy配列。

---

### `score(X, y)`

`scikit-learn`の規約に従い、モデルの性能を評価します。デフォルトでは決定係数（R²スコア）を返します。

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
-   `model.best_model_recipes_` (tuple): 最良モデルを構成する`FeatureRecipe`オブジェクトのタプル。
-   `model.coef_` (np.ndarray): 最良モデルの各項の係数。
-   `model.intercept_` (float): 最良モデルの切片。
-   `model.base_feature_names_` (list[str]): 学習に使用した特徴量の名前のリスト。

## 📜 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は`LICENSE`ファイルをご覧ください。

## 🙏 謝辞

このライブラリは、オリジナルのSISSOアルゴリズムの論文に大きなインスピレーションを受けています。また、NumPy、SciPy、Pandas、scikit-learnといった素晴らしいオープンソースプロジェクトの上に成り立っています。 