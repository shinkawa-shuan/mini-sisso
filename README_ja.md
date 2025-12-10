
# mini-sisso

[![PyPI version](https://badge.fury.io/py/mini-sisso.svg)](https://pypi.org/project/mini-sisso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/mini-sisso.svg)](https://pypi.org/project/mini-sisso/)

**`mini-sisso` は、SISSO (Sure Independence Screening and Sparsifying Operator) シンボリック回帰アルゴリズムの軽量で使いやすいPython実装です。scikit-learnエコシステムと完全に互換性があり、データから解釈可能な数式モデルを発見します。**

C++/Fortranベースのオリジナル実装が持つ高度な探索能力を受け継ぎつつ、`mini-sisso`は**爆速のRustバックエンド**によって、よりモダンでアクセスしやすいパッケージとしてこれらの機能を提供します：

-   **🚀 手軽な導入**: `pip install` するだけです。デフォルトのCPU版は依存関係が最小限（NumPy/SciPy）で、セットアップに悩みません。
-   **🦀 高パフォーマンスなRustバックエンド**:
    -   計算負荷の高い `exhaustive`（総当たり）探索は **Rust** で実装されており、純粋なPython実装よりも遥かに高速です。ユーザーはRustを意識することなく利用できます。
-   **🧠 メモリ効率と高速な探索**:
    -   「レシピベース」のアーキテクチャにより、特徴拡張時のメモリ消費を劇的に削減します。
    -   「レベルワイズSIS」機能（切り替え可能）は、見込みのない特徴を早期に枝刈りすることで探索を高速化します。
-   **⚖️ バランスの取れた特徴量選抜 (Split Selection)**:
    -   SIS（スクリーニング）の段階で、単項演算子（Unary）と二項演算子（Binary）の選抜枠を個別に確保するロジックを搭載。二項演算子の組み合わせ爆発によって単項演算子が締め出される（Crowding Out）のを防ぎ、多様な特徴量を維持します。
-   **🤝 完全な `scikit-learn` 互換性**: 標準的な `fit()`/`predict()` インターフェースに加え、`GridSearchCV` や `Pipeline` といった強力なツールとシームレスに統合できます。
-   **⚡ オプションのGPUサポート**: オプションのPyTorchバックエンドをインストールすることで、GPUアクセラレーションによる大幅な高速化が可能です。

## 📥 インストール

### CPU版 (デフォルト・推奨)

PyPIから軽量なCPU版をインストールします。これには最適化されたRustバックエンドが含まれています。

```bash
pip install mini-sisso
```

### GPU版 (オプション)

PyTorchバックエンドによるGPUアクセラレーションを有効にするには、`[gpu]` オプションを付けてインストールしてください。

```bash
pip install "mini-sisso[gpu]"
```

## 🚀 クイックスタート

わずか数行のコードで、データから数式モデルを発見できます。

```python
import pandas as pd
import numpy as np
from mini_sisso.model import MiniSisso

# 1. データの準備
np.random.seed(42) # 再現性のためにシードを固定
X_df = pd.DataFrame(np.random.rand(100, 2) * [2, 3], columns=["feature_A", "feature_B"])
# 真の式: y = 2*sin(feature_A) + feature_B^2 + ノイズ
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1)

# 2. モデルのインスタンス化 (全ハイパーパラメータリスト)
# 変更が必要なパラメータのコメントを外してください。
model = MiniSisso(
    # --- 基本的な探索空間の制御 ---
    n_expansion=2,                      # 特徴拡張の深さ (深くするほど複雑な式が見つかります)
    operators=["+", "sin", "pow2"],     # 特徴拡張に使用する演算子のリスト
    
    # --- メインの探索戦略の選択 ---
    so_method="exhaustive",             # モデル探索戦略 ('exhaustive', 'lasso', 'lightgbm')
    
    # --- 各戦略の詳細設定 (selection_params) ---
    selection_params={
        # -- "exhaustive" メソッド用パラメータ --
        'n_term': 2,                    # 発見される数式の最大項数
        'n_sis_features': 10,           # 各項のSIS候補数
        
        # -- "lasso" メソッド用パラメータ --
        # 'alpha': 0.01,                # Lassoの正則化の強さ
        
        # -- "lightgbm" メソッド用パラメータ --
        # 'n_features_to_select': 20,   # LightGBMで選択する特徴の数
        # 'lightgbm_params': {'n_estimators': 100, 'random_state': 42}, # LightGBMモデル自体のパラメータ
        
        # -- "lasso"/"lightgbm" 用のオプション前処理フィルター --
        # 'n_global_sis_features': 200, # ターゲットとの相関に基づいて候補を事前スクリーニングする数
        # 'collinearity_filter': 'mi',  # 候補間の相関を計算する方法 ('mi' または 'dcor')
        # 'collinearity_threshold': 0.9, # 上記フィルターの相関閾値
    },
    
    # --- 計算効率の制御 ---
    use_levelwise_sis=True,             # 段階的探索を使用して高速化 (強く推奨)
    n_level_sis_features=50,            # 各拡張レベルで保持する有望な特徴の数
    
    # --- 実行環境の選択 ---
    # device="cuda",                      # GPUを使用する場合は 'cuda' を指定
)

# 3. モデルの学習
model.fit(X_df, y_series)

# 4. 結果の確認
print("\n--- Fit Results ---")
print(f"Discovered Equation: {model.equation_}")
print(f"Training RMSE: {model.rmse_:.4f}")
print(f"Training R2 Score: {model.r2_:.4f}")

# 5. 予測の実行
print("\n--- Prediction ---")
X_test_df = pd.DataFrame(np.array([[0.5, 1.0], [1.0, 2.0]]), columns=["feature_A", "feature_B"])
predictions = model.predict(X_test_df)
print(f"Predictions for new data ([0.5, 1.0], [1.0, 2.0]): {predictions}")
```

**出力例**:
```
...
Best Model Found (2 terms):
  RMSE: 0.0921
  R2:   0.9988
  Equation: +0.9985 * ^2(feature_B) +1.9712 * sin(feature_A) +0.0306

--- Fit Results ---
Discovered Equation: +0.9985 * ^2(feature_B) +1.9712 * sin(feature_A) +0.0306
Training RMSE: 0.0921
Training R2 Score: 0.9988

--- Prediction ---
Predictions for new data ([0.5, 1.0], [1.0, 2.0]): [2.0016 5.6797]
```

## 🛠️ 使い方ガイド: ハイパーパラメータによる探索の制御

`mini-sisso` の探索プロセスは以下のワークフローに従い、各ステップはハイパーパラメータによって制御されます。

### ワークフローの概要

1.  **特徴拡張 (Feature Expansion)**: `operators` と `n_expansion` に基づいて多数の候補特徴を生成します。
    -   このプロセスは `use_levelwise_sis=True` と `n_level_sis_features` によって効率化されます。
2.  **[オプション] 前処理フィルター**: `lasso` または `lightgbm` を使用する際に、候補特徴を削減するための一連のフィルターです (`selection_params` で設定)。
    -   **Global SIS**: ターゲット `y` との相関が低い特徴を削除します。
    -   **多重共線性フィルター (Collinearity Filter)**: 互いに高い相関を持つ特徴を削除します。
3.  **モデル探索 (Sparsifying Operator)**: `so_method` で指定された戦略を使用して、削減された候補から最終的なモデルを発見します。

---

### 主要なハイパーパラメータ

#### `so_method`: 3つのモデル探索戦略

`so_method` パラメータは、中心となる探索アプローチを決定します。

##### 1. `so_method="exhaustive"` (デフォルト)
古典的なSISSOアプローチです。反復的なSISと、**Rustによる総当たり探索**を使用して最適なモデルを見つけます。シンプルで解釈可能なモデルを見つけるのに最適です。

```python
# 3項までのモデルを総当たりで探索
model = MiniSisso(
    so_method="exhaustive",
    selection_params={
        'n_term': 3,          # 探索する最大項数
        'n_sis_features': 15  # 各SISステップでプールに追加する候補数
    }
)
```

##### 2. `so_method="lasso"`
**Lasso回帰** を特徴選択器として使用し、モデルを素早く構築します。大規模な特徴空間に対して有効です。

```python
# Lassoを使用して特徴を選択
model = MiniSisso(
    so_method="lasso",
    selection_params={
        'alpha': 0.01 # Lassoの正則化パラメータ
    }
)
```

##### 3. `so_method="lightgbm"`
**LightGBM** を特徴選択器として使用します。非線形な関係を捉えるのに優れています。

```python
# LightGBMを使用して上位20個の特徴を選択
model = MiniSisso(
    so_method="lightgbm",
    selection_params={
        'n_features_to_select': 20
    }
)
```

---
#### `selection_params`: 各戦略の詳細制御

`selection_params` 辞書を使用すると、前処理フィルターを適用したり、各 `so_method` を微調整したりできます。

##### 前処理フィルター (`lasso`/`lightgbm` 用)

-   **`n_global_sis_features`**: ターゲット `y` との相関が低いものを削除して候補を事前スクリーニングします。
-   **`collinearity_filter`**: Lasso/LightGBMを安定させるために、高い相関を持つ特徴を削除します。`'mi'` (相互情報量) または `'dcor'` (距離相関) を指定できます。

```python
# LightGBMを実行する前に、候補を上位200個にスクリーニングし、
# さらにMIスコアが0.9を超えるペアを削除
model = MiniSisso(
    so_method='lightgbm',
    selection_params={
        'n_global_sis_features': 200,
        'collinearity_filter': 'mi',
        'collinearity_threshold': 0.9,
        'n_features_to_select': 20
    }
)
```

##### エキスパート設定 (`lightgbm` 用)
LightGBMモデルの内部ハイパーパラメータを直接指定することもできます。

```python
model = MiniSisso(
    so_method='lightgbm',
    selection_params={
        'n_features_to_select': 20,
        'lightgbm_params': {
            'n_estimators': 200,         # 木の数
            'num_leaves': 31,            # 1つの木における葉の最大数
            'learning_rate': 0.05,       # 学習率
            'colsample_bytree': 0.8,     # 各木で考慮される特徴の割合
            'subsample': 0.8,            # 各木で使用されるデータの割合
            'reg_alpha': 0.1,            # L1正則化
            'reg_lambda': 0.1,           # L2正則化
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
    }
)
```

---
#### その他の主要パラメータ

-   `use_levelwise_sis` (bool, default=True): **強く推奨します。** 特徴生成を高速化し、メモリを節約します。
-   `n_level_sis_features` (int, default=50): `use_levelwise_sis=True` の場合に各ステージで保持する特徴の数。
-   `device` (str, default="cpu"): GPUバックエンドを使用するには `"cuda"` に設定します。

### 利用可能な演算子
`operators` 引数に文字列のリストとして指定します。

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
| `'sqrt'` | 平方根 (sqrt(     | a | )) *負の値でもエラーになりません* |
| `'pow2'` | 2乗 (a^2)         |
| `'pow3'` | 3乗 (a^3)         |
| `'inv'`  | 逆数 (1/a)        |
| `'\|-\|'` | 絶対差分 (\|a - b\|) |
| `'cbrt'` | 立方根 (a^(1/3)) |
| `'abs'` | 絶対値 (\|a\|) |
| `'scd'` | 標準コーシー分布 (1 / (π * (1 + a^2))) |

## 🤝 `scikit-learn` エコシステムとの統合
`mini-sisso` は `scikit-learn` の `BaseEstimator` と `RegressorMixin` を継承しており、`scikit-learn` が提供する強力なツールとシームレスに統合できます。

### `Pipeline` のより詳細な使用法

`Pipeline` は、複数の処理ステップを接続し、それらを単一の推定器として扱うためのツールです。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mini_sisso.model import MiniSisso

# パイプラインの定義
# 注意: MiniSissoは入力特徴のスケールに敏感であるため、StandardScalerのような前処理は
# 発見された式の解釈可能性を損なう可能性があります。一般的には推奨されません。
# ここでは、Pipelineが技術的にどのように機能するかを示す例です。
pipeline = Pipeline([
# ステップ 1: 'scaler' という名前で標準化を実行
('scaler', StandardScaler()), # MiniSissoでは通常不要/非推奨
# ステップ 2: 'sisso' という名前でMiniSissoを実行
('sisso', MiniSisso(n_expansion=2, selection_params={'n_term': 2}, operators=["+", "sin", "pow2"]))
])

# パイプライン全体を学習: X -> scaler.fit_transform -> sisso.fit
pipeline.fit(X_df, y_series)

# パイプラインを使用して予測: X -> scaler.transform -> sisso.predict
predictions = pipeline.predict(X_df)

# パイプラインの各ステップのパラメータにアクセスして変更することもできます。
# 例: 学習後にSISSOの項数を変更する
# pipeline.set_params(sisso__selection_params={'n_term': 3})
print(f"Number of terms in the SISSO step of the pipeline: {pipeline.named_steps['sisso'].selection_params['n_term']}")
```

### 高度な `GridSearchCV` の使用法

`GridSearchCV` は、`so_method` 自体を含むハイパーパラメータの最適な組み合わせを自動的に見つけることができます。`__` (ダブルアンダースコア) 構文を使用すると、`selection_params` 内のネストされたパラメータを検索できます。

```python
from sklearn.model_selection import GridSearchCV

# 検索するパラメータグリッドのリストを定義
param_grid = [
    # ケース 1: exhaustiveメソッドの検索パターン
    {
        'so_method': ['exhaustive'],
        'selection_params': [
            {'n_term': 2, 'n_sis_features': 10},
            {'n_term': 3, 'n_sis_features': 15}
        ]
    },
    # ケース 2: lassoメソッドの検索パターン
    {
        'so_method': ['lasso'],
        'selection_params': [
            {'alpha': 0.01, 'collinearity_filter': 'mi'},
            {'alpha': 0.005}
        ]
    },
    # ケース 3: lightgbmメソッドの検索パターン
    {
        'so_method': ['lightgbm'],
        'selection_params__n_features_to_select': [10, 20],
        'selection_params__lightgbm_params__n_estimators': [100, 200],
    }
]

grid_search = GridSearchCV(
    MiniSisso(n_expansion=2, operators=['+', 'sin', 'pow2']),
    param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1
)

print("Starting GridSearchCV to find the best method and parameters...")
grid_search.fit(X_df, y_series)

print(f"\nBest search method and params: {grid_search.best_params_}")
print(f"Equation from the best model: {grid_search.best_estimator_.equation_}")
```

## ⚙️ API リファレンス

### `MiniSisso`
```python
class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, operators: list = None,
                 so_method: str = "exhaustive", selection_params: dict = None,
                 use_levelwise_sis: bool = True, n_level_sis_features: int = 50,
                 device: str = "cpu"):
```

### `MiniSisso`
-   `n_expansion` (int, default=2): 特徴拡張の最大レベル。
-   `operators` (list[str], required): 特徴生成のための演算子のリスト。
-   `so_method` (str, default="exhaustive"): モデル探索戦略 (`"exhaustive"`, `"lasso"`, `"lightgbm"`).
-   `selection_params` (dict, optional): 選択された `so_method` と前処理フィルターのための詳細パラメータの辞書。
-   `use_levelwise_sis` (bool, default=True): レベルワイズSIS機能を切り替えます。
-   `n_level_sis_features` (int, default=50): `use_levelwise_sis=True` の場合に各レベルで保持する特徴の数。
-   `device` (str, default="cpu"): 計算デバイス (`"cpu"` または `"cuda"`).

---

### `fit(X, y)`

モデルをトレーニングデータに適合させます。

#### パラメータ
-   `X` (array-like or pd.DataFrame): 特徴データ、形状 `(n_samples, n_features)`。
-   `y` (array-like or pd.Series): ターゲット変数データ、形状 `(n_samples,)`。

#### 戻り値
-   `self`: 適合された `MiniSisso` インスタンス。

---

### `predict(X)`

適合されたモデルを使用して予測を行います。

#### パラメータ
-   `X` (array-like or pd.DataFrame): 予測を行うデータ。

#### 戻り値
-   `np.ndarray`: 予測のNumPy配列。

---

### `score(X, y)`

予測の決定係数 (R² スコア) を返します。

#### パラメータ
-   `X` (array-like or pd.DataFrame): 特徴データ。
-   `y` (array-like or pd.Series): 真のターゲット変数データ。

#### 戻り値
-   `float`: R² スコア。

---

### 適合後の属性

`fit()` を呼び出した後、以下の属性にアクセスできます:

-   `model.equation_` (str): 発見された最良の数式モデル。
-   `model.rmse_` (float): トレーニングデータに対する最良モデルのRMSE。
-   `model.r2_` (float): トレーニングデータに対する最良モデルのR2スコア。
-   `model.coef_` (np.ndarray): 最良モデルの各項の係数。
-   `model.intercept_` (float): 最良モデルの切片。

## 📜 ライセンス
このプロジェクトはMITライセンスの下でライセンスされています。

## 🙏 謝辞
このライブラリは、オリジナルのSISSOアルゴリズムの論文に多大なインスピレーションを受けており、NumPy、SciPy、Pandas、scikit-learn、PyTorchなどの素晴らしいオープンソースプロジェクトの上に構築されています。