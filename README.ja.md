
# mini-sisso

[![PyPI version](https://badge.fury.io/py/mini-sisso.svg)](https://pypi.org/project/mini-sisso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/mini-sisso.svg)](https://pypi.org/project/mini-sisso/)

**`mini-sisso` は、シンボリック回帰アルゴリズム SISSOをPythonで実装した、軽量で手軽なライブラリです。scikit-learnエコシステムと完全に互換性があり、データから人間が解釈可能な数式モデルを発見します。**

### SISSOとの違い：なぜ`mini-sisso`なのか？

`mini-sisso`は、C++/Fortranベースのオリジナル実装が持つ高度な探索能力を、よりモダンで使いやすい形で提供します。

-   **🧠 メモリ効率と高速な探索**:
    -   **「レシピ化」アーキテクチャ**: 特徴拡張時のメモリ消費を劇的に削減し、オリジナル実装ではメモリ不足でクラッシュするような大規模な問題も扱えます。
    -   **「レベルワイズSIS」機能**: 無駄な計算を早期に枝刈りし、探索を大幅に高速化します。
-   **🚀 手軽な導入と利用**:
    -   `pip install` で簡単インストール。複雑なコンパイルは不要です。
    -   `scikit-learn`ライクな`fit()` / `predict()` インターフェースにより、直感的にモデルを構築・評価できます。
-   **🤝 `scikit-learn`エコシステムとの完全な互換性**:
    -   `GridSearchCV`によるハイパーパラメータ自動探索や、`Pipeline`によるワークフロー構築が可能です。
-   **⚡ 柔軟な探索戦略とGPUサポート**:
    -   古典的な`exhaustive`（総当たり）探索に加え、高速な`Lasso`や`LightGBM`を特徴選択器として利用できます。
    -   オプションでGPUアクセラレーションにも対応し、さらなる高速化が可能です。

## 📥 インストール

### CPU版 (デフォルト・推奨)

PyPIからインストールします。NumPy, SciPy, scikit-learn, LightGBM, dcorに依存します。

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
X_df = pd.DataFrame(np.random.rand(100, 2) *, columns=["feature_A", "feature_B"])
# 真の式: y = 2*sin(feature_A) + feature_B^2 + ノイズ
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1)

# 2. モデルのインスタンス化 (全ハイパーパラメータ)
# 実際に使うもの以外はコメントアウトして、デフォルト値を使用します。
model = MiniSisso(
    # --- 基本的な探索空間の制御 ---
    n_expansion=2,                      # 特徴拡張のレベル (深くするほど複雑な式を発見)
    operators=["+", "sin", "pow2"],     # 特徴拡張に使う演算子リスト
    
    # --- 探索戦略の主要な選択 ---
    so_method="exhaustive",             # モデル探索戦略 ('exhaustive', 'lasso', 'lightgbm')
    
    # --- 各戦略の詳細設定 (selection_params) ---
    selection_params={
        # -- "exhaustive"メソッド用のパラメータ --
        'n_term': 2,                    # 発見する数式の最大項数
        'n_sis_features': 10,           # 各項を見つけるためのSIS候補数
        
        # -- "lasso"メソッド用のパラメータ --
        # 'alpha': 0.01,                # Lassoの正則化の強さ
        
        # -- "lightgbm"メソッド用のパラメータ --
        # 'n_features_to_select': 20,   # LightGBMで選択する特徴の数
        # 'lightgbm_params': {'n_estimators': 100, 'random_state': 42}, # LightGBMモデル自体のパラメータ
        
        # -- "lasso"/"lightgbm"用の前処理フィルター (オプション) --
        # 'n_global_sis_features': 200, # 最初にターゲットとの相関で候補を絞る数
        # 'collinearity_filter': 'mi',  # 候補同士の相関を計算する方法 ('mi' or 'dcor')
        # 'collinearity_threshold': 0.9, # 上記フィルターで削除する相関の閾値
    },
    
    # --- 計算効率の制御 ---
    use_levelwise_sis=True,             # 段階的探索による高速化 (Trueを強く推奨)
    n_level_sis_features=50,            # 各拡張レベルで残す有望な特徴の数
    
    # --- 実行環境の選択 ---
    # device="cuda",                      # GPUを使う場合は 'cuda' を指定
)

# 3. モデルの学習
model.fit(X_df, y_series)

# 4. 学習結果の確認
print("\n--- 学習結果 ---")
print(f"発見された数式: {model.equation_}")
print(f"訓練RMSE: {model.rmse_:.4f}")
print(f"訓練R2スコア: {model.r2_:.4f}")

# 5. 新しいデータで予測
print("\n--- 予測 ---")
X_test_df = pd.DataFrame(np.array([, ]), columns=["feature_A", "feature_B"])
predictions = model.predict(X_test_df)
print(f"新しいデータ ([0.5, 1.0], [1.0, 2.0]) に対する予測結果: {predictions}")
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

--- 学習結果 ---
発見された数式: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610
訓練RMSE: 0.0921
訓練R2スコア: 0.9988

--- 予測 ---
新しいデータに対する予測結果: [2.0016012 5.6796584]
```

## 🛠️ 使い方ガイド：ハイパーパラメータによる探索制御

`mini-sisso`の探索プロセスは、以下のワークフローで構成され、各ステップはハイパーパラメータで制御されます。

### ワークフロー概要

1.  **特徴拡張 (Feature Expansion)**: `operators`と`n_expansion`に基づき、多数の候補特徴を生成します。
    -   この過程は `use_levelwise_sis=True` と `n_level_sis_features` によって効率化されます（後述）。
2.  **[任意] 前処理フィルター (Preprocessing Filters)**: `lasso`/`lightgbm`使用時に、候補特徴を絞り込むためのフィルター群です。(`selection_params`で設定)
    -   **大域的SIS (Global SIS)**: ターゲット`y`との相関が低い特徴を除外します。
    -   **多重共線性フィルター (Collinearity Filter)**: 候補特徴同士の相関が高すぎるものを削除します。
3.  **モデル探索 (Sparsifying Operator)**: `so_method`で指定された戦略で、絞り込まれた候補の中から最終的なモデルを発見します。

---

### 主要なハイパーパラメータ

#### `so_method`: 3つのモデル探索戦略

`so_method`を選ぶことで、探索の基本的なアプローチを決定します。

##### 1. `so_method="exhaustive"` (デフォルト)
SISSOの古典的なアプローチ。SISで有望な特徴を絞り込みながら、**総当たり探索**で最適なモデルを探します。解釈しやすいシンプルなモデルが見つかりやすいです。

```python
# 3項までのモデルを総当たりで探索
model = MiniSisso(
    so_method="exhaustive",
    selection_params={
        'n_term': 3,          # 探索する最大項数
        'n_sis_features': 15  # 各項を見つけるためにプールに追加する候補数
    }
)
```

##### 2. `so_method="lasso"`
**Lasso回帰**を特徴選択器として使い、高速にモデルを構築します。大規模な特徴空間で有効です。

```python
# Lassoで特徴選択
model = MiniSisso(
    so_method="lasso",
    selection_params={
        'alpha': 0.01 # Lassoの正則化パラメータ
    }
)
```

##### 3. `so_method="lightgbm"`
**LightGBM**を特徴選択器として使います。非線形な関係性を捉える能力に優れています。

```python
# LightGBMで上位20個の特徴を選択
model = MiniSisso(
    so_method="lightgbm",
    selection_params={
        'n_features_to_select': 20
    }
)
```

---
#### `selection_params`: 各戦略の詳細制御

`selection_params`辞書を使うことで、各`so_method`の挙動を細かく制御したり、前処理フィルターを適用したりできます。

##### 前処理フィルター (`lasso`/`lightgbm`用)

-   **`n_global_sis_features`**: 大域的SISによる事前スクリーニング。最初に、ターゲット`y`と全く相関のない特徴をまとめて除外します。
-   **`collinearity_filter`**: 多重共線性（マルチコ）の排除。`'mi'` (相互情報量) or `'dcor'` (距離相関) を指定できます。

```python
# 全候補からyとの相関が高い上位200個に絞り込み、
# さらにMIが0.9以上のペアを除外してからLightGBMを実行
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

##### エキスパート向け設定 (`lightgbm`用)
`lightgbm`の内部ハイパーパラメータを直接指定することも可能です。
```python
model = MiniSisso(
    so_method='lightgbm',
    selection_params={
        'n_features_to_select': 20,
        'lightgbm_params': {
                'n_estimators': 100,         # 木の数: 100本あれば特徴重要度の評価には十分なことが多い
                'num_leaves': 31,            # 葉の最大数: デフォルト値。バランスが良い
                'max_depth': -1,             # 木の深さ: -1は無制限。num_leavesで制御するため、通常は-1でOK
                'learning_rate': 0.1,        # 学習率: デフォルト値。n_estimatorsとのバランスで決まる
                'colsample_bytree': 0.8,     # 特徴量のランダムサンプリング率: 過学習を防ぐための一般的な値
                'subsample': 0.8,            # データ（行）のランダムサンプリング率: 同上
                'reg_alpha': 0.1,            # L1正則化: わずかに正則化をかける
                'reg_lambda': 0.1,           # L2正則化: 同上
                'random_state': 42,          # 再現性のための乱数シード
                'n_jobs': -1,                # 利用可能なCPUコアをすべて使用
                'verbosity': -1,             # LightGBMのログを非表示に
            }   
        }
)
```

---
#### その他の主要パラメータ

-   `use_levelwise_sis` (bool, default=True): 特徴生成を段階的に行い、計算を高速化・省メモリ化します。**オフにすると計算量が爆発する可能性があるため、`True`を強く推奨します。**
-   `n_level_sis_features` (int, default=50): `use_levelwise_sis=True`の場合、各レベルで残す有望な特徴の数です。
-   `device` (str, default="cpu"): 計算バックエンド。`"cuda"`を指定するとGPUを使用します。

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

### `Pipeline`のより詳しい使い方

`Pipeline`は、複数の処理ステップを連結し、一つの推定器として扱うためのツールです。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mini_sisso.model import MiniSisso

# Pipelineの定義
# 注意: MiniSissoは入力特徴量のスケールに敏感なため、StandardScalerのような前処理は
# 発見される数式の解釈性を損なう可能性があります。通常は非推奨です。
# ここでは、Pipelineが技術的に動作することを示すための例です。
pipeline = Pipeline([
    # ステップ1: 'scaler'という名前で標準化を実行
    ('scaler', StandardScaler()), # MiniSissoでは通常不要/非推奨
    # ステップ2: 'sisso'という名前でMiniSissoを実行
    ('sisso', MiniSisso(n_expansion=2, selection_params={'n_term': 2}, operators=["+", "sin", "pow2"]))
])

# Pipeline全体を学習: X -> scaler.fit_transform -> sisso.fit
pipeline.fit(X_df, y_series)

# Pipelineを使って予測: X -> scaler.transform -> sisso.predict
predictions = pipeline.predict(X_df)

# パイプラインの各ステップのパラメータにアクセス・変更も可能
# 例: 学習後にSISSOの項数を変更する
# pipeline.set_params(sisso__selection_params={'n_term': 3})
print(f"PipelineのSISSOステップの項数: {pipeline.named_steps['sisso'].selection_params['n_term']}")
```

### `GridSearchCV`のより詳しい使い方

`GridSearchCV`は、ハイパーパラメータの最適な組み合わせを交差検証によって自動で探索します。`__`（ダブルアンダースコア）を使うことで、`selection_params`のような辞書内のパラメータも探索対象にできます。

#### 例：`exhaustive`と`lasso`で最適な手法とパラメータを同時に探索

```python
from sklearn.model_selection import GridSearchCV

# 探索したいパターンのリストを作成
param_grid = [
    # パターン1: exhaustiveメソッドの探索
    {
        'so_method': ['exhaustive'],
        'selection_params': [
            {'n_term': 2, 'n_sis_features': 10},
            {'n_term': 3, 'n_sis_features': 15}
        ]
    },
    # パターン2: lassoメソッドの探索
    {
        'so_method': ['lasso'],
        'selection_params': [
            {'alpha': 0.01, 'collinearity_filter': 'mi', 'collinearity_threshold': 0.9},
            {'alpha': 0.005, 'collinearity_filter': 'mi'}
        ]
    }
    # パターン3: lightgbmメソッドの探索
    {
        'so_method': ['lightgbm'],
        # selection_params内のパラメータを探索
        'selection_params__n_features_to_select': [10, 20],
        # lightgbm_params内のパラメータを探索 (二重アンダースコアに注意)
        'selection_params__lightgbm_params__n_estimators': [100, 200],
        'selection_params__lightgbm_params__num_leaves': [20, 31],
    }

]

# GridSearchCVのインスタンスを作成
grid_search = GridSearchCV(
    MiniSisso(n_expansion=2, operators=['+', 'sin', 'pow2']),
    param_grid,
    cv=3,   # 3分割交差検証
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, # 利用可能なCPUコアをすべて使う
    verbose=1, # ログを詳細に出力
)

# ハイパーパラメータ探索を実行
print("Starting GridSearchCV to find the best method and parameters...")
grid_search.fit(X_df, y_series)

print(f"\n最適だった探索手法とパラメータ: {grid_search.best_params_}")
print(f"最良モデルの数式: {grid_search.best_estimator_.equation_}")
```

## ⚙️ APIリファレンス

### `MiniSisso`
```python
class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, operators: list = None,
                 so_method: str = "exhaustive", selection_params: dict = None,
                 use_levelwise_sis: bool = True, n_level_sis_features: int = 50,
                 device: str = "cpu"):
```
#### パラメータ
-   `n_expansion` (int, default=2): 特徴拡張の最大レベル。
-   `operators` (list[str], required): 特徴拡張に使用する演算子のリスト。
-   `so_method` (str, default="exhaustive"): モデル探索戦略。`"exhaustive"`, `"lasso"`, `"lightgbm"`から選択。
-   `selection_params` (dict, optional): 各探索戦略の詳細な挙動を制御するパラメータの辞書。キーは`n_term`, `n_sis_features`, `alpha`, `n_features_to_select`, `lightgbm_params`, `n_global_sis_features`, `collinearity_filter`, `collinearity_threshold`など。
-   `use_levelwise_sis` (bool, default=True): レベルワイズSIS機能のオン/オフ。
-   `n_level_sis_features` (int, default=50): `use_levelwise_sis=True`の場合、各拡張レベルで残す特徴の数。
-   `device` (str, default="cpu"): 計算に使用するデバイス (`"cpu"` or `"cuda"`)。
 
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