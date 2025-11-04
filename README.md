# mini-sisso

# mini-sisso

[![PyPI version](https://badge.fury.io/py/mini-sisso.svg)](https://pypi.org/project/mini-sisso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/mini-sisso.svg)](https://pypi.org/project/mini-sisso/)

**`mini-sisso` is a lightweight and user-friendly Python implementation of the SISSO (Sure Independence Screening and Sparsifying Operator) symbolic regression algorithm. It offers full compatibility with the scikit-learn ecosystem for discovering interpretable mathematical models from data.**

Inheriting the advanced exploration capabilities of the original C++/Fortran-based implementation, `mini-sisso` provides these features in a more modern and accessible package:

-   **🚀 Easy Adoption**: Simple `pip install`. The default CPU version has minimal dependencies (NumPy/SciPy), ensuring a hassle-free setup.
-   **🧠 Memory Efficiency & Fast Exploration**:
    -   A "recipe-based" architecture dramatically reduces memory consumption during Feature Expansion.
    -   The "Level-wise SIS" feature (toggleable) speeds up exploration by pruning unpromising features early.
-   **🤝 Full `scikit-learn` Compatibility**: Seamlessly integrates with powerful tools like `GridSearchCV` and `Pipeline`, in addition to the standard `fit()`/`predict()` interface.
-   **⚡ Optional GPU Support**: Achieve significant speedups with GPU acceleration by installing the optional PyTorch backend.

## 📥 Installation

### CPU Version (Default, Recommended)

Installs the lightweight CPU version from PyPI, which depends only on NumPy/SciPy.

```bash
pip install mini-sisso
```

### GPU Version (Optional)

To enable GPU acceleration with the PyTorch backend, install with the `[gpu]` option.

```bash
pip install "mini-sisso[gpu]"
```

## 🚀 Quick Start

Discover a mathematical model from your data in just a few lines of code.

```python
import pandas as pd
import numpy as np
from mini_sisso.model import MiniSisso

# 1. Prepare Data
np.random.seed(42) # Set seed for reproducibility
# Create feature data (X)
X_df = pd.DataFrame(np.random.rand(100, 2) *, columns=["feature_A", "feature_B"])
# Create target data (y) from a true equation: y = 2*sin(feature_A) + feature_B^2 + noise
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1)

# 2. Instantiate the Model
# You can set all hyperparameters. Comment out or use defaults for those you don't need.
model = MiniSisso(
    # --- Control the search space ---
    n_expansion=2,          # Depth of feature expansion (higher value finds more complex equations but takes longer)
    operators=["+", "sin", "pow2"], # List of operators for feature expansion
    
    # --- Control model complexity ---
    n_term=2,               # Max number of terms in the equation (for 'exhaustive' method)
    
    # --- Select the search strategy ---
    so_method="exhaustive", # Model search strategy ('exhaustive' or 'lasso')
    # alpha=0.01,           # Regularization parameter for so_method='lasso'
    
    # --- Control computational efficiency ---
    use_levelwise_sis=True, # Use staged feature pruning for speed (strongly recommended)
    k_per_level=50,         # If use_levelwise_sis=True, number of promising features to keep at each level
    k=10,                   # Number of feature candidates for each term in the final model
    
    # --- Select the execution environment ---
    # device="cuda",          # Specify 'cuda' to use GPU (requires PyTorch)
)

# 3. Fit the Model
# Uses the same fit(X, y) interface as scikit-learn
model.fit(X_df, y_series)

# 4. Check the Results
# Access fitted attributes (ending with an underscore)
print(f"Discovered Equation: {model.equation_}")
print(f"Training RMSE: {model.rmse_:.4f}")
print(f"Training R2 Score: {model.r2_:.4f}")

# 5. Make Predictions
# Uses the same predict(X) interface as scikit-learn
X_test_df = pd.DataFrame(np.array([, ]), columns=["feature_A", "feature_B"])
predictions = model.predict(X_test_df)
print(f"\nPredictions for new data: {predictions}")
```

**Example Output**:
```
Using NumPy/SciPy backend for CPU execution.
*** Starting Level-wise Recipe Generation (Level-wise SIS: ON, k_per_level=50) ***
... (training logs) ...
Best Model Found (2 terms):
  RMSE: 0.092124
  R2:   0.998806
  Equation: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610

Discovered Equation: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610
Training RMSE: 0.0921
Training R2 Score: 0.9988

Predictions for new data: [2.0016012 5.6796584]
```

## 🛠️ Usage Guide

### `use_levelwise_sis`: Toggling the Feature Generation Strategy

This parameter toggles the "Level-wise SIS" feature, which is key to the high performance of `mini-sisso`.

#### `True` (Default)
Performs feature expansion level by level, with a screening (SIS) step immediately after each level. Only promising features are used to generate the next level, significantly reducing computation time and memory usage. **This is the recommended setting.**

```python
# k_per_level controls how many features are kept at each level
model_fast = MiniSisso(use_levelwise_sis=True, k_per_level=100)
```

#### `False`
Generates all possible features (recipes) for all expansion levels at once before proceeding to the final SIS/SO step.
-   **Pros**: Explores a wider feature space, potentially finding unexpected feature combinations.
-   **Cons**: **Memory usage and computation time increase exponentially.** There is a high risk of `MemoryError` for larger `n_expansion` or a greater number of base features.

```python
# It is recommended to set n_expansion to a small value
model_full_search = MiniSisso(use_levelwise_sis=False, n_expansion=2)
```

### `so_method`: Selecting the Model Search Strategy

#### `exhaustive` (Default)
An **exhaustive search** that tests every possible combination of candidate features. It's more likely to find the optimal, interpretable model but can be slow. The number of terms is specified with `n_term`.

```python
# Exhaustively search for models up to 3 terms
model_exhaustive = MiniSisso(
    so_method="exhaustive", 
    n_term=3,
    operators=["+", "-", "*", "sqrt"]
)
```

#### `lasso`
Uses **Lasso regression** to quickly select important features from a large pool of candidates. It's extremely fast and effective for large search spaces. The regularization strength is controlled by `alpha`.

```python
# Use Lasso to select features quickly
# A smaller alpha tends to select more features
model_lasso = MiniSisso(
    so_method="lasso",
    alpha=0.01,
    operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "pow2", "pow3"]
)
```

### Available Operators

Specify as a list of strings in the `operators` argument.

| Operator | Description            |
| :------- | :--------------------- |
| `'+'`    | Addition (a + b)       |
| `'-'`    | Subtraction (a - b)    |
| `'*'`    | Multiplication (a * b) |
| `'/'`    | Division (a / b)       |
| `'sin'`  | Sine (sin(a))          |
| `'cos'`  | Cosine (cos(a))        |
| `'exp'`  | Exponential (e^a)      |
| `'log'`  | Natural Log (ln(a))    |
| `'sqrt'` | Square Root (sqrt(     | a | )) *Safe for negative inputs* |
| `'pow2'` | Square (a^2)           |
| `'pow3'` | Cube (a^3)             |
| `'inv'`  | Inverse (1/a)          |

## 🤝 `scikit-learn` Ecosystem Integration

As a fully compliant `scikit-learn` estimator, `mini-sisso` works seamlessly with the entire ecosystem.

### `Pipeline` for Preprocessing

```python
from sklearn.pipeline import Pipeline
from mini_sisso.model import MiniSisso

# Note: MiniSisso can be sensitive to feature scaling. 
# Preprocessing like StandardScaler is often not recommended.
pipeline = Pipeline([
    # ('scaler', StandardScaler()),
    ('sisso', MiniSisso(n_expansion=2, n_term=2, operators=["+", "sin", "pow2"]))
])

pipeline.fit(X_df, y_series)
predictions = pipeline.predict(X_df)
```

### `GridSearchCV` for Hyperparameter Tuning

Automatically find the best hyperparameters like `n_term`, `k` (number of SIS candidates), or `alpha`.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_term':,
    'k':,
}

grid_search = GridSearchCV(
    MiniSisso(operators=["+", "sin", "pow2"]),
    param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_df, y_series)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Model Equation: {grid_search.best_estimator_.equation_}")
```

## ⚙️ API Reference

### `MiniSisso`

```python
class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, n_term: int = 2, k: int = 10, 
                 k_per_level: int = 50, use_levelwise_sis: bool = True,
                 operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01,
                 device: str = "cpu"):
```

#### Parameters
-   `n_expansion` (int, default=2): The maximum level of feature expansion.
-   `n_term` (int, default=2): The maximum number of terms in the final model (for `exhaustive` search).
-   `k` (int, default=10): The number of promising features to select in each iteration of the SIS step.
-   `k_per_level` (int, default=50): If `use_levelwise_sis=True`, this is the number of promising recipes to carry over to the next expansion level.
-   `use_levelwise_sis` (bool, default=True): Toggles the level-wise SIS feature.
-   `device` (str, default="cpu"): The computing device to use ('cpu' or 'cuda').
-   `operators` (list[str], required): A list of operators to use for feature expansion.
-   `so_method` (str, default="exhaustive"): The model search strategy. Can be `"exhaustive"` or `"lasso"`.
-   `alpha` (float, default=0.01): The regularization parameter used when `so_method="lasso"`.

---

### `fit(X, y)`

Fits the model to the training data.

#### Parameters
-   `X` (array-like or pd.DataFrame): The feature data, shape `(n_samples, n_features)`.
-   `y` (array-like or pd.Series): The target variable data, shape `(n_samples,)`.

#### Returns
-   `self`: The fitted `MiniSisso` instance.

---

### `predict(X)`

Makes predictions using the fitted model.

#### Parameters
-   `X` (array-like or pd.DataFrame): The data to make predictions on.

#### Returns
-   `np.ndarray`: A NumPy array of the predictions.

---

### `score(X, y)`

Returns the coefficient of determination (R² score) of the prediction.

#### Parameters
-   `X` (array-like or pd.DataFrame): The feature data.
-   `y` (array-like or pd.Series): The true target variable data.

#### Returns
-   `float`: The R² score.

---

### Fitted Attributes

After calling `fit()`, you can access the following attributes:

-   `model.equation_` (str): The best mathematical model found.
-   `model.rmse_` (float): The RMSE of the best model on the training data.
-   `model.r2_` (float): The R2 score of the best model on the training data.
-   `model.coef_` (np.ndarray): The coefficients for each term in the best model.
-   `model.intercept_` (float): The intercept of the best model.

## 📜 License
This project is licensed under the MIT License.

## 🙏 Acknowledgements
This library was greatly inspired by the original SISSO algorithm paper and is built upon the fantastic open-source projects NumPy, SciPy, Pandas, scikit-learn, and PyTorch.