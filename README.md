# mini-sisso

[![PyPI version](https://badge.fury.io/py/mini-sisso.svg)](https://pypi.org/project/mini-sisso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/mini-sisso.svg)](https://pypi.org/project/mini-sisso/)

**`mini-sisso` is a lightweight and user-friendly Python implementation of the SISSO (Sure Independence Screening and Sparsifying Operator) symbolic regression algorithm. It offers full compatibility with the scikit-learn ecosystem for discovering interpretable mathematical models from data.**

Inheriting the advanced exploration capabilities of the original C++/Fortran-based implementation, `mini-sisso` provides the following features:

-   **🚀 Easy Adoption and Use**:
    -   Simple `pip install`. Minimal dependencies (no heavy libraries like PyTorch) ensure easy setup and CPU-friendly operation.
    -   `scikit-learn`-like `fit()` / `predict()` interface for intuitive model building and evaluation.
-   **🧠 Memory Efficiency & Fast Exploration**:
    -   A "recipe-based" architecture dramatically reduces memory consumption during Feature Expansion.
    -   The "Level-wise SIS" feature (toggleable) speeds up exploration by pruning unpromising features early.
    -   `NumPy`/`SciPy`-based computation engine ensures numerical stability and efficiency.
-   **🤝 Full `scikit-learn` Ecosystem Compatibility**:
    -   Seamlessly integrates with powerful `scikit-learn` tools like `GridSearchCV` for hyperparameter tuning and `Pipeline` for preprocessing.

This library is valuable in any field where interpretable mathematical equations are desired to uncover underlying mechanisms in data, such as discovering physical laws, predicting material properties, or financial modeling.

## 📥 Installation

### Install directly from GitHub (Latest Version)

To install the latest development version, use `pip` to install directly from this GitHub repository.

```bash
pip install git+https://github.com/shinkawa-shuan/mini-sisso.git
```

To update the library to the newest version, use the `--upgrade` option.
```bash
pip install --upgrade git+https://github.com/shinkawa-shuan/mini-sisso.git
```

### Install from PyPI (Stable Version)

**(Note: Not yet registered on PyPI. The following command will be available in the future.)**

To install the stable version from PyPI, execute the following command.

```bash
pip install mini-sisso
```

## 🚀 Quick Start

You can train a mathematical model from data and make predictions with just a few lines of code.

```python
import pandas as pd
import numpy as np
from mini_sisso.model import MiniSisso # Note package and class name

# 1. Prepare benchmark data (X, y split for scikit-learn fit/predict)
np.random.seed(42)
X_df = pd.DataFrame(np.random.rand(100, 2) * [2, 3], columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1, name="target_value")

# 2. Create a MiniSisso model instance
model = MiniSisso(
    n_expansion=2,          # Level of feature expansion
    n_term=2,               # Maximum number of terms in the model
    operators=["+", "sin", "pow2"], # Operators to use
    use_levelwise_sis=True, # Enable memory-efficient level-wise SIS (default)
)

# 3. Fit the model
model.fit(X_df, y_series) # Pass X and y separately

# 4. Check the results
print(f"Discovered Equation: {model.equation_}")
print(f"Training RMSE: {model.rmse_:.4f}")
print(f"Training R2 Score: {model.r2_:.4f}")

# 5. Predict on new data (Pass X only for scikit-learn predict)
X_test_df = pd.DataFrame(np.array([[0.5, 1.0], [1.0, 2.0]]), columns=["feature_A", "feature_B"])
predictions = model.predict(X_test_df)
print(f"\nPredictions: {predictions}")
```

**Example Output**:
```
... (training logs) ...
Best Model Found (2 terms):
  RMSE: 0.088097
  R2:   0.998927
  Equation: +0.997666 * ^2(feature_B) +2.014116 * sin(feature_A) +0.001270

Discovered Equation: +0.997666 * ^2(feature_B) +2.014116 * sin(feature_A) +0.001270
Training RMSE: 0.0881
Training R2 Score: 0.9989

Predictions: [3.0481014 5.6796246]
```

## 🛠️ Usage Guide

### `use_levelwise_sis`: Toggling the Feature Generation Strategy

This parameter toggles the "Level-wise SIS" feature, which is key to the high performance of `mini-sisso`.

#### `True` (Default)
Performs feature expansion level by level, with a screening (SIS) step immediately after each level. Only promising features are used to generate the next level, significantly reducing computation time and memory usage. **This is the recommended setting.**

```python
model = MiniSisso(
    use_levelwise_sis=True, # Default, can be omitted
    k_per_level=100,        # Keep the top 100 promising features at each level
    # ... other parameters
)
```

#### `False`
Generates all possible features (recipes) for all expansion levels at once before proceeding to the final SIS/SO step.
-   **Pros**: Explores a wider feature space, potentially finding unexpected feature combinations.
-   **Cons**: **Memory usage and computation time increase exponentially.** There is a high risk of `MemoryError` for larger `n_expansion` or a greater number of base features.

```python
model = MiniSisso(
    use_levelwise_sis=False,
    n_expansion=2, # Recommended to keep this value small to conserve memory
    # ... other parameters
)
```

### `so_method`: Selecting the Model Search Strategy

`MiniSisso` provides two search strategies for the Sparsifying Operator (SO).

#### 1. `exhaustive` (Default)
An exhaustive search that tests every possible combination of candidate features.
-   **Pros**: Most likely to find the optimal combination. Tends to find simpler, more interpretable models.
-   **Cons**: Computation time grows combinatorially with the number of candidate features and terms.

```python
model = MiniSisso(
    n_term=3,
    so_method="exhaustive", # Default, can be omitted
    operators=["+", "-", "*", "sqrt"]
)
```

#### 2. `lasso`
Uses Lasso regression to quickly select important features from a large pool of candidates.
-   **Pros**: Extremely fast. Can find effective models in large search spaces where `exhaustive` is not feasible.
-   **Cons**: Requires tuning the `alpha` parameter. The resulting models can be more complex and are not guaranteed to be optimal.

```python
model = MiniSisso(
    so_method="lasso",
    alpha=0.01, # Regularization parameter for Lasso. Smaller values select more features.
    operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "pow2", "pow3"]
)
model.fit(X_df, y_series) # Pass X and y separately
```
**Tuning `alpha`**: `alpha` requires some trial and error. If the log shows `LASSO selected 0 features.`, decrease `alpha`. If it selects too many features, increase `alpha`.

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

`mini-sisso` inherits from `scikit-learn`'s `BaseEstimator` and `RegressorMixin`, allowing seamless integration with powerful `scikit-learn` tools.

### `Pipeline` for Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Included for demonstration, often not recommended for MiniSisso
from mini_sisso.model import MiniSisso # Note package and class name
import pandas as pd
import numpy as np

# Data preparation
X_df = pd.DataFrame(np.random.rand(100, 2) * [2, 3], columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1, name="target_value")

# Define the Pipeline
# Note: MiniSisso can be sensitive to feature scaling. StandardScaler might not always be beneficial.
pipeline = Pipeline([
    # ('scaler', StandardScaler()), # Uncomment if scaling is desired, but evaluate impact on interpretability/performance
    ('sisso', MiniSisso(n_expansion=2, n_term=2, operators=["+", "sin", "pow2"]))
])

# Fit the entire pipeline
pipeline.fit(X_df, y_series)

# Make predictions
predictions = pipeline.predict(X_df)
print(f"Predictions from Pipeline (partial): {predictions[:5]}")
```

### `GridSearchCV` for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from mini_sisso.model import MiniSisso # Note package and class name
import pandas as pd
import numpy as np

# Data preparation (same as above)
X_df = pd.DataFrame(np.random.rand(100, 2) * [2, 3], columns=["feature_A", "feature_B"])
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1, name="target_value")

# Define the grid of hyperparameters to tune
param_grid = {
    'n_expansion': [1, 2],
    'n_term': [1, 2],
    'k': [10, 20],
    'use_levelwise_sis': [True], # Often fixed to True for performance
    # 'alpha': [0.001, 0.01, 0.1] # For 'lasso' method
}

# Create a GridSearchCV instance
# scoring='neg_root_mean_squared_error' to evaluate RMSE
grid_search = GridSearchCV(
    MiniSisso(operators=["+", "sin", "pow2"], so_method="exhaustive"), # Common MiniSisso parameters
    param_grid,
    cv=3, # 3-fold cross-validation
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, # Use all available CPU cores
    verbose=1, # Output detailed logs
)

# Execute hyperparameter search
grid_search.fit(X_df, y_series)

print(f"\nOptimal Hyperparameters: {grid_search.best_params_}")
print(f"RMSE of the Best Model (Cross-validated): {-grid_search.best_score_:.4f}")
print(f"Equation of the Best Model: {grid_search.best_estimator_.equation_}")
```

## ⚙️ API Reference

### `MiniSisso`

```python
class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, n_term: int = 2, k: int = 10, 
                 k_per_level: int = 50, use_levelwise_sis: bool = True,
                 operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01):
```

#### Parameters
-   `n_expansion` (int, default=2): The maximum level of feature expansion.
-   `n_term` (int, default=2): The maximum number of terms in the final model (for `exhaustive` search).
-   `k` (int, default=10): The number of promising features to select in each iteration of the SIS step.
-   `k_per_level` (int, default=50): If `use_levelwise_sis=True`, this is the number of promising recipes to carry over to the next expansion level.
-   `use_levelwise_sis` (bool, default=True): Toggles the level-wise SIS feature.
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
-   `X` (array-like or pd.DataFrame): The data to make predictions on, shape `(n_samples, n_features)`. **It is recommended to have the same feature column names and order** as the training data (`pd.DataFrame` case).

#### Returns
-   `np.ndarray`: A NumPy array of the predictions.

---

### `score(X, y)`

Evaluates the model's performance according to `scikit-learn` conventions. By default, it returns the coefficient of determination (R² score).

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
-   `model.best_model_recipes_` (tuple): A tuple of `FeatureRecipe` objects that make up the best model.
-   `model.coef_` (np.ndarray): The coefficients for each term in the best model.
-   `model.intercept_` (float): The intercept of the best model.
-   `model.base_feature_names_` (list[str]): A list of feature names used during training.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgements

This library was greatly inspired by the original SISSO algorithm paper. It is also built upon the fantastic open-source projects NumPy, SciPy, Pandas, and scikit-learn.