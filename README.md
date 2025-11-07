# mini-sisso

[![PyPI version](https://badge.fury.io/py/mini-sisso.svg)](https://pypi.org/project/mini-sisso)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/mini-sisso.svg)](https://pypi.org/project/mini-sisso/)

**`mini-sisso` is a lightweight and user-friendly Python implementation of the SISSO (Sure Independence Screening and Sparsifying Operator) symbolic regression algorithm. It offers full compatibility with the scikit-learn ecosystem for discovering interpretable mathematical models from data.**

### Why `mini-sisso` over the original SISSO?

`mini-sisso` provides the advanced search capabilities of the original C++/Fortran implementation in a more modern and accessible package.

-   **🧠 Memory Efficiency & Fast Exploration**:
    -   **"Recipe-based" Architecture**: Dramatically reduces memory consumption during feature expansion, allowing it to handle large-scale problems that would crash the original implementation.
    -   **"Level-wise SIS" Feature**: Significantly speeds up the search by pruning unpromising features early.
-   **🚀 Easy Adoption and Use**:
    -   Simple `pip install` with no complex compilation required.
    -   Intuitive `scikit-learn`-like `fit()` / `predict()` interface for easy model building and evaluation.
-   **🤝 Full `scikit-learn` Ecosystem Compatibility**:
    -   Seamlessly integrates with powerful tools like `GridSearchCV` for automated hyperparameter tuning and `Pipeline` for building workflows.
-   **⚡ Flexible Search Strategies & GPU Support**:
    -   In addition to the classic `exhaustive` search, it supports fast feature selectors like `Lasso` and `LightGBM`.
    -   Offers optional GPU acceleration for further speedups.

## 📥 Installation

### CPU Version (Default, Recommended)

Installs the CPU version from PyPI. Depends on NumPy, SciPy, scikit-learn, LightGBM, and dcor.

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
X_df = pd.DataFrame(np.random.rand(100, 2) *, columns=["feature_A", "feature_B"])
# True equation: y = 2*sin(feature_A) + feature_B^2 + noise
y_series = pd.Series(2 * np.sin(X_df["feature_A"]) + X_df["feature_B"]**2 + np.random.randn(100) * 0.1)

# 2. Instantiate the Model (with all hyperparameters shown)
# Comment out the ones you don't need to use their default values.
model = MiniSisso(
    # --- Control the fundamental search space ---
    n_expansion=2,                      # Depth of feature expansion (deeper finds more complex eqns but takes longer)
    operators=["+", "sin", "pow2"],     # List of operators for feature expansion
    
    # --- Select the main search strategy ---
    so_method="exhaustive",             # Model search strategy ('exhaustive', 'lasso', 'lightgbm')
    
    # --- Detailed settings for each strategy (selection_params) ---
    selection_params={
        # -- Parameters for "exhaustive" method --
        'n_term': 2,                    # Maximum number of terms in the discovered equation
        'n_sis_features': 10,           # Number of SIS candidates for each term
        
        # -- Parameters for "lasso" method --
        # 'alpha': 0.01,                # Regularization strength for Lasso
        
        # -- Parameters for "lightgbm" method --
        # 'n_features_to_select': 20,   # Number of features to select with LightGBM
        # 'lightgbm_params': {'n_estimators': 100, 'random_state': 42}, # Parameters for the LightGBM model itself
        
        # -- Preprocessing filters for "lasso"/"lightgbm" (optional) --
        # 'n_global_sis_features': 200, # Number of candidates to pre-screen based on correlation with target
        # 'collinearity_filter': 'mi',  # Method to calculate correlation between candidates ('mi' or 'dcor')
        # 'collinearity_threshold': 0.9, # Correlation threshold for the above filter
    },
    
    # --- Control computational efficiency ---
    use_levelwise_sis=True,             # Speed up with staged search (strongly recommended)
    n_level_sis_features=50,            # Number of promising features to keep at each expansion level
    
    # --- Select the execution environment ---
    # device="cuda",                      # Specify 'cuda' to use GPU
)

# 3. Fit the model
# Uses the same fit(X, y) interface as scikit-learn
model.fit(X_df, y_series)

# 4. Check the results
# Access fitted attributes (ending with an underscore)
print("\n--- Fit Results ---")
print(f"Discovered Equation: {model.equation_}")
print(f"Training RMSE: {model.rmse_:.4f}")
print(f"Training R2 Score: {model.r2_:.4f}")

# 5. Make predictions
# Uses the same predict(X) interface as scikit-learn
print("\n--- Predictions ---")
X_test_df = pd.DataFrame(np.array([, ]), columns=["feature_A", "feature_B"])
predictions = model.predict(X_test_df)
print(f"Predictions for new data ([0.5, 1.0], [1.0, 2.0]): {predictions}")
```

**Example Output**:
```
Using NumPy/SciPy backend for CPU execution.
*** Starting Level-wise Recipe Generation (Level-wise SIS: ON, k_per_level=50) ***
Level 1: Generated 5, selected top 5. Total promising: 7. Time: 0.00s
Level 2: Generated 30, selected top 30. Total promising: 37. Time: 0.00s
***************** Starting SISSO Regressor (NumPy/SciPy Backend, Method: exhaustive) *****************

===== Searching for 1-term models =====
...
===== Searching for 2-term models =====
...
Best 2-term model: RMSE=0.092124, Eq: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610
Time: 0.01 seconds

==================================================
SISSO fitting finished. Total time: 0.02s
==================================================

Best Model Found (2 terms):
  RMSE: 0.092124
  R2:   0.998806
  Equation: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610

--- Fit Results ---
Discovered Equation: +0.998492 * ^2(feature_B) +1.971237 * sin(feature_A) +0.030610
Training RMSE: 0.0921
Training R2 Score: 0.9988

--- Predictions ---
Predictions for new data ([0.5, 1.0], [1.0, 2.0]): [2.0016012 5.6796584]
```

## 🛠️ Usage Guide: Controlling the Search with Hyperparameters

The `mini-sisso` search process follows this workflow, with each step controlled by hyperparameters.

### Workflow Overview

1.  **Feature Expansion**: Generates candidate features based on `operators` and `n_expansion`.
    -   This process is made efficient by `use_levelwise_sis=True` and `n_level_sis_features`.
2.  **[Optional] Preprocessing Filters**: A set of filters to prune candidate features when using `lasso` or `lightgbm`. (Configured in `selection_params`).
    -   **Global SIS**: Removes features with low correlation to the target `y`.
    -   **Collinearity Filter**: Removes highly correlated features from each other.
3.  **Model Search (Sparsifying Operator)**: The final model is discovered from the pruned candidates using the strategy specified by `so_method`.

---
### Main Hyperparameters

#### `so_method`: The Three Model Search Strategies

The `so_method` parameter determines the core search approach.

##### 1. `so_method="exhaustive"` (Default)
The classic SISSO approach. It uses iterative SIS and **exhaustive search** to find the optimal model. Best for finding simple, interpretable models.

```python
# Exhaustively search for models up to 3 terms
model = MiniSisso(
    so_method="exhaustive",
    selection_params={
        'n_term': 3,          # Max number of terms to search for
        'n_sis_features': 15  # Number of candidates to add to the pool at each SIS step
    }
)
```

##### 2. `so_method="lasso"`
Uses **Lasso regression** as a feature selector to build a model quickly. Effective for large feature spaces.

```python
# Select features using Lasso
model = MiniSisso(
    so_method="lasso",
    selection_params={
        'alpha': 0.01 # Regularization parameter for Lasso
    }
)
```

##### 3. `so_method="lightgbm"`
Uses **LightGBM** as a feature selector. Excels at capturing non-linear relationships.

```python
# Select top 20 features using LightGBM
model = MiniSisso(
    so_method="lightgbm",
    selection_params={
        'n_features_to_select': 20
    }
)
```

---
#### `selection_params`: Detailed Control for Each Strategy

The `selection_params` dictionary allows you to apply preprocessing filters and fine-tune each `so_method`.

##### Preprocessing Filters (for `lasso`/`lightgbm`)

-   **`n_global_sis_features`**: Pre-screens candidates by removing those with low correlation to the target `y`.
-   **`collinearity_filter`**: Removes highly correlated features to stabilize Lasso/LightGBM. Can be `'mi'` (Mutual Information) or `'dcor'` (Distance Correlation).

```python
# Filter candidates with Global SIS and MI before running LightGBM
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

##### Expert Settings (for `lightgbm`)
You can also pass internal hyperparameters directly to the LightGBM model.
```python
model = MiniSisso(
    so_method='lightgbm',
    selection_params={
        'n_features_to_select': 20,
        'lightgbm_params': {
            'n_estimators': 200,         # Number of trees
            'num_leaves': 40,            # Max number of leaves in one tree
            'learning_rate': 0.05,       # Learning rate
            'colsample_bytree': 0.8,     # Fraction of features to be considered for each tree
            'subsample': 0.8,            # Fraction of data to be used for each tree
            'reg_alpha': 0.1,            # L1 regularization
            'reg_lambda': 0.1,           # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
    }
)
```

---
#### Other Key Parameters

-   `use_levelwise_sis` (bool, default=True): **Strongly recommended.** Speeds up feature generation and saves memory.
-   `n_level_sis_features` (int, default=50): Number of features to keep at each stage when `use_levelwise_sis=True`.
-   `device` (str, default="cpu"): Set to `"cuda"` to use the GPU backend.

### Available Operators
Specify the `operators` argument as a list of strings.

| Operator | Description               |
| :------- | :------------------------ |
| `'+'`    | Addition (a + b)          |
| `'-'`    | Subtraction (a - b)       |
| `'*'`    | Multiplication (a * b)    |
| `'/'`    | Division (a / b)          |
| `'sin'`  | Sine (sin(a))             |
| `'cos'`  | Cosine (cos(a))           |
| `'exp'`  | Exponential (e^a)         |
| `'log'`  | Natural logarithm (ln(a)) |
| `'sqrt'` | Square root (sqrt(        | a | )) *No error for negative values* |
| `'pow2'` | Square (a^2)              |
| `'pow3'` | Cube (a^3)                |
| `'inv'`  | Reciprocal (1/a)          |

## 🤝 `scikit-learn` Ecosystem Integration
`mini-sisso` inherits `BaseEstimator` and `RegressorMixin` from `scikit-learn`, allowing it to seamlessly integrate with the powerful tools provided by `scikit-learn`.

### More detailed usage of `Pipeline`

`Pipeline` is a tool for connecting multiple processing steps and treating them as a single estimator.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mini_sisso.model import MiniSisso

# Pipeline definition
# Note: MiniSisso is sensitive to the scale of input features, so preprocessing such as StandardScaler
# may impair the interpretability of the discovered formula. It is generally not recommended.
# Here is an example to demonstrate how Pipeline technically works.
pipeline = Pipeline([
# Step 1: Run standardization using the name 'scaler'
('scaler', StandardScaler()), # Usually unnecessary/not recommended for MiniSisso
# Step 2: Run MiniSisso using the name 'sisso'
('sisso', MiniSisso(n_expansion=2, selection_params={'n_term': 2}, operators=["+", "sin", "pow2"]))
])

# Train the entire pipeline: X -> scaler.fit_transform -> sisso.fit
pipeline.fit(X_df, y_series)

# Predict using the pipeline: X -> scaler.transform -> sisso.predict
predictions = pipeline.predict(X_df)

# You can also access and change parameters for each step of the pipeline.
# Example: Changing the number of SISSO terms after training
# pipeline.set_params(sisso__selection_params={'n_term': 3})
print(f"Number of terms in the SISSO step of the pipeline: {pipeline.named_steps['sisso'].selection_params['n_term']}")
```

### Advanced `GridSearchCV` Usage

`GridSearchCV` can automatically find the best combination of hyperparameters, including the `so_method` itself. The `__` (double underscore) syntax allows you to search nested parameters within `selection_params`.

```python
from sklearn.model_selection import GridSearchCV

# Define a list of parameter grids to search over
param_grid = [
    # Case 1: Search patterns for exhaustive method
    {
        'so_method': ['exhaustive'],
        'selection_params': [
            {'n_term': 2, 'n_sis_features': 10},
            {'n_term': 3, 'n_sis_features': 15}
        ]
    },
    # Case 2: Search patterns for lasso method
    {
        'so_method': ['lasso'],
        'selection_params': [
            {'alpha': 0.01, 'collinearity_filter': 'mi'},
            {'alpha': 0.005}
        ]
    },
    # Case 3: Search patterns for lightgbm method
    {
        'so_method': ['lightgbm'],
        'selection_params__n_features_to_select':,
        'selection_params__lightgbm_params__n_estimators':,
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

## ⚙️ API Reference

### `MiniSisso`
```python
class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, operators: list = None,
                 so_method: str = "exhaustive", selection_params: dict = None,
                 use_levelwise_sis: bool = True, n_level_sis_features: int = 50,
                 device: str = "cpu"):
```

### `MiniSisso`
-   `n_expansion` (int, default=2): Max level of feature expansion.
-   `operators` (list[str], required): List of operators for feature generation.
-   `so_method` (str, default="exhaustive"): Model search strategy (`"exhaustive"`, `"lasso"`, `"lightgbm"`).
-   `selection_params` (dict, optional): Dictionary of detailed parameters for the selected `so_method` and preprocessing filters.
-   `use_levelwise_sis` (bool, default=True): Toggles the level-wise SIS feature.
-   `n_level_sis_features` (int, default=50): Number of features to keep at each level if `use_levelwise_sis=True`.
-   `device` (str, default="cpu"): Computation device (`"cpu"` or `"cuda"`).

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