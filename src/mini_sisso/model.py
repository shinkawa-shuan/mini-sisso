# model.py (Final fix for import statements)
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .feature_generator import FeatureGenerator
from .recipe import OPERATORS, FeatureRecipe

# from .executor import RecipeExecutor # ★★★ この行を削除 ★★★

# --- Dynamic Import Setup ---
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, n_term: int = 2, k: int = 10, k_per_level: int = 50, use_levelwise_sis: bool = True, operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01, device: str = "cpu"):

        self.n_expansion = n_expansion
        self.n_term = n_term
        self.k = k
        self.k_per_level = k_per_level
        self.use_levelwise_sis = use_levelwise_sis
        self.operators = operators
        self.so_method = so_method
        self.alpha = alpha
        self.device = device

        if self.device == "cuda" and not TORCH_AVAILABLE:
            raise ImportError("GPU support requires PyTorch. Please install with 'pip install \"mini-sisso[gpu]\"'")

        # Fitted attributes (will be defined in fit())
        self.coef_ = None
        self.intercept_ = None
        self.best_model_recipes_ = None
        self.base_feature_names_ = None
        self.equation_ = ""
        self.rmse_ = float("inf")
        self.r2_ = -float("inf")

    def fit(self, X, y):
        start_time = time.time()

        X_arr, y_arr = np.asarray(X), np.asarray(y)

        self.base_feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X_arr.shape[1])]
        FeatureRecipe.base_feature_names = self.base_feature_names_
        operators_dict = {op: OPERATORS[op] for op in self.operators + ["base"] if op in OPERATORS} if self.operators else OPERATORS

        # --- Backend Switching ---
        if self.device == "cuda":
            print("Using PyTorch backend for GPU acceleration.")
            from .executor_torch import RecipeExecutorTorch
            from .regressor_torch import SissoRegressorTorch

            X_fit = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            y_fit = torch.tensor(y_arr, dtype=torch.float32).to(self.device)

            ExecutorClass = RecipeExecutorTorch
            RegressorClass = SissoRegressorTorch
        else:
            print("Using NumPy/SciPy backend for CPU execution.")
            from .executor_numpy import RecipeExecutor as RecipeExecutorNumPy
            from .regressor_numpy import SissoRegressorNumPy

            X_fit = X_arr
            y_fit = y_arr

            ExecutorClass = RecipeExecutorNumPy
            RegressorClass = SissoRegressorNumPy

        executor = ExecutorClass(X_fit)
        generator = FeatureGenerator(self.base_feature_names_, operators_dict)

        if self.use_levelwise_sis:
            recipes = generator.expand_with_levelwise_sis(self.n_expansion, self.k_per_level, executor, y_fit)
        else:
            recipes = generator.expand_full(self.n_expansion)

        regressor = RegressorClass(all_recipes=recipes, executor=executor, y=y_fit, n_term=self.n_term, k=self.k, so_method=self.so_method, alpha=self.alpha)

        result = regressor.fit()

        print(f"\n{'='*50}\nSISSO fitting finished. Total time: {time.time() - start_time:.2f}s\n{'='*50}")

        if result:
            rmse, eq, r2, all_models = result
            if all_models:
                best_model = min(all_models.values(), key=lambda m: m["rmse"])
                self.best_model_recipes_ = best_model["recipes"]
                self.coef_ = best_model["coeffs"].cpu().numpy() if TORCH_AVAILABLE and isinstance(best_model["coeffs"], torch.Tensor) else np.asarray(best_model["coeffs"])
                self.intercept_ = best_model["intercept"].item() if TORCH_AVAILABLE and isinstance(best_model["intercept"], torch.Tensor) else float(best_model["intercept"])
                self.equation_ = eq
                self.rmse_ = best_model["rmse"]
                self.r2_ = r2
                print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):\n  RMSE: {self.rmse_:.6f}\n  R2:   {self.r2_:.6f}\n  Equation: {self.equation_}")

        if not hasattr(self, "coef_") or self.coef_ is None:
            print("\nCould not find a valid model.")
            self.coef_ = np.array([])
            self.intercept_ = 0.0

        FeatureRecipe.base_feature_names = []
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_recipes_") or self.best_model_recipes_ is None:
            raise RuntimeError("Model has not been fitted yet or no valid model was found.")

        X_arr = np.asarray(X)
        FeatureRecipe.base_feature_names = self.base_feature_names_

        from .executor_numpy import RecipeExecutor as RecipeExecutorNumPy

        pred_executor = RecipeExecutorNumPy(X_arr)

        y_pred = np.full(X_arr.shape[0], self.intercept_)
        for i, recipe in enumerate(self.best_model_recipes_):
            feature_vals = pred_executor.execute(recipe)
            y_pred += self.coef_.flatten()[i] * np.nan_to_num(feature_vals)

        FeatureRecipe.base_feature_names = []
        return y_pred
