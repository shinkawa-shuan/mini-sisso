# model.py
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .feature_generator import FeatureGenerator
from .recipe import OPERATORS, FeatureRecipe

try:
    from . import _mini_sisso_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust backend not found. Compile with maturin.")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, operators: list = None, so_method: str = "exhaustive", selection_params: dict = None, use_levelwise_sis: bool = True, n_level_sis_features: int = 50, device: str = "cpu"):

        self.n_expansion, self.operators, self.so_method = n_expansion, operators, so_method
        self.selection_params = selection_params
        self.use_levelwise_sis, self.n_level_sis_features = use_levelwise_sis, n_level_sis_features
        self.device = device

        if self.device == "cuda" and not TORCH_AVAILABLE:
            raise ImportError("GPU support requires PyTorch. Please install with 'pip install \"mini-sisso[gpu]\"'")

    def fit(self, X, y):
        start_time = time.time()
        X_arr, y_arr = np.asarray(X), np.asarray(y)
        self.base_feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X_arr.shape[1])]
        FeatureRecipe.base_feature_names = self.base_feature_names_
        operators_dict = {op: OPERATORS[op] for op in self.operators + ["base"] if op in OPERATORS} if self.operators else OPERATORS

        if self.so_method == 'exhaustive' and RUST_AVAILABLE:
            print("Using Rust backend for exhaustive search...")
            try:
                rmse, eq, coeffs, intercept, feature_structures = _mini_sisso_rs.rust_fit_exhaustive(
                    X_arr, y_arr, self.base_feature_names_, 
                    self.operators or list(operators_dict.keys()), # Pass operators list
                    self.n_expansion, self.selection_params['n_term'], 
                    self.selection_params['n_sis_features']
                )
                self.rmse_ = rmse
                self.equation_ = eq
                self.coef_ = np.array(coeffs)
                self.intercept_ = intercept
                
                # Calculate R2
                y_var = np.var(y_arr)
                if y_var > 1e-12:
                     self.r2_ = 1.0 - (self.rmse_ ** 2) / y_var
                else:
                     self.r2_ = 0.0
                
                self.best_model_recipes_ = [
                    FeatureRecipe.from_structure(struct, operators_dict) 
                    for struct in feature_structures
                ]
                
                print(f"\nBest Model Found (Rust):\n  RMSE: {self.rmse_:.6f}\n  Equation: {self.equation_}")
                result = None 
                
            except Exception as e:
                print(f"Rust execution failed: {e}. Falling back to Python.")
                # Fallback code...
                params = self.selection_params
                # ... (rest of python code)
                # To avoid duplicating code, we can just let it fall through if we didn't set result?
                # But we need to skip the python execution if rust succeeded.
                
        # If Rust succeeded, we are done.
        if hasattr(self, "rmse_"):
             return self

        if self.device == "cuda":
            # ... (GPU backend switching logic)
            pass
        else:  # CPU
            from .executor_numpy import RecipeExecutor
            from .regressor_numpy import SissoRegressorNumPy

            X_fit, y_fit, ExecutorClass, RegressorClass = X_arr, y_arr, RecipeExecutor, SissoRegressorNumPy

        executor = ExecutorClass(X_fit)
        generator = FeatureGenerator(self.base_feature_names_, operators_dict)

        if self.use_levelwise_sis:
            recipes = generator.expand_with_levelwise_sis(self.n_expansion, self.n_level_sis_features, executor, y_fit)
        else:
            recipes = generator.expand_full(self.n_expansion)

        regressor = RegressorClass(recipes, executor, y_fit, self.so_method, self.selection_params)
        result = regressor.fit()

        print(f"\n{'='*50}\nSISSO fitting finished. Total time: {time.time() - start_time:.2f}s\n{'='*50}")

        if result:
            rmse, eq, r2, all_models = result
            if all_models:
                best_model = min(all_models.values(), key=lambda m: m["rmse"])
                self.best_model_recipes_, self.coef_, self.intercept_ = best_model["recipes"], np.asarray(best_model["coeffs"]), float(best_model["intercept"])
                self.equation_, self.rmse_, self.r2_ = eq, best_model["rmse"], r2
                print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):\n  RMSE: {self.rmse_:.6f}\n  R2:   {self.r2_:.6f}\n  Equation: {self.equation_}")

        if not hasattr(self, "coef_") or self.coef_ is None:
            print("\nCould not find a valid model.")
            self.coef_, self.intercept_ = np.array([]), 0.0

        FeatureRecipe.base_feature_names = []
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_recipes_") or self.best_model_recipes_ is None:
            raise RuntimeError("Model not fitted.")
        from .executor_numpy import RecipeExecutor

        X_arr = np.asarray(X)
        FeatureRecipe.base_feature_names = self.base_feature_names_
        pred_executor = RecipeExecutor(X_arr)
        y_pred = np.full(X_arr.shape[0], self.intercept_)
        for i, recipe in enumerate(self.best_model_recipes_):
            y_pred += self.coef_.flatten()[i] * np.nan_to_num(pred_executor.execute(recipe))
        FeatureRecipe.base_feature_names = []
        return y_pred
