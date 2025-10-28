# model.py (scikit-learn compatible NumPy/SciPy version)
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .executor import RecipeExecutor
from .feature_generator import FeatureGenerator
from .recipe import OPERATORS, FeatureRecipe
from .regressor import SissoRegressor


class MiniSisso(BaseEstimator, RegressorMixin):
    def __init__(self, n_expansion: int = 2, n_term: int = 2, k: int = 10, k_per_level: int = 50, use_levelwise_sis: bool = True, operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01):

        self.n_expansion = n_expansion
        self.n_term = n_term
        self.k = k
        self.k_per_level = k_per_level
        self.use_levelwise_sis = use_levelwise_sis
        self.operators = operators
        self.so_method = so_method
        self.alpha = alpha

    def fit(self, X, y):
        start_time = time.time()

        # 入力データをNumPy配列に変換
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if isinstance(X, pd.DataFrame):
            self.base_feature_names_ = X.columns.tolist()
        else:
            self.base_feature_names_ = [f"f{i}" for i in range(X.shape[1])]

        FeatureRecipe.base_feature_names = self.base_feature_names_

        operators_dict = {op: OPERATORS[op] for op in self.operators + ["base"] if op in OPERATORS} if self.operators else OPERATORS

        executor = RecipeExecutor(X_arr)
        generator = FeatureGenerator(self.base_feature_names_, operators_dict)

        if self.use_levelwise_sis:
            recipes = generator.expand_with_levelwise_sis(self.n_expansion, self.k_per_level, executor, y_arr)
        else:
            recipes = generator.expand_full(self.n_expansion)

        regressor = SissoRegressor(recipes, executor, y_arr, self.n_term, self.k, self.so_method, self.alpha)
        result = regressor.fit()

        print(f"\n{'='*50}\nSISSO fitting finished. Total time: {time.time() - start_time:.2f}s\n{'='*50}")

        if result:
            rmse, eq, r2, all_models = result
            if all_models:
                best_model = min(all_models.values(), key=lambda m: m["rmse"])
                self.best_model_recipes_ = best_model["recipes"]
                self.coef_ = best_model["coeffs"]
                self.intercept_ = best_model["intercept"]
                self.equation_ = eq
                self.rmse_ = best_model["rmse"]
                self.r2_ = r2
                print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):\n  RMSE: {self.rmse_:.6f}\n  R2:   {self.r2_:.6f}\n  Equation: {self.equation_}")

        if not hasattr(self, "coef_"):
            print("\nCould not find a valid model.")
            # 学習が失敗してもエラーにならないように、ダミーの値を設定
            self.coef_ = np.array([])
            self.intercept_ = 0.0

        FeatureRecipe.base_feature_names = []
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_recipes_") or self.best_model_recipes_ is None:
            raise RuntimeError("Model has not been fitted yet or no valid model was found.")

        X_arr = np.asarray(X)
        FeatureRecipe.base_feature_names = self.base_feature_names_
        pred_executor = RecipeExecutor(X_arr)

        y_pred = np.full(X_arr.shape[0], self.intercept_)
        for i, recipe in enumerate(self.best_model_recipes_):
            feature_vals = pred_executor.execute(recipe)
            y_pred += self.coef_.flatten()[i] * np.nan_to_num(feature_vals)

        FeatureRecipe.base_feature_names = []
        return y_pred
