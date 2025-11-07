# regressor_numpy.py
import time
from itertools import combinations
from typing import Dict, List, Tuple

import dcor
import lightgbm as lgb
import numpy as np
from scipy.linalg import lstsq as sp_lstsq
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso

from .executor_numpy import RecipeExecutor
from .recipe import FeatureRecipe


class SissoRegressorNumPy:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutor, y: np.ndarray, so_method: str, selection_params: dict):
        self.all_recipes = all_recipes
        self.executor = executor
        self.y = y
        self.so_method = so_method
        self.selection_params = self._parse_selection_params(selection_params)
        self.y_mean, self.y_centered = np.mean(y), y - np.mean(y)
        self.best_models: Dict[int, dict] = {}

    def _parse_selection_params(self, params: dict) -> dict:
        """ユーザーからのパラメータを解析し、デフォルト値を設定する"""
        if params is None:
            params = {}
        # exhaustive
        params.setdefault("n_term", 2)
        params.setdefault("n_sis_features", 10)
        # lasso
        params.setdefault("alpha", 0.01)
        # lightgbm
        params.setdefault("n_features_to_select", 40)
        params.setdefault("lightgbm_params", {"random_state": 42, "n_jobs": -1, "verbosity": -1})
        # filters
        params.setdefault("n_global_sis_features", None)
        params.setdefault("collinearity_filter", None)
        params.setdefault("collinearity_threshold", 0.9)
        params.setdefault("coeff_threshold", 1e-5)
        return params

    def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: np.ndarray, intercept: float) -> str:
        if not recipes:
            return f"{intercept:+.6f}"
        return "".join(f"{c:+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten())) + f"{intercept:+.6f}"

    def _run_sis(self, target: np.ndarray, recipes: List[FeatureRecipe], k: int) -> List[FeatureRecipe]:
        if not recipes:
            return []
        scores = []
        for r in recipes:
            arr = self.executor.execute(r)
            valid = ~np.isnan(arr) & ~np.isnan(target)
            if valid.sum() < 2:
                scores.append(0.0)
                continue
            vf, vt = arr[valid], target[valid]
            mean, std = np.mean(vf), np.std(vf)
            if std > 1e-8:
                scores.append(np.abs(np.dot(vt - np.mean(vt), (vf - mean) / std)))
            else:
                scores.append(0.0)
        return [recipes[i] for i in np.argsort(scores)[::-1][:k]]

    def _filter_collinear_features(self, recipes: List[FeatureRecipe], method: str, threshold: float) -> List[FeatureRecipe]:
        print(f"--- Applying Collinearity Filter (method: {method}, threshold: {threshold}) ---")
        X = np.stack([self.executor.execute(r) for r in recipes], axis=1)
        # NaNを平均値で補完
        if np.isnan(X).any():
            X[np.where(np.isnan(X))] = np.take(np.nanmean(X, axis=0), np.where(np.isnan(X))[1])

        removed_indices = set()
        for i in range(len(recipes)):
            if i in removed_indices:
                continue
            for j in range(i + 1, len(recipes)):
                if j in removed_indices:
                    continue

                if method == "mi":
                    corr = mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j])[0]
                elif method == "dcor":
                    corr = dcor.distance_correlation(X[:, i], X[:, j])
                else:
                    # デフォルトはピアソン相関
                    corr = np.abs(np.corrcoef(X[:, i], X[:, j])[0, 1])

                if corr > threshold:
                    removed_indices.add(j)  # 単純に後ろのインデックスを削除

        final_recipes = [r for i, r in enumerate(recipes) if i not in removed_indices]
        print(f"Collinearity filter removed {len(removed_indices)} features. Kept {len(final_recipes)}.")
        return final_recipes

    def _get_final_model_np(self, recipes_list: list) -> Tuple:
        if not recipes_list:
            return float("inf"), None, None, None
        X = np.stack([self.executor.execute(r) for r in recipes_list], axis=1)
        if np.isnan(X).any():
            X[np.where(np.isnan(X))] = np.take(np.nanmean(X, axis=0), np.where(np.isnan(X))[1])
        X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
        X_std[X_std < 1e-8] = 1.0
        try:
            c_std, res, _, _ = sp_lstsq((X - X_mean) / X_std, self.y_centered, check_finite=False)
            rmse = np.sqrt(res / self.y.shape[0]) if res.size > 0 and res >= 0 else float("inf")
            coeffs, intercept = c_std / X_std, self.y_mean - np.dot(c_std / X_std, X_mean)
            return rmse, tuple(recipes_list), coeffs, intercept
        except np.linalg.LinAlgError:
            return float("inf"), None, None, None

    def _run_feature_selector(self, candidate_recipes: List[FeatureRecipe], method: str, params: dict):
        print(f"--- Running Feature Selection with {method.upper()} ---")

        # 1. [任意] Global SIS
        if params["n_global_sis_features"]:
            print(f"Performing Global SIS, selecting top {params['n_global_sis_features']} features...")
            candidate_recipes = self._run_sis(self.y_centered, candidate_recipes, k=params["n_global_sis_features"])

        # 2. [任意] Collinearity Filter
        if params["collinearity_filter"]:
            candidate_recipes = self._filter_collinear_features(candidate_recipes, params["collinearity_filter"], params["collinearity_threshold"])

        # 3. Lasso/LightGBM
        X_candidates, valid_recipes = [], []
        for r in candidate_recipes:
            arr = self.executor.execute(r)
            valid = ~np.isnan(arr)
            if valid.sum() > 1:
                vf, mean, std = arr[valid], np.mean(arr[valid]), np.std(arr[valid])
                if std > 1e-8:
                    std_arr = np.zeros_like(arr)
                    std_arr[valid] = (vf - mean) / std
                    X_candidates.append(std_arr)
                    valid_recipes.append(r)

        if not X_candidates:
            print(f"No valid features for {method}.")
            return
        X_matrix = np.stack(X_candidates, axis=1)

        if method == "lasso":
            model = Lasso(alpha=params["alpha"], max_iter=5000, random_state=42, tol=1e-4)
            model.fit(X_matrix, self.y_centered)
            importances = np.abs(model.coef_)
            selected_indices = np.where(importances > params["coeff_threshold"])[0]
        elif method == "lightgbm":
            model = lgb.LGBMRegressor(**params["lightgbm_params"])
            model.fit(X_matrix, self.y_centered)
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1]
            # 重要度が0でない特徴の中から上位を選択
            selected_indices = [i for i in top_indices if importances[i] > 0][: params["n_features_to_select"]]

        if len(selected_indices) == 0:
            print(f"{method.upper()} selected 0 features.")
            return

        final_recipes = [valid_recipes[i] for i in selected_indices]
        print(f"{method.upper()} selected {len(final_recipes)} features.")

        rmse, recipes_tuple, coeffs, intercept = self._get_final_model_np(final_recipes)
        if recipes_tuple:
            n_terms = len(recipes_tuple)
            self.best_models[n_terms] = {"rmse": rmse, "recipes": recipes_tuple, "coeffs": coeffs, "intercept": intercept}
            print(f"Found {n_terms}-term model via {method.upper()}: RMSE={rmse:.6f}, Eq: {self._format_equation(recipes_tuple, coeffs, intercept)}")

    def fit(self):
        print(f"***************** Starting SISSO Regressor (NumPy/SciPy Backend, Method: {self.so_method}) *****************")

        if self.so_method == "lasso" or self.so_method == "lightgbm":
            self._run_feature_selector(self.all_recipes, self.so_method, self.selection_params)

        elif self.so_method == "exhaustive":
            params = self.selection_params
            residual, pool = self.y_centered, []
            for i in range(1, params["n_term"] + 1):
                start_time = time.time()
                print(f"\n===== Searching for {i}-term models =====")
                top_k = self._run_sis(residual, [r for r in self.all_recipes if r not in pool], k=params["n_sis_features"])
                pool.extend(top_k)
                print(f"SIS selected {len(top_k)} new features. Pool size: {len(pool)}")
                combos = list(combinations(pool, i))
                print(f"--- Running SO for {i}-term models. Total combinations: {len(combos)} ---")
                best_rmse, best_model = float("inf"), None
                for combo in combos:
                    rmse, recipes, coeffs, intercept = self._get_final_model_np(list(combo))
                    if rmse < best_rmse:
                        best_rmse, best_model = rmse, {"r": recipes, "c": coeffs, "i": intercept}
                if best_model:
                    self.best_models[i] = {"rmse": best_rmse, "recipes": best_model["r"], "coeffs": best_model["c"], "intercept": best_model["i"]}
                    y_pred = best_model["i"] + np.sum(best_model["c"] * np.stack([np.nan_to_num(self.executor.execute(r)) for r in best_model["r"]], axis=1), axis=1)
                    residual = self.y - y_pred
                    print(f"Best {i}-term model: RMSE={best_rmse:.6f}, Eq: {self._format_equation(best_model['r'], best_model['c'], best_model['i'])}")
                else:
                    print(f"No valid model found for term {i}.")
                print(f"Time: {time.time() - start_time:.2f} seconds")

        if not self.best_models:
            return None
        best_model = min(self.best_models.values(), key=lambda m: m["rmse"])
        r2 = 1.0 - (best_model["rmse"] ** 2 * self.y.shape[0]) / np.sum(self.y_centered[~np.isnan(self.y_centered)] ** 2)
        return best_model["rmse"], self._format_equation(best_model["recipes"], best_model["coeffs"], best_model["intercept"]), r2, self.best_models
