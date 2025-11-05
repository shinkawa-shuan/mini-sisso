# regressor_numpy.py (Final version with LightGBM)
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import lstsq as sp_lstsq
from sklearn.linear_model import Lasso

from .executor_numpy import RecipeExecutor
from .recipe import FeatureRecipe

# LightGBMはオプションなので、try-exceptでインポート
try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class SissoRegressorNumPy:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutor, y: np.ndarray, n_term: int, k: int, so_method: str = "exhaustive", alpha: float = 0.01):
        self.all_recipes, self.executor, self.y = all_recipes, executor, y
        self.n_term, self.k, self.so_method, self.alpha = n_term, k, so_method, alpha
        self.y_mean, self.y_centered = np.mean(y), y - np.mean(y)
        self.best_models: Dict[int, dict] = {}

    def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: np.ndarray, intercept: float) -> str:
        if not recipes:
            return f"{intercept:+.6f}"
        return "".join(f"{c:+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten())) + f"{intercept:+.6f}"

    def _run_sis(self, target: np.ndarray, recipes: List[FeatureRecipe]) -> List[FeatureRecipe]:
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
        return [recipes[i] for i in np.argsort(scores)[::-1][: self.k]]

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

    def _run_so_lasso(self, coeff_threshold: float = 1e-5):
        print(f"--- Running SO with LASSO (alpha={self.alpha}). Candidates: {len(self.all_recipes)} ---")
        X_candidates, valid_recipes = [], []
        for r in self.all_recipes:
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
            print("No valid features for LASSO.")
            return

        lasso = Lasso(alpha=self.alpha, max_iter=5000, random_state=42, tol=1e-4).fit(np.stack(X_candidates, axis=1), self.y_centered)
        selected_indices = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        if len(selected_indices) == 0:
            print("LASSO selected 0 features.")
            return

        recipes = [valid_recipes[i] for i in selected_indices]
        rmse, recipes_tuple, coeffs, _ = self._get_final_model_np(recipes)
        if recipes_tuple:
            final_recipes = tuple(np.array(recipes_tuple)[np.abs(coeffs) > coeff_threshold])
            print(f"Pruning: Kept {len(final_recipes)} of {len(recipes)} features.")
            final_rmse, final_recipes_tuple, final_coeffs, final_intercept = self._get_final_model_np(list(final_recipes))
            if final_recipes_tuple:
                n_terms = len(final_recipes_tuple)
                self.best_models[n_terms] = {"rmse": final_rmse, "recipes": final_recipes_tuple, "coeffs": final_coeffs, "intercept": final_intercept}
                print(f"Found pruned {n_terms}-term model via LASSO: RMSE={final_rmse:.6f}, Eq: {self._format_equation(final_recipes_tuple, final_coeffs, final_intercept)}")

    def _run_so_lightgbm(self, n_features_to_select: int = 40):
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is required for this method. Please install with 'pip install \"mini-sisso[lightgbm]\"'")
        print(f"--- Running SO with LightGBM. Selecting top {n_features_to_select} features. ---")
        X_candidates, valid_recipes = [], []
        for r in self.all_recipes:
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
            print("No valid features for LightGBM.")
            return

        lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(np.stack(X_candidates, axis=1), self.y_centered)
        selected_indices = np.argsort(lgbm.feature_importances_)[::-1][:n_features_to_select]
        if len(selected_indices) == 0:
            print("LightGBM selected 0 features.")
            return

        recipes = [valid_recipes[i] for i in selected_indices]
        rmse, recipes_tuple, coeffs, intercept = self._get_final_model_np(recipes)
        if recipes_tuple:
            n_terms = len(recipes_tuple)
            self.best_models[n_terms] = {"rmse": rmse, "recipes": recipes_tuple, "coeffs": coeffs, "intercept": intercept}
            print(f"Found {n_terms}-term model via LightGBM: RMSE={rmse:.6f}, Eq: {self._format_equation(recipes_tuple, coeffs, intercept)}")

    def fit(self):
        print(f"***************** Starting SISSO Regressor (NumPy/SciPy Backend, Method: {self.so_method}) *****************")

        if self.so_method == "lasso":
            self._run_so_lasso()
        elif self.so_method == "lightgbm":
            self._run_so_lightgbm()
        elif self.so_method == "exhaustive":
            residual, pool = self.y_centered, []
            for i in range(1, self.n_term + 1):
                start_time = time.time()
                print(f"\n===== Searching for {i}-term models =====")
                top_k = self._run_sis(residual, [r for r in self.all_recipes if r not in pool])
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
