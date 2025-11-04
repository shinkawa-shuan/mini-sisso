# # regressor_numpy.py (Corrected __init__ signature)
# import time
# from itertools import combinations
# from typing import Dict, List, Tuple

# import numpy as np
# from scipy.linalg import lstsq as sp_lstsq

# from .executor_numpy import RecipeExecutor
# from .recipe import FeatureRecipe


# class SissoRegressorNumPy:
#     # ★★★ CRITICAL CORRECTION: Adjust __init__ arguments ★★★
#     def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutor, y: np.ndarray, n_term: int, k: int, alpha: float):  # so_method is handled by model.py

#         # if so_method != 'exhaustive':
#         #     raise NotImplementedError("Only 'exhaustive' is supported.") # ★★★ このチェックを削除 ★★★

#         self.all_recipes = all_recipes
#         self.executor = executor
#         self.y = y
#         self.n_term = n_term
#         self.k = k
#         self.y_mean, self.y_centered = np.mean(y), y - np.mean(y)
#         self.best_models: Dict[int, dict] = {}

#     def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: np.ndarray, intercept: float) -> str:
#         equation = "".join(f"{c:+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten()))
#         return equation + f"{intercept:+.6f}"

#     def _run_sis(self, target: np.ndarray, recipes: List[FeatureRecipe]) -> List[FeatureRecipe]:
#         if not recipes:
#             return []
#         scores = []
#         for recipe in recipes:
#             tensor = self.executor.execute(recipe)
#             valid = ~np.isnan(tensor) & ~np.isnan(target)
#             if valid.sum() < 2:
#                 scores.append(0.0)
#                 continue
#             valid_f, valid_t = tensor[valid], target[valid]
#             mean, std = np.mean(valid_f), np.std(valid_f)
#             if std > 1e-8:
#                 scores.append(np.abs(np.dot(valid_t - np.mean(valid_t), (valid_f - mean) / std)))
#             else:
#                 scores.append(0.0)
#         return [recipes[i] for i in np.argsort(scores)[::-1][: self.k]]

#     def fit(self):
#         residual, pool, n_samples = self.y_centered, [], self.y.shape[0]
#         for i in range(1, self.n_term + 1):
#             start_time = time.time()
#             print(f"\n===== Searching for {i}-term models =====")

#             top_k = self._run_sis(residual, [r for r in self.all_recipes if r not in pool])
#             pool.extend(top_k)
#             print(f"SIS selected {len(top_k)} new features. Pool size: {len(pool)}")

#             combos = list(combinations(pool, i))
#             if not combos:
#                 continue
#             print(f"--- Running SO for {i}-term models. Total combinations: {len(combos)} ---")

#             best_rmse, best_model = float("inf"), None
#             for combo in combos:
#                 X = np.stack([self.executor.execute(r) for r in combo], axis=1)

#                 if np.isnan(X).any():
#                     col_mean = np.nanmean(X, axis=0)
#                     X[np.where(np.isnan(X))] = np.take(col_mean, np.where(np.isnan(X))[1])

#                 X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
#                 X_std[X_std < 1e-8] = 1.0
#                 X_std_np = (X - X_mean) / X_std

#                 try:
#                     c_std, res, _, _ = sp_lstsq(X_std_np, self.y_centered)
#                     rmse = np.sqrt(res / n_samples) if res.size > 0 else float("inf")
#                     if rmse < best_rmse:
#                         best_rmse = rmse
#                         best_model = {"r": combo, "cs": c_std, "xm": X_mean, "xs": X_std}
#                 except np.linalg.LinAlgError:
#                     continue

#             if best_model:
#                 coeffs = best_model["cs"] / best_model["xs"]
#                 intercept = self.y_mean - np.dot(coeffs, best_model["xm"])
#                 self.best_models[i] = {"rmse": best_rmse, "recipes": best_model["r"], "coeffs": coeffs, "intercept": intercept}

#                 y_pred = intercept + np.sum(coeffs * np.stack([np.nan_to_num(self.executor.execute(r)) for r in best_model["r"]], axis=1), axis=1)
#                 residual = self.y - y_pred

#                 print(f"Best {i}-term model: RMSE={best_rmse:.6f}, Eq: {self._format_equation(best_model['r'], coeffs, intercept)}")
#             else:
#                 print(f"No valid model found for term {i}.")
#             print(f"Time: {time.time() - start_time:.2f} seconds")

#         if not self.best_models:
#             return None
#         best_model = min(self.best_models.values(), key=lambda m: m["rmse"])
#         r2 = 1.0 - (best_model["rmse"] ** 2 * n_samples) / np.sum(self.y_centered[~np.isnan(self.y_centered)] ** 2)
#         final_equation = self._format_equation(best_model["recipes"], best_model["coeffs"], best_model["intercept"])
#         return best_model["rmse"], final_equation, r2, self.best_models

# regressor_numpy.py (Final version with Lasso)
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import lstsq as sp_lstsq
from sklearn.linear_model import Lasso

from .executor_numpy import RecipeExecutor
from .recipe import FeatureRecipe


class SissoRegressorNumPy:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutor, y: np.ndarray, n_term: int, k: int, so_method: str = "exhaustive", alpha: float = 0.01):
        self.all_recipes = all_recipes
        self.executor = executor
        self.y = y
        self.n_term = n_term
        self.k = k
        self.so_method = so_method
        self.alpha = alpha
        self.y_mean, self.y_centered = np.mean(y), y - np.mean(y)
        self.best_models: Dict[int, dict] = {}

    def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: np.ndarray, intercept: float) -> str:
        equation = "".join(f"{c:+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten()))
        return equation + f"{intercept:+.6f}"

    def _run_sis(self, target: np.ndarray, recipes: List[FeatureRecipe]) -> List[FeatureRecipe]:
        if not recipes:
            return []
        scores = []
        for recipe in recipes:
            array = self.executor.execute(recipe)
            valid = ~np.isnan(array) & ~np.isnan(target)
            if valid.sum() < 2:
                scores.append(0.0)
                continue
            valid_f, valid_t = array[valid], target[valid]
            mean, std = np.mean(valid_f), np.std(valid_f)
            if std > 1e-8:
                scores.append(np.abs(np.dot(valid_t - np.mean(valid_t), (valid_f - mean) / std)))
            else:
                scores.append(0.0)
        return [recipes[i] for i in np.argsort(scores)[::-1][: self.k]]

    def _get_final_model_np(self, model_recipes_list: list) -> Tuple:
        num_terms = len(model_recipes_list)
        n_samples = self.y.shape[0]

        X = np.stack([self.executor.execute(r) for r in model_recipes_list], axis=1)

        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])

        X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
        X_std[X_std < 1e-8] = 1.0
        X_std_np = (X - X_mean) / X_std

        try:
            coeffs_std, res, _, _ = sp_lstsq(X_std_np, self.y_centered, check_finite=False)
            rmse = np.sqrt(res / n_samples) if res.size > 0 else float("inf")

            coeffs = coeffs_std / X_std
            intercept = self.y_mean - np.dot(coeffs, X_mean)

            return rmse, tuple(model_recipes_list), coeffs, intercept
        except np.linalg.LinAlgError:
            return float("inf"), None, None, None

    def _run_so_lasso(self):
        print(f"--- Running SO with LASSO (alpha={self.alpha}). Candidates: {len(self.all_recipes)} ---")

        X_candidates, valid_recipes = [], []
        for recipe in self.all_recipes:
            array = self.executor.execute(recipe)
            valid = ~np.isnan(array)
            if valid.sum() > 1:
                valid_f = array[valid]
                mean, std = np.mean(valid_f), np.std(valid_f)
                if std > 1e-8:
                    std_array = np.zeros_like(array)
                    std_array[valid] = (valid_f - mean) / std
                    X_candidates.append(std_array)
                    valid_recipes.append(recipe)

        if not X_candidates:
            print("No valid features for LASSO.")
            return

        X_matrix = np.stack(X_candidates, axis=1)
        lasso = Lasso(alpha=self.alpha, max_iter=3000, random_state=42, tol=1e-4)
        lasso.fit(X_matrix, self.y_centered)

        selected_indices = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        if len(selected_indices) == 0:
            print("LASSO selected 0 features.")
            return

        model_recipes = [valid_recipes[i] for i in selected_indices]
        num_terms = len(model_recipes)
        print(f"LASSO selected {num_terms} features.")

        rmse, recipes_tuple, coeffs, intercept = self._get_final_model_np(model_recipes)
        if recipes_tuple:
            self.best_models[num_terms] = {"rmse": rmse, "recipes": recipes_tuple, "coeffs": coeffs, "intercept": intercept}
            print(f"Found a {num_terms}-term model via LASSO. RMSE: {rmse:.6f}")
            print(f"Equation: {self._format_equation(recipes_tuple, coeffs, intercept)}")

    def fit(self):
        print(f"***************** Starting SISSO Regressor (NumPy/SciPy Backend, Method: {self.so_method}) *****************")

        if self.so_method == "lasso":
            self._run_so_lasso()

        elif self.so_method == "exhaustive":
            residual, pool = self.y_centered, []
            for i in range(1, self.n_term + 1):
                start_time = time.time()
                print(f"\n===== Searching for {i}-term models =====")

                top_k = self._run_sis(residual, [r for r in self.all_recipes if r not in pool])
                pool.extend(top_k)
                print(f"SIS selected {len(top_k)} new features. Pool size: {len(pool)}")

                combos = list(combinations(pool, i))
                if not combos:
                    continue
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
        final_equation = self._format_equation(best_model["recipes"], best_model["coeffs"], best_model["intercept"])
        return best_model["rmse"], final_equation, r2, self.best_models
