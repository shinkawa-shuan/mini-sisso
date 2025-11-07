# regressor_torch.py
import time
from itertools import combinations
from typing import Dict, List, Tuple

import dcor
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso

from .executor_torch import RecipeExecutorTorch
from .recipe import FeatureRecipe

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class SissoRegressorTorch:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutorTorch, y: torch.Tensor, so_method: str, selection_params: dict):
        self.all_recipes = all_recipes
        self.executor = executor
        self.y = y
        self.so_method = so_method
        self.selection_params = self._parse_selection_params(selection_params)
        self.device = executor.device
        self.y_mean, self.y_centered = y.mean(), y - y.mean()
        self.best_models: Dict[int, dict] = {}

    def _parse_selection_params(self, params: dict) -> dict:
        if params is None:
            params = {}
        params.setdefault("n_term", 2)
        params.setdefault("n_sis_features", 10)
        params.setdefault("alpha", 0.01)
        params.setdefault("n_features_to_select", 40)
        params.setdefault("lightgbm_params", {"random_state": 42, "n_jobs": -1, "verbosity": -1})
        params.setdefault("n_global_sis_features", None)
        params.setdefault("collinearity_filter", None)
        params.setdefault("collinearity_threshold", 0.9)
        params.setdefault("coeff_threshold", 1e-5)
        return params

    def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: torch.Tensor, intercept: torch.Tensor) -> str:
        if not recipes:
            return f"{intercept.item():+.6f}"
        return "".join(f"{c.item():+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten())) + f"{intercept.item():+.6f}"

    def _run_sis(self, target: torch.Tensor, recipes: List[FeatureRecipe], k: int) -> List[FeatureRecipe]:
        if not recipes:
            return []
        scores = []
        for r in recipes:
            t = self.executor.execute(r)
            v = ~torch.isnan(t) & ~torch.isnan(target)
            if v.sum() < 2:
                scores.append(0.0)
                continue
            vf, vt = t[v], target[v]
            mean, std = vf.mean(), vf.std()
            if std > 1e-8:
                scores.append(torch.abs(torch.dot(vt - vt.mean(), (vf - mean) / std)).item())
            else:
                scores.append(0.0)
        return [recipes[i] for i in np.argsort(scores)[::-1][:k]]

    def _filter_collinear_features(self, recipes: List[FeatureRecipe], method: str, threshold: float) -> List[FeatureRecipe]:
        print(f"--- Applying Collinearity Filter (method: {method}, threshold: {threshold}) ---")
        # PyTorchテンソルをNumPyに変換して計算
        X_torch = torch.stack([self.executor.execute(r) for r in recipes], dim=1)
        X_np = X_torch.cpu().numpy()
        if np.isnan(X_np).any():
            X_np[np.where(np.isnan(X_np))] = np.take(np.nanmean(X_np, axis=0), np.where(np.isnan(X_np))[1])

        removed_indices = set()
        for i in range(len(recipes)):
            if i in removed_indices:
                continue
            for j in range(i + 1, len(recipes)):
                if j in removed_indices:
                    continue
                if method == "mi":
                    corr = mutual_info_regression(X_np[:, i].reshape(-1, 1), X_np[:, j])[0]
                elif method == "dcor":
                    corr = dcor.distance_correlation(X_np[:, i], X_np[:, j])
                else:
                    corr = np.abs(np.corrcoef(X_np[:, i], X_np[:, j])[0, 1])
                if corr > threshold:
                    removed_indices.add(j)

        final_recipes = [r for i, r in enumerate(recipes) if i not in removed_indices]
        print(f"Collinearity filter removed {len(removed_indices)} features. Kept {len(final_recipes)}.")
        return final_recipes

    def _get_final_model_torch(self, recipes_list: list) -> Tuple:
        if not recipes_list:
            return float("inf"), None, None, None
        n_terms, n_samples = len(recipes_list), self.y.shape[0]
        X_batch = torch.stack([self.executor.execute(r) for r in recipes_list], dim=1).unsqueeze(0)

        for j in range(n_terms):
            col = X_batch[..., j]
            v = ~torch.isnan(col) & ~torch.isinf(col)
            if v.any():
                col[~v] = col[v].mean()
        X_batch.clamp_(-1e9, 1e9)

        X_mean, X_std = X_batch.mean(dim=1, keepdim=True), X_batch.std(dim=1, keepdim=True)
        X_std[X_std < 1e-8] = 1.0
        X_batch_std = (X_batch - X_mean) / X_std

        y_batch = self.y_centered.expand(1, -1)

        try:
            c_std, res, _, _ = torch.linalg.lstsq(X_batch_std, y_batch)
            rmse = torch.sqrt(res[0] / n_samples).item()
            coeffs = c_std[0] / X_std.squeeze()
            intercept = self.y_mean - torch.dot(coeffs.flatten(), X_mean.flatten())
            return rmse, tuple(recipes_list), coeffs, intercept
        except torch.linalg.LinAlgError:
            return float("inf"), None, None, None

    def _run_feature_selector(self, candidate_recipes: List[FeatureRecipe], method: str, params: dict):
        print(f"--- Running Feature Selection with {method.upper()} (GPU Backend) ---")

        if params["n_global_sis_features"]:
            print(f"Performing Global SIS, selecting top {params['n_global_sis_features']} features...")
            candidate_recipes = self._run_sis(self.y_centered, candidate_recipes, k=params["n_global_sis_features"])
        if params["collinearity_filter"]:
            candidate_recipes = self._filter_collinear_features(candidate_recipes, params["collinearity_filter"], params["collinearity_threshold"])

        X_candidates, valid_recipes = [], []
        for r in candidate_recipes:
            t = self.executor.execute(r)
            v = ~torch.isnan(t)
            if v.sum() > 1:
                vf, mean, std = t[v], t[v].mean(), t[v].std()
                if std > 1e-8:
                    st = torch.zeros_like(t)
                    st[v] = (vf - mean) / std
                    X_candidates.append(st)
                    valid_recipes.append(r)

        if not X_candidates:
            print(f"No valid features for {method}.")
            return
        X_matrix = torch.stack(X_candidates, dim=1).cpu().numpy()
        y_np = self.y_centered.cpu().numpy()

        if method == "lasso":
            model = Lasso(alpha=params["alpha"], max_iter=5000, random_state=42, tol=1e-4).fit(X_matrix, y_np)
            importances = np.abs(model.coef_)
            selected_indices = np.where(importances > params["coeff_threshold"])[0]
        elif method == "lightgbm":
            if not LGBM_AVAILABLE:
                raise ImportError("LightGBM is required. Please install with 'pip install \"mini-sisso[lightgbm]\"'")
            model = lgb.LGBMRegressor(**params["lightgbm_params"]).fit(X_matrix, y_np)
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1]
            selected_indices = [i for i in top_indices if importances[i] > 0][: params["n_features_to_select"]]

        if len(selected_indices) == 0:
            print(f"{method.upper()} selected 0 features.")
            return

        final_recipes = [valid_recipes[i] for i in selected_indices]
        print(f"{method.upper()} selected {len(final_recipes)} features.")

        rmse, recipes_tuple, coeffs, intercept = self._get_final_model_torch(final_recipes)
        if recipes_tuple:
            n_terms = len(recipes_tuple)
            self.best_models[n_terms] = {"rmse": rmse, "recipes": recipes_tuple, "coeffs": coeffs, "intercept": intercept}
            print(f"Found {n_terms}-term model via {method.upper()}: RMSE={rmse:.6f}, Eq: {self._format_equation(recipes_tuple, coeffs, intercept)}")

    def fit(self):
        print(f"***************** Starting SISSO Regressor (GPU Backend, Method: {self.so_method}) *****************")

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

                # Note: The original batch-processing SO for torch is omitted for simplicity here.
                # A full implementation would batch the 'combinations' list.
                combos = list(combinations(pool, i))
                print(f"--- Running SO for {i}-term models. Total combinations: {len(combos)} ---")
                best_rmse, best_model = float("inf"), None
                for combo in combos:
                    rmse, recipes, coeffs, intercept = self._get_final_model_torch(list(combo))
                    if rmse < best_rmse:
                        best_rmse, best_model = rmse, {"r": recipes, "c": coeffs, "i": intercept}

                if best_model:
                    self.best_models[i] = {"rmse": best_rmse, "recipes": best_model["r"], "coeffs": best_model["c"], "intercept": best_model["i"]}
                    y_pred = best_model["i"] + torch.sum(best_model["c"].flatten() * torch.stack([torch.nan_to_num(self.executor.execute(r)) for r in best_model["r"]], dim=1), dim=1)
                    residual = self.y - y_pred
                    print(f"Best {i}-term model: RMSE={best_rmse:.6f}, Eq: {self._format_equation(best_model['r'], best_model['c'], best_model['i'])}")
                else:
                    print(f"No valid model found for term {i}.")
                print(f"Time: {time.time() - start_time:.2f} seconds")

        if not self.best_models:
            return None
        best_model = min(self.best_models.values(), key=lambda m: m["rmse"])
        r2 = 1.0 - (best_model["rmse"] ** 2 * self.y.shape[0]) / torch.sum((self.y_centered[~torch.isnan(self.y_centered)]) ** 2).item()
        return best_model["rmse"], self._format_equation(best_model["recipes"], best_model["coeffs"], best_model["intercept"]), r2, self.best_models
