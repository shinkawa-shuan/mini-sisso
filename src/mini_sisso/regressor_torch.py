# regressor_torch.py (Final version with LightGBM)
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Lasso

from .executor_torch import RecipeExecutorTorch
from .recipe import FeatureRecipe

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class SissoRegressorTorch:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutorTorch, y: torch.Tensor, n_term: int, k: int, so_method: str = "exhaustive", alpha: float = 0.01):
        self.all_recipes, self.executor, self.y = all_recipes, executor, y
        self.n_term, self.k, self.so_method, self.alpha = n_term, k, so_method, alpha
        self.device = executor.device
        self.y_mean, self.y_centered = y.mean(), y - y.mean()
        self.best_models: Dict[int, dict] = {}

    def _format_equation(self, recipes, coeffs, intercept) -> str:
        return "".join(f"{c.item():+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten())) + f"{intercept.item():+.6f}"

    def _run_sis(self, target: torch.Tensor, recipes: List[FeatureRecipe]) -> List[FeatureRecipe]:
        # (実装は変更なし)
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
        return [recipes[i] for i in np.argsort(scores)[::-1][: self.k]]

    def _get_final_model_torch(self, recipes_list: list) -> Tuple:
        if not recipes_list:
            return float("inf"), None, None, None
        X_batch = torch.stack([self.executor.execute(r) for r in recipes_list], dim=1).unsqueeze(0)
        for j in range(len(recipes_list)):
            col = X_batch[..., j]
            v = ~torch.isnan(col) & ~torch.isinf(col)
            if v.any():
                col[~v] = col[v].mean()
        X_batch.clamp_(-1e9, 1e9)
        X_mean, X_std = X_batch.mean(dim=1, keepdim=True), X_batch.std(dim=1, keepdim=True)
        X_std[X_std < 1e-8] = 1.0
        try:
            c_std, res, _, _ = torch.linalg.lstsq((X_batch - X_mean) / X_std, self.y_centered.expand(1, -1))
            rmse = torch.sqrt(res[0] / self.y.shape[0]).item()
            coeffs = c_std[0] / X_std.squeeze()
            intercept = self.y_mean - torch.dot(coeffs.flatten(), X_mean.flatten())
            return rmse, tuple(recipes_list), coeffs, intercept
        except torch.linalg.LinAlgError:
            return float("inf"), None, None, None

    def _run_so_lasso(self, coeff_threshold: float = 1e-5):
        # (実装は変更なし)
        print(f"--- Running SO with LASSO (GPU Backend, alpha={self.alpha})...")
        X_candidates, valid_recipes = [], []
        for r in self.all_recipes:
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
            print("No valid features for LASSO.")
            return

        lasso = Lasso(alpha=self.alpha, max_iter=5000, random_state=42, tol=1e-4).fit(torch.stack(X_candidates, dim=1).cpu().numpy(), self.y_centered.cpu().numpy())
        sel_indices = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        if len(sel_indices) == 0:
            print("LASSO selected 0 features.")
            return

        recipes = [valid_recipes[i] for i in sel_indices]
        _, recipes_tuple, coeffs, _ = self._get_final_model_torch(recipes)
        if recipes_tuple:
            final_recipes = tuple(np.array(recipes_tuple)[(np.abs(coeffs.cpu().numpy()) > coeff_threshold)])
            print(f"Pruning: Kept {len(final_recipes)} of {len(recipes)} features.")
            final_rmse, final_recipes_tuple, final_coeffs, final_intercept = self._get_final_model_torch(list(final_recipes))
            if final_recipes_tuple:
                n_terms = len(final_recipes_tuple)
                self.best_models[n_terms] = {"rmse": final_rmse, "recipes": final_recipes_tuple, "coeffs": final_coeffs, "intercept": final_intercept}
                print(f"Found pruned {n_terms}-term model via LASSO: RMSE={final_rmse:.6f}, Eq: {self._format_equation(final_recipes_tuple, final_coeffs, final_intercept)}")

    def _run_so_lightgbm(self, n_features_to_select: int = 40):
        # (NumPy版とほぼ同じ実装。データをCPUに送り返して計算)
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is required for this method. Please install with 'pip install \"mini-sisso[lightgbm]\"'")
        print(f"--- Running SO with LightGBM (GPU Backend). Selecting top {n_features_to_select} features. ---")
        X_candidates, valid_recipes = [], []
        for r in self.all_recipes:
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
            print("No valid features for LightGBM.")
            return

        lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(torch.stack(X_candidates, dim=1).cpu().numpy(), self.y_centered.cpu().numpy())
        sel_indices = np.argsort(lgbm.feature_importances_)[::-1][:n_features_to_select]
        if len(sel_indices) == 0:
            print("LightGBM selected 0 features.")
            return

        recipes = [valid_recipes[i] for i in sel_indices]
        rmse, recipes_tuple, coeffs, intercept = self._get_final_model_torch(recipes)
        if recipes_tuple:
            n_terms = len(recipes_tuple)
            self.best_models[n_terms] = {"rmse": rmse, "recipes": recipes_tuple, "coeffs": coeffs, "intercept": intercept}
            print(f"Found {n_terms}-term model via LightGBM: RMSE={rmse:.6f}, Eq: {self._format_equation(recipes_tuple, coeffs, intercept)}")

    def fit(self):
        print(f"***************** Starting SISSO Regressor (GPU Backend, Method: {self.so_method}) *****************")
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
                rmse, recipes, coeffs, intercept = self._run_so_exhaustive(pool, i)
                if recipes:
                    self.best_models[i] = {"rmse": rmse, "recipes": recipes, "coeffs": coeffs, "intercept": intercept}
                    y_pred = intercept + torch.sum(coeffs.flatten() * torch.stack([torch.nan_to_num(self.executor.execute(r)) for r in recipes], dim=1), dim=1)
                    residual = self.y - y_pred
                    print(f"Best {i}-term model: RMSE={rmse:.6f}, Eq: {self._format_equation(recipes, coeffs, intercept)}")
                else:
                    print(f"No valid model found for term {i}.")
                print(f"Time: {time.time() - start_time:.2f} seconds")

        if not self.best_models:
            return None
        best_model = min(self.best_models.values(), key=lambda m: m["rmse"])
        r2 = 1.0 - (best_model["rmse"] ** 2 * self.y.shape[0]) / torch.sum((self.y_centered[~torch.isnan(self.y_centered)]) ** 2).item()
        return best_model["rmse"], self._format_equation(best_model["recipes"], best_model["coeffs"], best_model["intercept"]), r2, self.best_models
