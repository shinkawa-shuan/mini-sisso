# regressor_torch.py (Final version, corrected for dot product shapes)
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch

from .executor_torch import RecipeExecutorTorch
from .recipe import FeatureRecipe


class SissoRegressorTorch:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutorTorch, y: torch.Tensor, n_term: int, k: int, alpha: float):
        self.all_recipes, self.executor, self.y, self.n_term, self.k = all_recipes, executor, y, n_term, k
        self.device = executor.device
        self.y_mean, self.y_centered = y.mean(), y - y.mean()
        self.best_models: Dict[int, dict] = {}

    def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: torch.Tensor, intercept: torch.Tensor) -> str:
        equation = "".join(f"{c.item():+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten()))
        return equation + f"{intercept.item():+.6f}"

    def _run_sis(self, target: torch.Tensor, recipes: List[FeatureRecipe]) -> List[FeatureRecipe]:
        if not recipes:
            return []
        scores = []
        for recipe in recipes:
            tensor = self.executor.execute(recipe)
            valid = ~torch.isnan(tensor) & ~torch.isnan(target)
            if valid.sum() < 2:
                scores.append(0.0)
                continue
            valid_f, valid_t = tensor[valid], target[valid]
            mean, std = valid_f.mean(), valid_f.std()
            if std > 1e-8:
                scores.append(torch.abs(torch.dot(valid_t - valid_t.mean(), (valid_f - mean) / std)).item())
            else:
                scores.append(0.0)
        return [recipes[i] for i in np.argsort(scores)[::-1][: self.k]]

    def _run_so_exhaustive(self, so_candidate_pool: List[FeatureRecipe], n_current_term: int, batch_size: int = 5000) -> Tuple:
        print(f"--- Running SO (GPU) for {n_current_term}-term models. Candidates: {len(so_candidate_pool)} ---")
        model_combinations = list(combinations(so_candidate_pool, n_current_term))
        if not model_combinations:
            return float("inf"), None, None, None
        print(f"Total combinations to test: {len(model_combinations)}")

        best_rmse, best_model_recipe_tuple, best_coeffs, best_intercept = float("inf"), None, None, None
        n_samples = self.y.shape[0]

        for i in range(0, len(model_combinations), batch_size):
            batch_combinations = model_combinations[i : i + batch_size]
            current_batch_size = len(batch_combinations)
            X_batch = torch.zeros((current_batch_size, n_samples, n_current_term), device=self.device)
            unique_recipes_in_batch = set(r for combo in batch_combinations for r in combo)
            tensor_map = {r: self.executor.execute(r) for r in unique_recipes_in_batch}

            for j, combo in enumerate(batch_combinations):
                for l, recipe in enumerate(combo):
                    X_batch[j, :, l] = tensor_map[recipe]

            for j in range(n_current_term):
                col = X_batch[..., j]
                valid_mask = ~torch.isnan(col) & ~torch.isinf(col)
                if valid_mask.any():
                    col_mean = col[valid_mask].mean()
                    col[~valid_mask] = col_mean
            X_batch.clamp_(-1e9, 1e9)

            X_mean = X_batch.mean(dim=1, keepdim=True)
            X_std = X_batch.std(dim=1, keepdim=True)
            X_std[X_std < 1e-8] = 1.0
            X_batch_std = (X_batch - X_mean) / X_std

            y_batch = self.y_centered.expand(current_batch_size, -1)

            try:
                coeffs_std, residuals, _, _ = torch.linalg.lstsq(X_batch_std, y_batch)

                if residuals.numel() == 0:
                    continue

                min_residual, min_idx = torch.min(residuals, dim=0)
                rmse = torch.sqrt(min_residual / n_samples).item()

                if not (np.isinf(rmse) or np.isnan(rmse)):
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model_recipe_tuple = batch_combinations[min_idx]

                        best_coeffs_std = coeffs_std[min_idx]
                        best_X_mean = X_mean[min_idx]
                        best_X_std = X_std[min_idx]

                        best_coeffs = best_coeffs_std / best_X_std.squeeze(0)

                        # ★★★ FINAL CORRECTION ★★★
                        # Ensure both tensors are 1D before dot product
                        intercept_correction = torch.dot(best_coeffs.flatten(), best_X_mean.flatten())
                        best_intercept = self.y_mean - intercept_correction
                        # ★★★★★★★★★★★★★★★★★★★★★★

            except torch.linalg.LinAlgError:
                continue

        return best_rmse, best_model_recipe_tuple, best_coeffs, best_intercept

    def fit(self):
        print(f"***************** Starting SISSO Regressor (GPU Backend) *****************")

        residual = self.y_centered
        so_candidate_pool = []
        n_samples = self.y.shape[0]

        for i in range(1, self.n_term + 1):
            start_time = time.time()
            print(f"\n===== Searching for {i}-term models =====")

            recipes_to_screen = [r for r in self.all_recipes if r not in so_candidate_pool]
            top_k_recipes = self._run_sis(residual, recipes_to_screen)
            so_candidate_pool.extend(top_k_recipes)
            print(f"SIS selected {len(top_k_recipes)} new features. Pool size: {len(so_candidate_pool)}")

            rmse, model_recipes, coeffs, intercept = self._run_so_exhaustive(so_candidate_pool, i)

            if model_recipes:
                self.best_models[i] = {"rmse": rmse, "recipes": model_recipes, "coeffs": coeffs, "intercept": intercept}
                y_pred = intercept
                for j, recipe in enumerate(self.best_models[i]["recipes"]):
                    y_pred += self.best_models[i]["coeffs"].flatten()[j] * torch.nan_to_num(self.executor.execute(recipe))
                residual = self.y - y_pred

                print(f"Best {i}-term model found. RMSE: {rmse:.6f}")
                print(f"Equation: {self._format_equation(model_recipes, coeffs, intercept)}")
            else:
                print(f"No valid model found for term {i}.")
            print(f"Time for {i}-term search: {time.time() - start_time:.2f} seconds")

        if not self.best_models:
            print("Fit process finished, but no valid models were found.")
            return None

        best_model = min(self.best_models.values(), key=lambda m: m["rmse"])
        r2 = 1.0 - (best_model["rmse"] ** 2 * n_samples) / torch.sum((self.y_centered[~torch.isnan(self.y_centered)]) ** 2).item()
        final_equation = self._format_equation(best_model["recipes"], best_model["coeffs"], best_model["intercept"])

        return best_model["rmse"], final_equation, r2, self.best_models
