# regressor_numpy.py
import time
import numpy as np
from typing import List, Tuple, Dict
from itertools import combinations
from scipy.linalg import lstsq as sp_lstsq
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import dcor
from .recipe import FeatureRecipe
from .executor_numpy import RecipeExecutor

class SissoRegressorNumPy:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutor, 
                 y: np.ndarray, so_method: str, selection_params: dict):
        # 入力されたレシピリストを即座にソートして固定
        self.all_recipes = sorted(all_recipes, key=lambda r: str(r))
        self.executor = executor
        self.y = y
        self.so_method = so_method
        self.selection_params = self._parse_selection_params(selection_params)
        self.y_mean, self.y_centered = np.mean(y), y - np.mean(y)
        self.best_models: Dict[int, dict] = {}
        
        # 再現性のための乱数シード取得
        self.random_state = self.selection_params.get('random_state', 42)

    def _parse_selection_params(self, params: dict) -> dict:
        if params is None: params = {}
        params.setdefault('n_term', 2)
        params.setdefault('n_sis_features', 10)
        params.setdefault('alpha', 0.01)
        params.setdefault('n_features_to_select', 40)
        # LightGBMにrandom_stateを強制
        lgbm_params = params.get('lightgbm_params', {})
        lgbm_params.setdefault('random_state', 42)
        lgbm_params.setdefault('n_jobs', 1) # 再現性のためシングルスレッド推奨
        lgbm_params.setdefault('verbosity', -1)
        params['lightgbm_params'] = lgbm_params
        
        params.setdefault('n_global_sis_features', None)
        params.setdefault('collinearity_filter', None)
        params.setdefault('collinearity_threshold', 0.9)
        params.setdefault('coeff_threshold', 1e-5)
        params.setdefault('random_state', 42) # グローバルなシード設定
        return params

    def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: np.ndarray, intercept: float) -> str:
        if not recipes: return f"{intercept:+.6f}"
        return "".join(f"{c:+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten())) + f"{intercept:+.6f}"

    def _run_sis(self, target: np.ndarray, recipes: List[FeatureRecipe], k: int) -> List[FeatureRecipe]:
        if not recipes: return []
        scores = []
        for r in recipes:
            arr = self.executor.execute(r)
            with np.errstate(all='ignore'):
                valid = ~np.isnan(arr) & ~np.isnan(target) & ~np.isinf(arr)
                if valid.sum() < 2: scores.append(0.0); continue
                vf, vt = arr[valid], target[valid]
                mean, std = np.mean(vf), np.std(vf)
                if std > 1e-9: scores.append(np.abs(np.dot(vt - np.mean(vt), (vf - mean) / std)))
                else: scores.append(0.0)
        
        # ★★★ 決定論的ソート ★★★
        pairs = list(zip(recipes, scores))
        pairs.sort(key=lambda x: (-x[1], str(x[0])))
        return [p[0] for p in pairs[:k]]

    def _filter_collinear_features(self, recipes: List[FeatureRecipe], method: str, threshold: float) -> List[FeatureRecipe]:
        print(f"--- Applying Collinearity Filter (method: {method}, threshold: {threshold}) ---")
        # recipesは既にソートされている前提
        
        X_list = []
        for r in recipes:
            val = self.executor.execute(r)
            # NaN/Inf処理
            val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            X_list.append(val)
        X = np.stack(X_list, axis=1)
        
        # 念のため平均補完
        if np.isnan(X).any(): 
             col_mean = np.nanmean(X, axis=0)
             inds = np.where(np.isnan(X))
             X[inds] = np.take(col_mean, inds[1])
        
        removed_indices = set()
        n_features = len(recipes)
        
        # 相関行列を一括計算する方が高速だが、メモリ消費するためループ処理
        # 再現性のため、ループ順序は固定(0..n)
        for i in range(n_features):
            if i in removed_indices: continue
            for j in range(i + 1, n_features):
                if j in removed_indices: continue
                
                corr = 0.0
                try:
                    with np.errstate(all='ignore'):
                        if method == 'mi':
                            # random_stateを指定して再現性を確保
                            corr = mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j], random_state=self.random_state)[0]
                        elif method == 'dcor':
                            # dcorは決定論的だが、念のため型変換
                            corr = dcor.distance_correlation(X[:, i], X[:, j])
                        else:
                            corr = np.abs(np.corrcoef(X[:, i], X[:, j])[0, 1])
                except Exception:
                    corr = 0.0 # エラー時は無相関扱い
                
                if np.isnan(corr): corr = 0.0
                
                if corr > threshold:
                    removed_indices.add(j)

        final_recipes = [r for i, r in enumerate(recipes) if i not in removed_indices]
        print(f"Collinearity filter removed {len(removed_indices)} features. Kept {len(final_recipes)}.")
        return final_recipes

    def _get_final_model_np(self, recipes_list: list) -> Tuple:
        if not recipes_list: return float('inf'), None, None, None
        # ... (変更なし) ...
        X = np.stack([self.executor.execute(r) for r in recipes_list], axis=1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) # ロバスト化
        
        X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
        X_std[X_std < 1e-9] = 1.0 # 閾値を少し厳しく
        try:
            c_std, res, _, _ = sp_lstsq((X - X_mean) / X_std, self.y_centered, check_finite=False)
            rmse = np.sqrt(res / self.y.shape[0]) if isinstance(res, (int, float, np.float64)) and res >= 0 else (np.sqrt(res[0]/self.y.shape[0]) if hasattr(res, '__len__') and len(res)>0 else float('inf'))
            coeffs, intercept = c_std / X_std, self.y_mean - np.dot(c_std / X_std, X_mean)
            return rmse, tuple(recipes_list), coeffs, intercept
        except Exception: return float('inf'), None, None, None

    def _run_feature_selector(self, candidate_recipes: List[FeatureRecipe], method: str, params: dict):
        # ... (変更なし) ...
        # _run_sis や _filter_collinear_features は修正版が呼ばれる
        
        print(f"--- Running Feature Selection with {method.upper()} ---")
        if params['n_global_sis_features']:
            print(f"Performing Global SIS, selecting top {params['n_global_sis_features']} features...")
            candidate_recipes = self._run_sis(self.y_centered, candidate_recipes, k=params['n_global_sis_features'])
        
        if params['collinearity_filter']:
            candidate_recipes = self._filter_collinear_features(candidate_recipes, params['collinearity_filter'], params['collinearity_threshold'])

        # Lasso/LightGBM用の行列作成
        X_candidates, valid_recipes = [], []
        for r in candidate_recipes:
            arr = self.executor.execute(r)
            with np.errstate(all='ignore'):
                valid = ~np.isnan(arr) & ~np.isinf(arr)
                if valid.sum() > 10: # ある程度のデータ点が必要
                    vf = arr[valid]
                    if np.std(vf) > 1e-9:
                        # 全体を埋める
                        clean_arr = np.nan_to_num(arr, nan=np.mean(vf))
                        # 標準化
                        clean_arr = (clean_arr - np.mean(clean_arr)) / (np.std(clean_arr) + 1e-9)
                        X_candidates.append(clean_arr)
                        valid_recipes.append(r)
        
        if not X_candidates: print(f"No valid features for {method}."); return
        X_matrix = np.stack(X_candidates, axis=1)

        # Lassoにもrandom_stateを渡す
        if method == 'lasso':
            model = Lasso(alpha=params['alpha'], max_iter=10000, random_state=self.random_state, tol=1e-4)
            model.fit(X_matrix, self.y_centered)
            importances = np.abs(model.coef_)
            selected_indices = np.where(importances > params['coeff_threshold'])[0]
        elif method == 'lightgbm':
            # lightgbm_paramsには既にrandom_stateが含まれているはず
            model = lgb.LGBMRegressor(**params['lightgbm_params'])
            model.fit(X_matrix, self.y_centered)
            importances = model.feature_importances_
            # 決定論的ソート
            indices_scores = list(enumerate(importances))
            indices_scores.sort(key=lambda x: (-x[1], x[0])) # スコア降順、インデックス昇順
            selected_indices = [i for i, score in indices_scores if score > 0][:params['n_features_to_select']]

        if len(selected_indices) == 0: print(f"{method.upper()} selected 0 features."); return
        
        final_recipes = [valid_recipes[i] for i in selected_indices]
        print(f"{method.upper()} selected {len(final_recipes)} features.")
        
        rmse, recipes_tuple, coeffs, intercept = self._get_final_model_np(final_recipes)
        if recipes_tuple:
            n_terms = len(recipes_tuple)
            self.best_models[n_terms] = {'rmse': rmse, 'recipes': recipes_tuple, 'coeffs': coeffs, 'intercept': intercept}
            print(f"Found {n_terms}-term model via {method.upper()}: RMSE={rmse:.6f}, Eq: {self._format_equation(recipes_tuple, coeffs, intercept)}")

    def fit(self):
        # ... (変更なし) ...
        # 既存のfitメソッド内で、_run_sisなどを呼び出しているので自動的に修正が反映される
        print(f"***************** Starting SISSO Regressor (NumPy/SciPy Backend, Method: {self.so_method}) *****************")
        
        if self.so_method == 'lasso' or self.so_method == 'lightgbm':
            self._run_feature_selector(self.all_recipes, self.so_method, self.selection_params)
        
        elif self.so_method == 'exhaustive':
            params = self.selection_params
            residual, pool = self.y_centered, []
            for i in range(1, params['n_term'] + 1):
                start_time = time.time()
                print(f"\n===== Searching for {i}-term models =====")
                # プールにないものをSISにかける
                candidates = [r for r in self.all_recipes if r not in pool] # rはFeatureRecipeオブジェクト
                # ここでソートしておくとさらに安全
                candidates.sort(key=lambda r: str(r))
                
                top_k = self._run_sis(residual, candidates, k=params['n_sis_features'])
                pool.extend(top_k)
                
                # 重複排除してソート（念のため）
                pool = sorted(list(set(pool)), key=lambda r: str(r))
                
                print(f"SIS selected {len(top_k)} new features. Pool size: {len(pool)}")

                combos = list(combinations(pool, i))
                print(f"--- Running SO for {i}-term models. Total combinations: {len(combos)} ---")
                
                best_rmse, best_model = float('inf'), None
                for combo in combos:
                    rmse, recipes, coeffs, intercept = self._get_final_model_np(list(combo))
                    if rmse < best_rmse:
                        best_rmse, best_model = rmse, {'r': recipes, 'c': coeffs, 'i': intercept}
                
                if best_model:
                    self.best_models[i] = {'rmse': best_rmse, 'recipes': best_model['r'], 'coeffs': best_model['c'], 'intercept': best_model['i']}
                    # 残差更新
                    X_best = np.stack([self.executor.execute(r) for r in best_model['r']], axis=1)
                    X_best = np.nan_to_num(X_best)
                    y_pred = best_model['i'] + np.sum(best_model['c'] * X_best, axis=1)
                    residual = self.y - y_pred
                    print(f"Best {i}-term model: RMSE={best_rmse:.6f}, Eq: {self._format_equation(best_model['r'], best_model['c'], best_model['i'])}")
                else:
                    print(f"No valid model found for term {i}.")
                print(f"Time: {time.time() - start_time:.2f} seconds")
        
        # 戻り値処理 (変更なし)
        if not self.best_models: return None
        best_model = min(self.best_models.values(), key=lambda m: m['rmse'])
        r2 = 1.0 - (best_model['rmse']**2 * self.y.shape[0]) / np.sum(self.y_centered[~np.isnan(self.y_centered)]**2)
        return best_model['rmse'], self._format_equation(best_model['recipes'], best_model['coeffs'], best_model['intercept']), r2, self.best_models
