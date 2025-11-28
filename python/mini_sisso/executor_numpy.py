# executor_numpy.py
import numpy as np
from .recipe import FeatureRecipe

class RecipeExecutor:
    def __init__(self, base_features_array: np.ndarray):
        self.base_features = base_features_array
        self._cache = {}

    def execute(self, recipe: FeatureRecipe) -> np.ndarray:
        recipe_hash = hash(recipe)
        if recipe_hash in self._cache:
            return self._cache[recipe_hash]

        if recipe.op.name == 'base':
            result = self.base_features[:, recipe.base_feature_index]
        else:
            input_arrays = [self.execute(inp) for inp in recipe.inputs]
            try:
                # ★★★ 警告を抑制し、計算後に不正値を処理する ★★★
                with np.errstate(all='ignore'):
                    result = recipe.op.np_func(*input_arrays)
                
                # inf/nan を有限の最大値や0に置き換える（ロバスト化）
                if not np.isfinite(result).all():
                    # float64の最大値より少し小さい値でクリップ（オーバーフロー対策）
                    max_val = 1e100 
                    result = np.nan_to_num(result, nan=0.0, posinf=max_val, neginf=-max_val)
            except Exception:
                result = np.zeros(self.base_features.shape[0])
        
        if result.ndim == 0:
            result = np.full(self.base_features.shape[0], result)

        self._cache[recipe_hash] = result
        return result
