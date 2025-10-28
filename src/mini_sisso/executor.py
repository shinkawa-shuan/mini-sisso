# executor.py (NumPy version)
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

        if recipe.op.name == "base":
            result = self.base_features[:, recipe.base_feature_index]
        else:
            input_arrays = [self.execute(inp) for inp in recipe.inputs]
            try:
                # np.divideで0除算が起きるとwarningが出るのを抑制
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = recipe.op.np_func(*input_arrays)
                if not isinstance(result, np.ndarray):  # For safety
                    result = np.array(result)
            except Exception:
                result = np.full(self.base_features.shape[0], np.nan)

        # 結果がスカラーの場合、配列に変換
        if result.ndim == 0:
            result = np.full(self.base_features.shape[0], result)

        self._cache[recipe_hash] = result
        return result
