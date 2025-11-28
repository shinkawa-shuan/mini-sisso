# executor_torch.py
import torch

from .recipe import FeatureRecipe


class RecipeExecutorTorch:
    def __init__(self, base_features_tensor: torch.Tensor):
        self.base_features = base_features_tensor
        self.device = base_features_tensor.device
        self._cache = {}

    def execute(self, recipe: FeatureRecipe) -> torch.Tensor:
        recipe_hash = hash(recipe)
        if recipe_hash in self._cache:
            return self._cache[recipe_hash]

        if recipe.op.name == "base":
            result = self.base_features[:, recipe.base_feature_index]
        else:
            input_tensors = [self.execute(inp) for inp in recipe.inputs]
            try:
                result = recipe.op.torch_func(*input_tensors)
            except Exception:
                result = torch.full_like(self.base_features[:, 0], float("nan"))

        if result.ndim == 0:
            result = torch.full_like(self.base_features[:, 0], result.item())

        self._cache[recipe_hash] = result
        return result
