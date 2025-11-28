# # feature_generator.py (NumPy/SciPy version)
# import time
# from itertools import combinations
# from typing import Dict, List, Set

# import numpy as np

# from .executor import RecipeExecutor
# from .recipe import BinaryOperator, FeatureRecipe, Operator, UnaryOperator


# class FeatureGenerator:
#     def __init__(self, base_feature_names: List[str], operators: Dict[str, Operator]):
#         self.base_feature_names = base_feature_names
#         self.operators = operators
#         self.base_recipes = [FeatureRecipe(op=operators["base"], base_feature_index=i) for i in range(len(base_feature_names))]

#     def expand_full(self, n_expansion: int) -> List[FeatureRecipe]:
#         """
#         レベルワイズSISをオフにした場合のメソッド。
#         n_expansionレベルまでのすべてのレシピを一括で生成する。
#         """
#         print(f"*** Starting Full Recipe Generation (Level-wise SIS: OFF) ***")
#         all_recipes: Set[FeatureRecipe] = set(self.base_recipes)

#         recipes_at_level = [set(self.base_recipes)]

#         for i in range(1, n_expansion + 1):
#             start_time = time.time()
#             prev_all_recipes = set.union(*recipes_at_level)
#             newly_generated = self._generate_next_level(prev_all_recipes, recipes_at_level[-1])

#             recipes_at_level.append(newly_generated)
#             all_recipes.update(newly_generated)

#             print(f"Level {i}: Generated {len(newly_generated)} new recipes. Total unique recipes: {len(all_recipes)}. Time: {time.time() - start_time:.2f}s")

#         return list(all_recipes)

#     def expand_with_levelwise_sis(self, n_expansion: int, k_per_level: int, executor: RecipeExecutor, y_target: np.ndarray) -> List[FeatureRecipe]:
#         """
#         レベルワイズSISをオンにした場合のメソッド。
#         """
#         print(f"*** Starting Level-wise Recipe Generation (Level-wise SIS: ON, k_per_level={k_per_level}) ***")
#         all_promising_recipes: Set[FeatureRecipe] = set(self.base_recipes)
#         recipes_at_prev_level: Set[FeatureRecipe] = set(self.base_recipes)

#         for i in range(1, n_expansion + 1):
#             start_time = time.time()
#             newly_generated_recipes = self._generate_next_level(all_promising_recipes, recipes_at_prev_level)

#             if not newly_generated_recipes:
#                 print(f"Level {i}: No new recipes generated. Stopping expansion.")
#                 break

#             promising_new_recipes = self._run_sis(list(newly_generated_recipes), executor, y_target, k_per_level)

#             all_promising_recipes.update(promising_new_recipes)
#             recipes_at_prev_level = set(promising_new_recipes)

#             print(f"Level {i}: Generated {len(newly_generated_recipes)}, selected top {len(promising_new_recipes)}. Total promising: {len(all_promising_recipes)}. Time: {time.time() - start_time:.2f}s")

#         return list(all_promising_recipes)

#     def _generate_next_level(self, all_recipes: Set[FeatureRecipe], prev_level_recipes: Set[FeatureRecipe]) -> Set[FeatureRecipe]:
#         next_level_recipes: Set[FeatureRecipe] = set()
#         binary_ops = [op for op in self.operators.values() if isinstance(op, BinaryOperator)]
#         unary_ops = [op for op in self.operators.values() if isinstance(op, UnaryOperator)]

#         for op in binary_ops:
#             for r1 in all_recipes:
#                 for r2 in prev_level_recipes:
#                     if r1 != r2:
#                         next_level_recipes.add(FeatureRecipe(op=op, inputs=(r1, r2)))
#         for op in unary_ops:
#             for r in prev_level_recipes:
#                 next_level_recipes.add(FeatureRecipe(op=op, inputs=(r,)))

#         return next_level_recipes - (all_recipes | prev_level_recipes)

#     def _run_sis(self, recipes: List[FeatureRecipe], executor: RecipeExecutor, target: np.ndarray, k: int) -> List[FeatureRecipe]:
#         if not recipes:
#             return []
#         scores = []
#         for recipe in recipes:
#             array = executor.execute(recipe)
#             valid = ~np.isnan(array) & ~np.isnan(target)
#             if valid.sum() < 2:
#                 scores.append(0.0)
#                 continue
#             valid_f, valid_t = array[valid], target[valid]
#             mean, std = np.mean(valid_f), np.std(valid_f)
#             if std > 1e-8:
#                 # np.dotを使用して相関を計算
#                 scores.append(np.abs(np.dot(valid_t - np.mean(valid_t), (valid_f - mean) / std)))
#             else:
#                 scores.append(0.0)

#         # np.argsortを使用してスコアの高いインデックスを取得
#         return [recipes[i] for i in np.argsort(scores)[::-1][:k]]

# feature_generator.py (Dependency on executor removed)
import time
from itertools import combinations

# RecipeExecutorの型ヒントのためだけなので、前方参照を使うか、Anyを使う
from typing import Any, Dict, List, Set

import numpy as np

from .recipe import BinaryOperator, FeatureRecipe, Operator, UnaryOperator

# from .executor import RecipeExecutor # ★★★ この行を削除 ★★★


class FeatureGenerator:
    def __init__(self, base_feature_names: List[str], operators: Dict[str, Operator]):
        self.base_feature_names = base_feature_names
        self.operators = operators
        self.base_recipes = [FeatureRecipe(op=operators["base"], base_feature_index=i) for i in range(len(base_feature_names))]

    # ... (expand_full, _generate_next_levelメソッドは変更なし) ...
    def expand_full(self, n_expansion: int) -> List[FeatureRecipe]:
        """
        レベルワイズSISをオフにした場合のメソッド。
        n_expansionレベルまでのすべてのレシピを一括で生成する。
        """
        print(f"*** Starting Full Recipe Generation (Level-wise SIS: OFF) ***")
        all_recipes: Set[FeatureRecipe] = set(self.base_recipes)

        recipes_at_level = [set(self.base_recipes)]

        for i in range(1, n_expansion + 1):
            start_time = time.time()
            prev_all_recipes = set.union(*recipes_at_level)
            newly_generated = self._generate_next_level(prev_all_recipes, recipes_at_level[-1])

            recipes_at_level.append(newly_generated)
            all_recipes.update(newly_generated)

            print(f"Level {i}: Generated {len(newly_generated)} new recipes. Total unique recipes: {len(all_recipes)}. Time: {time.time() - start_time:.2f}s")

        return list(all_recipes)

    # ★★★ シグネチャを変更: executorを引数として受け取る ★★★
    def expand_with_levelwise_sis(self, n_expansion: int, k_per_level: int, executor: Any, y_target: np.ndarray) -> List[FeatureRecipe]:
        print(f"*** Starting Level-wise Recipe Generation (Level-wise SIS: ON, k_per_level={k_per_level}) ***")
        all_promising_recipes: Set[FeatureRecipe] = set(self.base_recipes)
        recipes_at_prev_level: Set[FeatureRecipe] = set(self.base_recipes)

        for i in range(1, n_expansion + 1):
            start_time = time.time()
            newly_generated_recipes = self._generate_next_level(all_promising_recipes, recipes_at_prev_level)

            if not newly_generated_recipes:
                print(f"Level {i}: No new recipes generated. Stopping expansion.")
                break

            # ★★★ executorを_run_sisに渡す ★★★
            promising_new_recipes = self._run_sis(list(newly_generated_recipes), executor, y_target, k_per_level)

            all_promising_recipes.update(promising_new_recipes)
            recipes_at_prev_level = set(promising_new_recipes)

            print(f"Level {i}: Generated {len(newly_generated_recipes)}, selected top {len(promising_new_recipes)}. Total promising: {len(all_promising_recipes)}. Time: {time.time() - start_time:.2f}s")

        return list(all_promising_recipes)

    def _generate_next_level(self, all_recipes: Set[FeatureRecipe], prev_level_recipes: Set[FeatureRecipe]) -> Set[FeatureRecipe]:
        next_level_recipes: Set[FeatureRecipe] = set()
        binary_ops = [op for op in self.operators.values() if isinstance(op, BinaryOperator)]
        unary_ops = [op for op in self.operators.values() if isinstance(op, UnaryOperator)]

        for op in binary_ops:
            for r1 in all_recipes:
                for r2 in prev_level_recipes:
                    if r1 != r2:
                        next_level_recipes.add(FeatureRecipe(op=op, inputs=(r1, r2)))
        for op in unary_ops:
            for r in prev_level_recipes:
                next_level_recipes.add(FeatureRecipe(op=op, inputs=(r,)))

        return next_level_recipes - (all_recipes | prev_level_recipes)

    # ★★★ シグネチャを変更し、NumPy/PyTorch両対応にする ★★★
    def _run_sis(self, recipes: List[FeatureRecipe], executor: Any, target: np.ndarray, k: int) -> List[FeatureRecipe]:
        if not recipes:
            return []

        # バックエンド（NumPy/PyTorch）を判定
        is_torch = "torch" in str(type(target))

        scores = []
        for recipe in recipes:
            array = executor.execute(recipe)

            # NumPy/PyTorch共通の操作
            if is_torch:
                import torch

                valid = ~torch.isnan(array) & ~torch.isnan(target)
                if valid.sum() < 2:
                    scores.append(0.0)
                    continue
                valid_f, valid_t = array[valid], target[valid]
                mean, std = valid_f.mean(), valid_f.std()
                if std > 1e-8:
                    scores.append(torch.abs(torch.dot(valid_t - valid_t.mean(), (valid_f - mean) / std)).item())
                else:
                    scores.append(0.0)
            else:  # NumPy
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

        return [recipes[i] for i in np.argsort(scores)[::-1][:k]]
