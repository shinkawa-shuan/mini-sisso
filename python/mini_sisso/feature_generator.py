# feature_generator.py
import time
from typing import List, Set, Dict, Any
import numpy as np
from .recipe import FeatureRecipe, Operator, UnaryOperator, BinaryOperator

class FeatureGenerator:
    def __init__(self, base_feature_names: List[str], operators: Dict[str, Operator], use_split_selection: bool = True):
        self.base_feature_names = base_feature_names
        self.operators = operators
        self.use_split_selection = use_split_selection
        self.base_recipes = [
            FeatureRecipe(op=operators['base'], base_feature_index=i)
            for i in range(len(base_feature_names))
        ]

    def expand_full(self, n_expansion: int) -> List[FeatureRecipe]:
        print(f"*** Starting Full Recipe Generation (Level-wise SIS: OFF) ***")
        all_recipes_set: Set[FeatureRecipe] = set(self.base_recipes)
        recipes_at_level = [set(self.base_recipes)]

        for i in range(1, n_expansion + 1):
            start_time = time.time()
            prev_all = set.union(*recipes_at_level)
            newly_generated = self._generate_next_level(prev_all, recipes_at_level[-1])
            
            recipes_at_level.append(newly_generated)
            all_recipes_set.update(newly_generated)
            print(f"Level {i}: Generated {len(newly_generated)}. Total: {len(all_recipes_set)}. Time: {time.time() - start_time:.2f}s")

        return sorted(list(all_recipes_set), key=lambda r: str(r))

    def expand_with_levelwise_sis(self, n_expansion: int, k_per_level: int, executor: Any, y_target: np.ndarray) -> List[FeatureRecipe]:
        print(f"*** Starting Level-wise Recipe Generation (Level-wise SIS: ON, k={k_per_level}, SplitSelection={self.use_split_selection}) ***")
        
        all_promising_recipes = sorted(self.base_recipes, key=lambda r: str(r))
        recipes_at_prev_level = sorted(self.base_recipes, key=lambda r: str(r))

        for i in range(1, n_expansion + 1):
            start_time = time.time()
            newly_generated_set = self._generate_next_level(set(all_promising_recipes), set(recipes_at_prev_level))
            
            if not newly_generated_set:
                print(f"Level {i}: No new recipes. Stopping.")
                break

            newly_generated_list = sorted(list(newly_generated_set), key=lambda r: str(r))
            promising_new_recipes = self._run_sis(newly_generated_list, executor, y_target, k_per_level)
            
            combined = set(all_promising_recipes)
            combined.update(promising_new_recipes)
            all_promising_recipes = sorted(list(combined), key=lambda r: str(r))
            
            recipes_at_prev_level = sorted(promising_new_recipes, key=lambda r: str(r))

            print(f"Level {i}: Gen {len(newly_generated_set)}, Sel {len(promising_new_recipes)}. Total: {len(all_promising_recipes)}. Time: {time.time() - start_time:.2f}s")
            
        return all_promising_recipes

    def _generate_next_level(self, all_recipes: Set[FeatureRecipe], prev_level_recipes: Set[FeatureRecipe]) -> Set[FeatureRecipe]:
        next_level = set()
        binary_ops = [op for op in self.operators.values() if isinstance(op, BinaryOperator)]
        unary_ops = [op for op in self.operators.values() if isinstance(op, UnaryOperator)]
        
        sorted_all = sorted(list(all_recipes), key=lambda r: str(r))
        sorted_prev = sorted(list(prev_level_recipes), key=lambda r: str(r))

        for op in binary_ops:
            for r1 in sorted_all:
                for r2 in sorted_prev:
                    if r1 != r2: next_level.add(FeatureRecipe(op=op, inputs=(r1, r2)))
        for op in unary_ops:
            for r in sorted_prev:
                next_level.add(FeatureRecipe(op=op, inputs=(r,)))
        
        return next_level - (all_recipes | prev_level_recipes)

    def _run_sis(self, recipes: List[FeatureRecipe], executor: Any, target: np.ndarray, k: int) -> List[FeatureRecipe]:
        if not recipes: return []
        is_torch = 'torch' in str(type(target))
        
        scores = []
        for recipe in recipes:
            array = executor.execute(recipe)
            with np.errstate(all='ignore'):
                if is_torch:
                    import torch
                    valid = ~torch.isnan(array) & ~torch.isnan(target)
                    if valid.sum() < 2: scores.append(0.0); continue
                    vf, vt = array[valid], target[valid]
                    mean, std = vf.mean(), vf.std()
                    if std > 1e-9: scores.append(torch.abs(torch.dot(vt - vt.mean(), (vf - mean) / std)).item())
                    else: scores.append(0.0)
                else: # NumPy
                    valid = ~np.isnan(array) & ~np.isnan(target) & ~np.isinf(array)
                    if valid.sum() < 2: scores.append(0.0); continue
                    vf, vt = array[valid], target[valid]
                    mean, std = np.mean(vf), np.std(vf)
                    if std > 1e-9: scores.append(np.abs(np.dot(vt - np.mean(vt), (vf - mean) / std)))
                    else: scores.append(0.0)
        
        recipe_score_pairs = list(zip(recipes, scores))
        recipe_score_pairs.sort(key=lambda x: (-x[1], str(x[0])))
        
        # --- Logic Branch ---
        if not self.use_split_selection:
            # Traditional Top-K
            return [r for r, s in recipe_score_pairs[:k]]
        else:
            # Split Selection (Balanced)
            selected_recipes = []
            selected_set = set()
            
            k_split = k // 2 + 1
            count_unary = 0
            count_binary = 0
            
            # Pass 1: Quota filling
            for r, s in recipe_score_pairs:
                is_base = (len(r.inputs) == 0) or (r.op.name == 'base')
                is_unary = (len(r.inputs) == 1) and not is_base
                is_binary = (len(r.inputs) == 2)
                
                add = False
                if is_base: add = True
                elif is_unary and count_unary < k_split:
                    count_unary += 1; add = True
                elif is_binary and count_binary < k_split:
                    count_binary += 1; add = True
                
                if add:
                    selected_recipes.append(r)
                    selected_set.add(r)
            
            # Pass 2: Fill remaining
            for r, s in recipe_score_pairs:
                if len(selected_recipes) >= k: break
                if r not in selected_set:
                    selected_recipes.append(r)
                    selected_set.add(r)
                    
            return selected_recipes
