# recipe.py (NumPy version)
from typing import Callable, List, Tuple

import numpy as np


class FeatureRecipe:
    base_feature_names: List[str] = []

    def __init__(self, op: "Operator", inputs: Tuple["FeatureRecipe", ...] = (), base_feature_index: int = -1):
        self.op, self.inputs, self.base_feature_index = op, inputs, base_feature_index
        if op.is_commutative and len(inputs) > 1:
            self._hash = hash((self.op.name, tuple(sorted(self.inputs, key=hash))))
        else:
            self._hash = hash((self.op.name, self.inputs, self.base_feature_index))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, FeatureRecipe) and self._hash == other._hash

    def __repr__(self):
        return self.op.format_string(self.inputs, self.base_feature_index, self.base_feature_names)


class Operator:
    def __init__(self, name: str, np_func: Callable, is_commutative: bool = False):
        self.name, self.np_func, self.is_commutative = name, np_func, is_commutative

    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        raise NotImplementedError


class UnaryOperator(Operator):
    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        return f"{self.name}({repr(inputs[0])})"


class BinaryOperator(Operator):
    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        return f"({repr(inputs[0])} {self.name} {repr(inputs[1])})"


class BaseFeatureOperator(Operator):
    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        if feature_names and 0 <= base_feature_index < len(feature_names):
            return feature_names[base_feature_index]
        return f"f{base_feature_index}"


# PyTorch関数をNumPy関数に置き換え
OPERATORS = {
    "base": BaseFeatureOperator("base", lambda x: x),
    "+": BinaryOperator("+", np.add, is_commutative=True),
    "-": BinaryOperator("-", np.subtract),
    "*": BinaryOperator("*", np.multiply, is_commutative=True),
    "/": BinaryOperator("/", np.divide),
    "exp": UnaryOperator("exp", np.exp),
    "log": UnaryOperator("log", np.log),
    "sin": UnaryOperator("sin", np.sin),
    "cos": UnaryOperator("cos", np.cos),
    "sqrt": UnaryOperator("sqrt", lambda x: np.sqrt(np.abs(x))),
    "pow2": UnaryOperator("^2", lambda x: np.power(x, 2)),
    "pow3": UnaryOperator("^3", lambda x: np.power(x, 3)),
    "inv": UnaryOperator("^-1", lambda x: 1.0 / x),  # np.reciprocalは整数型で0を返すため変更
}
