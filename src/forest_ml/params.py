from typing import Any

# эти импорты нужны для корректного eval распределений параметров
from scipy.stats import loguniform, uniform  # noqa: F401
from numpy import linspace, logspace  # noqa: F401
import numpy as np  # noqa: F401


def parse_hyperparams(click_in: dict[str, str]) -> dict[str, Any]:
    # парсим экстра-параметры
    hyperparams_dict = {}
    for hyperparam in click_in:
        # конвертация типов
        curr_split = hyperparam.split("=", 1)
        if len(curr_split) != 2:
            continue

        # Выполняет оценку выражения по строке
        hyperparams_dict[curr_split[0]] = eval(curr_split[1])

    if hyperparams_dict.get("c") is not None:
        hyperparams_dict["C"] = hyperparams_dict.pop(
            "c"
        )  # единственный параметр в Uppercase для LogReg

    return hyperparams_dict
