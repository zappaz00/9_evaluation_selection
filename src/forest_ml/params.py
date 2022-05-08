import json
import numpy as np

from scipy.stats import loguniform, uniform
from numpy import linspace, logspace


def parse_hyperparams(click_in):
    # парсим экстра-параметры
    hyperparams_dict = {}
    for hyperparam in click_in:
        # конвертация типов
        curr_split = hyperparam.split('=', 1)
        if len(curr_split) != 2:
            continue

        curr_split[1] = eval(curr_split[1]) # Выполняет оценку выражения по строке
        hyperparams_dict.update([curr_split])

    if hyperparams_dict.get('c') is not None:
        hyperparams_dict['C'] = hyperparams_dict.pop('c')  # единственный параметр в Uppercase для LogReg

    return hyperparams_dict