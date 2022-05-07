import json
import numpy as np

from scipy.stats import loguniform, uniform
from numpy import linspace, logspace
from .defs import TuneType


def get_params_distr(conf_path: str, tune_type: TuneType):
    if tune_type == TuneType.manual:
        return

    with open(conf_path) as json_file:
        hyperparams_conf = json.load(json_file)
        params_distr = {}

        for obj_in_pipe in hyperparams_conf:
            comp_name = obj_in_pipe[0]
            params_distr[comp_name] = {}

            for param_in_obj in obj_in_pipe[1]:
                param_name = param_in_obj[1]
                if obj_in_pipe[1]['type'] == 'range':
                    min_val = obj_in_pipe[1]['min']
                    max_val = obj_in_pipe[1]['max']

                    if tune_type == TuneType.auto_random:
                        if max_val - min_val > 1000:
                            params_distr[comp_name] = loguniform.stats(min_val, max_val)
                        else:
                            params_distr[comp_name] = uniform.stats(min_val, max_val)
                    elif tune_type == TuneType.auto_grid:
                        if max_val - min_val > 1000:
                            params_distr[comp_name] = np.logspace(min_val, max_val, 50)
                        else:
                            params_distr[comp_name] = np.linspace(min_val, max_val, 50)

                elif obj_in_pipe[1]['type'] == 'list':
                    params_distr[comp_name] = obj_in_pipe['list']

        return params_distr


def parse_hyperparams(click_in):
    # парсим экстра-параметры
    hyperparams_dict = {}
    for hyperparam in click_in:
        # конвертация типов
        curr_split = hyperparam.split('=')
        if len(curr_split) != 2:
            continue

        curr_split[1] = eval(curr_split[1]) # Выполняет оценку выражения по строке
        hyperparams_dict.update([curr_split])

    if hyperparams_dict.get('c') is not None:
        hyperparams_dict['C'] = hyperparams_dict.pop('c')  # единственный параметр в Uppercase для LogReg

    return hyperparams_dict