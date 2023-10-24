import os
import json
import numpy as np
from pathlib import Path
from itertools import product

### write your experiments here
class SampleExp1():
    script_name = 'train'
    def get_hparams(self):
        # grid of hparams to sweep over
        # the combination function will take the cartesian product of these
        hparams = { 
            'exp_name': ['exp1'],
            'dataset': ['ds1', 'ds2'],
            'lr': list(10.**np.arange(-5, 1, 1)),
            'use_early_stopping': [True, False], # when value is bool, will add --use_early_stopping if True, nothing if False
            'seed': [0, 1, 2]
        }
        return combinations(hparams)
    
class SampleExp2():
    script_name = 'train'
    def get_hparams(self):
        # more complicated experiment, with multiple sub experiments
        # here, each sub experiment has a different grid of "dataset", "use_early_stopping" and "lr"
        # but they share a common grid of "exp_name" and "seed"
        hparams = { 
            'exp_name': ['exp1'],
            'seed': [0, 1, 2],
            'dataset': {
                'sub_exp1': ['ds1'],
                'sub_exp2': ['ds2'],
            },
            'use_early_stopping':{
                'sub_exp1': [True],
                'sub_exp2': [False],
            },
            'lr': {
                'sub_exp1': [1e-3, 1e-4],
                'sub_exp2': [1e-1, 1e-2]
            }
        }
        return combinations(hparams)
    
class SampleExp3():
    script_name = 'train'
    def get_hparams(self):
        # this grid is equivalent to the one in SampleExp2
        common_hparams = { 
            'exp_name': ['exp1'],
            'seed': [0, 1, 2]
        }
        sub_exp1 =  { 
                'dataset': ['ds1'],
                'use_early_stopping': [True],
                'lr': [1e-3, 1e-4]
            }
        sub_exp2 = {
                'dataset': ['ds2'],
                'use_early_stopping': [False],
                'lr': [1e-1, 1e-2]
        }
        return combinations({**common_hparams, **sub_exp1}) + combinations({**common_hparams, **sub_exp2})


### helper functions
def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def combinations(grid):
    sub_exp_names = set()
    for i in grid:
        if isinstance(grid[i], dict):
            for j in grid[i]:
                sub_exp_names.add(j)
    if len(sub_exp_names) == 0:
        return combinations_base(grid)

    for i in grid:
        if isinstance(grid[i], dict):
            assert set(list(grid[i].keys())) == sub_exp_names, f'{i} does not have all sub exps ({sub_exp_names})'
    args = []
    for n in sub_exp_names:
        sub_grid = grid.copy()
        for i in sub_grid:
            if isinstance(sub_grid[i], dict):
                sub_grid[i] = sub_grid[i][n]
        args += combinations_base(sub_grid)
    return args

def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].script_name

