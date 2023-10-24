import argparse
import pickle
from pathlib import Path
import json
import numpy as np
import random
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help = 'Experiment name for downstream tracking purposes')
parser.add_argument('--output_dir', type=str, required = True)

# example hyperparameters
parser.add_argument('--dataset', type = str, choices = ['ds1', 'ds2'], required = True)
parser.add_argument('--lr', type = float, default = 1e-2)
parser.add_argument('--use_early_stopping', action = 'store_true') 
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

hparams = vars(args)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents = True, exist_ok=True)

# print out arguments to screen
print('Args:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

# REQUIRED: save hparams to file; necessary to match results with experiment later
with (output_dir/'args.json').open('w') as f:
    json.dump(hparams, f, indent = 4)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

## add your training + evaluation code here


# REQUIRED: save all results to the output_dir
dummy_results = {
    'train_acc': 0.9,
    'test_acc': 0.8
}
pickle.dump(dummy_results, (output_dir/'results.pkl').open('wb'))

# REQUIRED: dummy file so that the launcher script knows this job is done
with (output_dir/'done').open('w') as f:
    f.write('done')