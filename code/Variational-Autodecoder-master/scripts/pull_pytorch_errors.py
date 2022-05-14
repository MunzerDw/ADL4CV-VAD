import json
import sys
import os

import pandas as pd

config_name = sys.argv[1]

folder = os.path.join('model_saves', 'pytorch', config_name)

if not os.path.isdir(folder):
    raise ValueError("Run this script from the repo root directory. {} not found".format(folder))

configs = sorted(os.listdir(folder))
idxes = [int(f.split('_')[1]) for f in configs]

losses = []

for i in range(max(idxes) + 1):
    config_folder = 'config_' + str(i)
    path = os.path.join(folder, config_folder, 'results.json')
    pred = os.path.join(folder, config_folder, 'pred.h5')
    if not os.path.isfile(pred):
        losses.append({'config': i,})
        continue
    results = json.load(open(path, 'r'))

    # if type(results['train_loss'][-1]) is list:
    #     train_loss = results['train_loss'][-1][0]
    # else:
    #     train_loss = results['train_loss'][-1]

    row = {
        'config': i,
        'test_loss': results['final_test_loss'],
        'clean_loss': results['final_clean_loss'],
        'train_loss': results['train_loss'][-1][0],
    }
    if len(results['train_loss'][-1]) > 1:
        row['train_kl'] = results['train_loss'][-1][1]
        row['test_kl'] = results['test_loss'][-1][1]

    losses.append(row)
    # losses.append({
    #     'config': i,
    #     'train_loss': results['train_loss'][-1],
    #     'test_loss': results['final_test_loss'],
    #     'clean_loss': results['final_clean_loss']
    # })

df = pd.DataFrame(losses)
df = df.sort_values('config')

# print(df)

df.to_csv('results/{}_torch.csv'.format(config_name), index=False)
