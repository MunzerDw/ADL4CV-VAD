import sys
import os
import json

from PIL import Image
import numpy as np
import h5py

dataset = sys.argv[1]
model = sys.argv[2]
config = sys.argv[3]

assert dataset in ['mnist', 'fashion_mnist']

"""
print some comparisons of masked data and predicted data
"""

# get masked amount to get appropriate masked data

model_folder = os.path.join('model_saves', 'pytorch', model, f'config_{config}')
model_args = os.path.join(model_folder, 'args.json')

args = json.load(open(model_args, 'r'))
ratio = args['masked']
print(ratio)

if ratio is not None:
    data_file = os.path.join('data', dataset, f'{dataset}_{ratio}.h5')
    with h5py.File(data_file, 'r') as f:
        test_data = f['test_y'][:]
        true_data = f['test_y_true'][:]
else:
    data_file = os.path.join('data', dataset, f'{dataset}.h5')
    with h5py.File(data_file, 'r') as f:
        test_data = f['test_x'][:]
        true_data = test_data

# load predictions
pred_file = os.path.join(model_folder, 'pred.h5')

with h5py.File(pred_file, 'r') as f:
    preds = f['y_pred'][:]

print(test_data.shape)
test_data = test_data.reshape((-1, 28, 28))
true_data = true_data.reshape((-1, 28, 28))
print(test_data.dtype)
preds = preds.reshape((-1, 28, 28))

print(preds.shape)
print(test_data.shape)

# some statistics
print(np.count_nonzero(preds < 0))
print(np.count_nonzero(preds > 1))
print(np.count_nonzero(np.logical_and(preds >= 0, preds <= 1)))

print(preds.min(), preds.max())
print(test_data.min(), test_data.max())

print(test_data.shape)
print(preds.shape)

# unmask preds?

# plot, with masked values as grey??
plot_folder = os.path.join('plot', model, config)
if not os.path.isdir(plot_folder):
    os.makedirs(plot_folder)

for i in range(10):
    in_arr = test_data[i]
    in_arr[in_arr < 0] = 100
    in_arr = in_arr.astype(np.uint8)
    input_image = Image.fromarray(in_arr)
    input_image.save(os.path.join(plot_folder, f'in{i}.jpg'))

    out_arr = np.clip(preds[i] * 255, 0, 255).astype(np.uint8)
    out_image = Image.fromarray(out_arr)
    out_image.save(os.path.join(plot_folder, f'out{i}.jpg'))

    true_arr = true_data[i]
    true_arr[true_arr < 0] = 100
    true_arr = true_arr.astype(np.uint8)
    true_image = Image.fromarray(true_arr)
    true_image.save(os.path.join(plot_folder, f'true{i}.jpg'))

