

Implementation of [Variational Auto-Decoder](https://arxiv.org/pdf/1903.00840.pdf)
(A. Zadeh, Y.C. Lim, P. Liang, L.-P. Morency, 2019.). Code is implemented and made easy to run by Yao Chong Lim. Our paper shows that encoderless implementation of the AEVB algorithm (named in our paper as Variational Auto-Decoder - VAD) shows very promising performance in generative modeling from data with missingness (low and high missingness). We claim that even through rigorous training and hyperparameter search, missingness causes instabilities in the encoder which in turn harms the reconstruction or imputations performance of the decoder. We show that for a probabilistic decoder with only one mode, the approximate posterior disitrbution can be infered efficiently using only gradient ascend (or descned) without the need for MCMC sampling from the decoder input. This makes the process of inference faster and only a matter of gradient ascent (or descent) during test-time. 

![alt text](https://github.com/A2Zadeh/Variational-Autodecoder/blob/master/VAD.png)


# 1. Introduction #

Variational Auto-Decoder refers to **encoderless** implementation of the Auto-Encoding Variational Bayes (AEVB) Algorithm. As opposed to using an encoder to infer the parameters of the posterior in the latent space ![equation](https://latex.codecogs.com/gif.latex?q%28z%7Cx_i%29): 

By assuming a reparamterization of samples from certain distribution, for the probabilistic decoder, the process of inference can be simplified by gradient approaches: 

--> First - An arbitrary distribution (which is easy to sample from) is sampled ![equation](https://latex.codecogs.com/gif.latex?z%5Csim%20q%28z%7Cx_i%29). This distribution is assumed to be compatible with reparameterization trick in AEVB. One such distribution is a multivariate normal distribution ![equation](https://latex.codecogs.com/gif.latex?q%28z%7Cx_i%29%3A%3D%5Cmathcal%7BN%7D%28z%3B%5Cmu_i%2C%5CSigma_i%29). Other distributions with one mode exist and can be used as well. 

--> Second - After the sample(s, one or more samples) ![equation](https://latex.codecogs.com/gif.latex?z) is drawn, the sample is used as input to the decoder ![equation](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BF%7D%20%28%5Ccdot%3B%5Ctheta%29). 

--> Third - Training is done by gradient ascend (or descend) w.r.t ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta%2C%20%5Cmu_i%2C%20%5CSigma_i), inference (testing) is done by gradient ascend (or descend) w.r.t ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_i%2C%5CSigma_i). 

The above method finds a mixture model based on ![equation](https://latex.codecogs.com/gif.latex?%5C%7Bq%28z%7Cx_i%29%5C%7D%5E%7Bx_i%20%5Cin%20X%7D). Sampling from this mixture model should essentially generate the learned density of ![equation](https://latex.codecogs.com/gif.latex?x). This mixture model may or may not have desirable generative properties such as meaningful manifold walk, given a limited dataset ![equation](https://latex.codecogs.com/gif.latex?X). The VAE reparameterization trick can be used to enforce certain properties using another distribution - one example is unit multivariate gaussian. 

![alt text](https://github.com/A2Zadeh/Variational-Autodecoder/blob/master/Algorithm.png)

# 2. Results

We make comparisons between VAE and VAD (both example implementations of AEVB algorithm) in generative modeling from partial data (data with missigness). In the figures below, ![equation](https://latex.codecogs.com/gif.latex?r) indicates the missing ratio which changes between 0.1 to 0.9 (10% to 90%). In all the figures lower is better. Please refer to the paper for exact details of each figure. 

![alt text](https://github.com/A2Zadeh/Variational-Autodecoder/blob/master/Results.png)

The following demonstrates a comparison between VAD and VAE for the adversarial case where missingness ratio is different between train and test. Models are trained on the data with no missingness (data similar to Example Image) and tested on data with missingness (the missingness pattern is Missing Completely at Random - MCAR). This experiment can also be seen as inpainting from MCAR missingness.

![alt text](https://github.com/A2Zadeh/Variational-Autodecoder/blob/master/MNISTs.png)

## Practical Considerations 

The losses for the reconstruction and the loss for KL (reparameterization) may act in opposite directions; simiar to VAE, enforcing nice distributional properties may come at the cost of reconstruction inaccuracy for certain data points. The code allows you to balance between the reconstruction and the KL terms. As a general rule of thumb, more complex datasets may not conform to simple ![equation](https://latex.codecogs.com/gif.latex?q%28z%29), such as densities with one mode. Therefore, the reconstruction may be bad. Therefore, we also allow for dropping the reparameterization fully, thus the model only learns a mixture based on ![equation](https://latex.codecogs.com/gif.latex?%5C%7Bq%28z%7Cx_i%29%5C%7D%5E%7Bx_i%20%5Cin%20X%7D). Due to missingness in data, learning the ![equation](https://latex.codecogs.com/gif.latex?%5CSigma_i) may also be problematic hence it can be treated as a hyperparameter. 

# 3. Code
The rest of this readme contains details of how to run the code and the requirements for it.

## Requirements
- Python 3

See `requirements_vad.txt` for remaining dependencies. To install,
```
pip install -r requirements_vad.txt
```

## Obtaining the data
Obtain the data from [here](https://drive.google.com/file/d/1rZN3oWRvKUgJk-wIGm8r0Tj7cDSoZCcv/view?usp=sharing), and extract it to `data/`. Then your directory should look like
```
data/
- mnist/
- fashion-mnist/
... other datasets
```

## Generating hyperparameter configurations

For convenient grid search over hyperparameters, use the script at `configs/make_configs.py` to generate
a folder of JSON files representing different hyperparameter configurations.

Usage:
```
python configs/make_configs.py <config name>
```

## Training a model

- `pytorch/train_model.py`

Usage:
```
python pytorch/train_model.py train <dataset> <config name> <config number>
```

Optional arguments can be found using `python pytorch/train_model.py --help`.

## Test on missing

To test a trained model on partial data, use the `test_missing` option. `--missing_mode` must be selected, with either the `fixed` or `random` option.

Example:
```
python pytorch/train_model.py test_missing mnist test_configs 0 --model vae --cuda --log_var -14 --n_test_epochs 500 --test_batch_size 256 --missing_mode fixed
```

## Test on clean

To test a trained model on clean data, use the `test_clean` option.

Example:
```
python pytorch/train_model.py test_clean mnist test_configs 0 --model vae --log_var -3 --cuda --n_test_epochs 5 --test_batch_size 256
```

## Sample commands

```
python configs/make_configs.py test_configs
python pytorch/train_model.py train mnist test_configs 0 --model vae --cuda --log_var -14 --n_train_epochs 500 --n_test_epochs 500 --test_batch_size 256 --batch_size 32
python pytorch/train_model.py test_missing mnist test_configs 0 --model vae --cuda --log_var -14 --n_test_epochs 500 --test_batch_size 256 --missing_mode fixed
```
The above models are applied to data imputation. Since imputation is related to recreating the exact missing value, very small variances work better than larger ones. 

For the synthetic datasets, you will need to provide the file prefix for the data files:
```
python pytorch/train_model.py train artificial art_vae 800 --cuda --model vae --log_var -14 --dataset_path data/artificial/1/activated_masked_50k
```

## Helper scripts to gather results

```
# gather all results into a single csv file
python scripts/pull_pytorch_errors.py <config name>
python scripts/pull_pytorch_errors.py test_config

# plot predictions and input for mnist-type data
python scripts/plot_mnist.py <mnist / fashion_mnist> <config name> <config number>
python scripts/plot_mnist.py fashion_mnist test_config 0
```
