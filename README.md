# Generative Shape Synthesis with Variational Auto-Decoder
This project is part of the 'Advanced Deep Learning for Computer Vision' course at the chair of Prof. Dr. Niessner at Technical University of Munich. You can view our final report [**here**](./docs/Generative_Shape_Synthesis_VAD.pdf).

## Requirements
- Python 3
- Install dependencies from ```requirements.txt```  

```shell 
pip install -r requirements.txt
```
- Install [chamferdist package](https://github.com/krrish94/chamferdist) by [krrish94](https://github.com/krrish94)  

*pip*: 
```shell 
pip install chamferdist
``` 
*conda*: download the repository to the root of this project, then run ```python setup.py install``` that is inside the folder ```chamferdist/```  
Note: we only tested it with conda

## Data
- Download [shapenet_dim35_df.zip](https://drive.google.com/file/d/1-0WDifB7km53JgfsTSEjxIXTakFilEEE/view?usp=sharing) and unzip the file under ```data/```
Your directory should then look like this:
```
data/
    shapenet_dim32_df/
        02691156/...
        02747177/...
        ...
    splits/
        shapenet/
            airplane_test.txt
            airplane_train.txt
            ...
```

## Training a model
### Variational Auto Decoder
To train a variational auto-decoder use the following command  
```shell
python scripts/train.py --var <experiment_name> <class>
```  
*Available classes*: car, airplane, chair, sofa, lamp, cabinet, watercraft, table
### Non-Variational Auto Decoder
To train a non-variational auto-decoder, use the following command  
```shell 
python scripts/train.py <experiment_name> <class>
```  
*Available classes*: same as above

## Testing a model
### Variational Auto Decoder
To test a trained variational auto-decoder on the validation data, use the following command  
```shell
python scripts/train.py --var --test <experiment_name> <class> 
```  
### Non-Variational Auto Decoder
To test a trained non-variational auto-decoder on the validation data use the following command  
```shell
python scripts/train.py --test <experiment_name> <class> 
```  

## Evaluating a model
### Variational Auto Decoder
To evaluate a variational auto decoder on the 1-NN score, use the following command  
```shell
python scripts/evaluate.py --split test <experiment_name> <class> 1NN 
```  
If you wish to test with a fewer number of samples from the reference set, use the ```--n 200``` flag  
### Non-Variational Auto Decoder
To evaluate a non-variational auto decoder on the IOU score, use the following command  
```shell
python scripts/evaluate.py --split test <experiment_name> <class> IOU 
```  
*split*: train, val

## Visualizations
For visulizing samples from shape synthesis, inter-class and intra-class interpolation we prepared a jupyter notebook ```visualize.ipynb```.

## Logging
You can use tensorboard to see the losses (under ```logs/```) during training and testing.    
```shel
tensorboard --logdir logs
```
