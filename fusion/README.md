# End-to-End multi-modal outcomes prediction using image and tabular data

This project use tabular deep learning techniques (e.g. TabNet, or similar) to featurize the clinical variables and therapies, unite the two tower stacks (image and clinical model) using trainable layers, and train the entire network end-to-end. 

## Preprocess EMR data
Run the following to preprocess and generate cross validation split for the EMR data: 
```python
python ./twotowers/datasets/preprocess.py
```

## Training the model
To train a model, run the following:

```python
python run.py <path_to_config_file> --train
```

You can also train the model on a particular split of the data: 

```python
python run.py <path_to_config_file> --train --cv_split 2
```

To pretrain TabNet, run it with the **--pretrain** flag:
```python
python run.py <path_to_config_file> --train --cv_split 2 --pretrain
```

### Run hyperparameter sweep with wandb
Example sweep configs can be found in *./configs/emr/supervised/sweep00\*.yaml

```
wandb sweep <path_to_sweep_config>
wandb agent <sweep-id>
```

Currently, the sweeps 1-5 represents sweeps for each of the 5 data splits. Each, of the sweeps read the same config file. 


### Test and get prediction
To test the model from a sweep for a particular split (i.e. 2), run the following with the same config file used for training (should be the same config file in sweep00\*.yaml): 

```python
python run.py <path_to_config_file> --test --cv_split 2
```

The configuration and ckpt with the best validation AUROC will be loaded for the test set. The path to the predictions will be printed. 

