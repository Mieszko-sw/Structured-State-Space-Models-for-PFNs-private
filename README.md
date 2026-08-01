This repository provides training and evaluation pipelines for Tydra, the first hybrid model for tabular data.

## Training

To train Tydra {4HT}, run:

​```
python3 train_{4HT}.py
​```. 

Other Tydra variants can be trained analogously.

## Evaluation

To evaluate AUROC and inference speed across all datasets, run:

​```
python3 evaluation_per_dataset_speed_synchronized.py
​```

To evaluate performance on longer context lengths, run:

​```
python3 evaluation_script_speed.py
​```
