This repository provides training and evaluation pipelines for Tydra, the first hybrid model for tabular data.


## Getting started

A Dockerfile is provided, along with a `setup.sh` script in the repository root, to build an Ubuntu-based Docker container with all dependencies installed.

To launch the experiments, we used the following combination of dependencies:

- Python == 3.8.10
- CUDA == 11.8
- PyTorch == 2.4.1


## Training

To train Tydra {4HT}, run:

​```
python3 train_{4HT}.py
​```. 

Other Tydra variants can be trained analogously.

## Evaluation

To evaluate AUROC and inference speed across all OpenML datasets, run:

​```
python3 evaluation_per_dataset_speed_synchronized.py
​```

To evaluate inference performance on longer context lengths, run:

​```
python3 evaluation_script_speed.py
​```
