# hallucination_by_analogy
Tensorflow implementation of the hallucination part of the paper [1]. The few-shot learning experiments are conducted using the CIFAR-100 dataset downloaded from the official website [2].

## Usage
### (1) Create sets of base classes (of size 80) and novel classes (of size 20, one for each superclass)
```
python3 preprocessing.py \
  --raw_path [Path to the CIFAR-100 raw data (e.g., the path containing train and test)] \
  --data_path [Path to save the split train and test data (train_novel, train_base, test_novel, and test_base)]
```
You can also use `--n_novel_per_coarse` to specify other numbers of novel classes per superclass (default 1).


### (2) Train the CNN-based feature extractor and extract features
```
python3 train_extractor.py \
  --data_path [Path of the split train and test data (train_novel, train_base, test_novel, and test_base)] \
  --result_path [Path to save all the results] \
  --extractor_name [Folder name (under result_path) to save the checkpoints of the extractor] \
  --vgg16_npy_path [Path of the imagenet pre-trained weights of VGG16 (e.g., the path to vgg16.npy)]
```
A model consisted of a VGG-based CNN and a 3-layer MLP will be trained using `train_base`. After training, the performance will be evaluated on `test_base`, and then all feature vectors will be extracted from `train_novel`, `train_base`, `test_novel`, and `test_base` by the CNN part of the model.

### (3) Make the quadruplet dataset for training the analogy-based hallucinator
```
python3 make_quadruplet.py \
  --result_path [Path to save all the results] \
  --extractor_name [Folder name (under result_path) of the saved checkpoints of the extractor] \
  --quadruplet_name [File name (under result_path/extractor_name) to save the quadruplet dataset] \
  --n_clusters [Number of clusters used in the KMeans clustering algorithm, default: 10]
```
Multi-processing version. You can also use `--n_cores` to specify the number of CPU cores you want to use (default 8).
You can also use `make_quadruplet_coarse.py` to make quadruplets, each coming from the same superclass.

### (4) Train the analogy-based hallucinator
```
python3 train_hallucinator.py \
  --result_path [Path to save all the results] \
  --extractor_name [Folder name (under result_path) of the saved checkpoints of the extractor] \
  --quadruplet_name [File name (under result_path/extractor_name) of the saved quadruplet dataset] \
  --hallucinator_name [Folder name (under result_path) to save the checkpoints of the hallucinator] \
  --loss_lambda [Scale to control the weighting of the MSE loss and the classification loss, default: 10.0]
```

### (5) Train the few-shot learning classifier
```
python3 train_fsl.py \
  --result_path [Path to save all the results] \
  --model_name [Folder name (under result_path) to save the checkpoints of the few-shot learning classifier] \
  --extractor_name [Folder name (under result_path) of the saved checkpoints of the extractor] \
  --hallucinator_name [Folder name (under result_path) of the saved checkpoints of the hallucinator] \
  --data_path [Path of the saved class_mapping dictionary] \
  --n_shot [Number of shot] \
  --n_min [Minimum number of samples per training class you want to have]
```
If `n_shot == n_min`, it means that there is no hallucination.
If `--data_path` is `None` (the default value), it means that the hallucination is done in an usual manner proposed in [1]; otherwise, base sample pairs will be selected from the same superclass as the seed novel sample for hallucination before few-shot learning.

### (6) Infer the few-shot learning classifier
```
python3 infer_fsl.py \
  --result_path [Path to save all the results] \
  --model_name [Folder name (under result_path) of the saved checkpoints of the few-shot learning classifier] \
  --extractor_name [Folder name (under result_path) of the saved checkpoints of the extractor] \
  --hallucinator_name [Folder name (under result_path) of the saved checkpoints of the hallucinator]
```

## References
[1] Bharath Hariharan and Ross Girshick, "Low-shot Visual Recognition by Shrinking and Hallucinating Features," *ICCV*, 2017  
[2] https://www.cs.toronto.edu/~kriz/cifar.html  