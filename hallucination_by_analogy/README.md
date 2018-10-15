# hallucination_by_analogy
Tensorflow implementation of the hallucination part of the paper [1]. The few-shot learning experiments are conducted using the CIFAR-100 dataset downloaded from the official website [2].

## Usage
### (1) Create sets of base classes (of size 80) and novel classes (of size 20, one for each superclass)
```
python3 preprocessing.py \
  --raw_path [Path to the CIFAR-100 raw data (e.g., the path containing train and test)] \
  --data_path [Path to save the split train and test data (train_novel, train_base, test_novel, and test_base)]
```

### (2) Train the CNN-based feature extractor and extract features
```
python3 train_extractor.py \
  --data_path [Path of the split train and test data (e.g., the path to save train_novel, train_base, test_novel, and test_base)] \
  --result_path [Path to save all the results] \
  --extractor_name [Folder name (under result_path) to save the checkpoints of the extractor] \
  --vgg16_npy_path [Path of the imagenet pre-trained weights of VGG16 (e.g., the path to vgg16.npy)]
```

### (3) Make the quadruplet dataset for training the analogy-based hallucinator
```
python3 make_quadruplet.py \
  --result_path [Path to save all the results] \
  --extractor_name [Folder name (under result_path) of the saved checkpoints of the extractor] \
  --quadruplet_name [File name (under result_path/extractor_name) to save the quadruplet dataset] \
  --n_clusters [Number of clusters used in the KMeans clustering algorithm, default: 10]
```

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
  --hallucinator_name [Folder name (under result_path) of the saved checkpoints of the hallucinator]
```

### (6) Infer the few-shot learning classifier
```
python3 infer_fsl.py \
  --result_path [Path to save all the results] \
  --model_name [Folder name (under result_path) of the saved checkpoints of the few-shot learning classifier] \
  --extractor_name [Folder name (under result_path) of the saved checkpoints of the extractor]
```

## References
[1] Bharath Hariharan and Ross Girshick, "Low-shot Visual Recognition by Shrinking and Hallucinating Features," *ICCV*, 2017  
[2] https://www.cs.toronto.edu/~kriz/cifar.html  