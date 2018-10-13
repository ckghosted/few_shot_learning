# hallucination_by_analogy
Tensorflow implementation of the hallucination part of the paper [1]. The few-shot learning experiments are conducted using the CIFAR-100 dataset downloaded from the official website [2].

## Usage
### (1) Create sets of base classes (of size 80) and novel classes (of size 20, one for each superclass)
```
python3 preprocessing.py \
  --raw_path [CIFAR-100 raw data (e.g., train and test)] \
  --data_path [train and test split into novel/base classes by 20/80 (e.g., train_novel, train_base, test_novel, and test_base)]
```

### (2) Train the CNN-based feature extractor and extract features
```
python3 train_extractor.py \
  --data_path [path of the split datasets] \
  --result_path [where you want to put the results] \
  --extractor_name [the folder name of the saved checkpoints of the extractor] \
  --vgg16_npy_path [path of the imagenet pre-trained weights of VGG16: vgg16.npy]
```

### (3) Train the hallucinator
```
python3 train_hallucinator.py \
  --result_path [where you want to put the results] \
  --extractor_name [the folder name of the saved checkpoints of the extractor] \
  --hallucinator_name [the folder name of the saved checkpoints of the hallucinator]
```

### (4) Train the few-shot learning classifier
```
python3 train_fsl.py \
  --data_path [path of the split datasets] \
  --result_path [where you want to put the results] \
  --extractor_name [the folder name of the saved checkpoints of the extractor] \
  --hallucinator_name [the folder name of the saved checkpoints of the hallucinator] \
  --model_name [the folder name of the saved checkpoints of the few-shot learning classifier]
```

## References
[1] Bharath Hariharan and Ross Girshick, "Low-shot Visual Recognition by Shrinking and Hallucinating Features," *ICCV*, 2017  
[2] https://www.cs.toronto.edu/~kriz/cifar.html  