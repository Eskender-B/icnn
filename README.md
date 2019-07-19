# Interlinked Convolutional Neural Network for Face Parsing
This is a pytorch implementation of the [ICNN](https://arxiv.org/abs/1806.02479) paper.

## How to Run
Copy and unzip [Helen](http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/SmithCVPR2013_dataset_resized.zip) dataset in "../data" (one level up from project directory)

Prepare facial parts (Only once)
```python3 extract_parts.py```

Create result folder
```mkdir res/```

Train Stage1
```python3 train_stage1.py```

Train Stage2
```python3 train_stage2.py```

Run end2end
```python3 end2end.py```
