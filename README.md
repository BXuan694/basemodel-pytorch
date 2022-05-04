# basemodel-pytorch
Some classification mathods. 

# Request

python3.8.13, PyTorch1.7.1

# How to use

1. prepare data in which a folder with images acts as a class.

2. set configs (dataset path, classifier, etc.) in data/config.py.

3. set input size and modify the corresponding input shape of fc layer.

4. start training
```bash
python train.py
```

# TODO:

move step 3 above into config.py

move preprocess part(data augmentation) from train.py into config.py

add valid.py

finish README.md

# ref

https://github.com/kuangliu/pytorch-cifar


