First, generate the prototypes by launching "prototype_learning" by specifying the dimension and the number of classes as:

```shell
python prototype_learning.py -d 3 -c 100
```

Then, launch the training by specifying the loss and the dataset:
```shell
python train_busemann.py --dataset "cifar10" --loss "hsw_mixt" --dims 3 --lambd 1 --scale_var 0.1 --prop 0.75 --mult 0.1 --batch_size 128
```

To get the datasets, you can look at the notebooks from [Hyperbolic Representation Learning for Computer Vision](https://sites.google.com/view/hyperbolic-tutorial-eccv22/).
