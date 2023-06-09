# Learning To Invert: Simple Adaptive Attacks for Gradient Inversion in Federated Learning

To reproduce the CIFAR10 experiment, run the following script
```
python main_learn_dlg.py --lr 1e-4 --epochs 200 --leak_mode $leak_mode --model MLP-3000 --dataset CIFAR10 --batch_size 256 --shared_model LeNet
```
by setting `leak_mode` as `None`, `sign`, `prune-0.99`, `gauss-0.1`
