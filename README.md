# This is a pytorch implementation of MobileNetV2, ShuffleNetV2 and GhostNet on CIFAR-10 datasets.


Models are modified as the inputs are all 32*32 RGB images. We only conduct three downsample operations and you can find the modified parts in .py files.

# Reference
[MobileNetV2](https://arxiv.org/abs/1801.04381)
	[ShuffleNetV2](https://arxiv.org/abs/1807.11164)
	[ghostnet](https://arxiv.org/abs/1911.11907)

# Network
	-MobileNetV2
	-ShuffleNetV2
	-GhostNet

# Accuracy After Training:

	Model				Accuracy
	MobileNetV2			93.92%
	ShuffleNetV2			92.60%
	GhostNet			92.57%

![](https://github.com/MonkeyKing-KK/Huaguoshan/blob/master/compare.jpg) 

# Requirements:
    -requirements.txt

# Quick Start:
default settings: batch_size = 128, max_epoch = 190, lr = 0.1. 
    
    python train.py

# More Training Options
    -bs			batch_size, type=int, default=128
    -max_epoch		num of training epochs, type=int, default=190
    -lr			learning_rate, default=0.1
    -arc		model to use['MobileNetV2 shufflenetv2 ghost_net'] [default=MobileNetV2]

# To Do
	Add more lightweight models
	Clean up the code


