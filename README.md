# This is a pytorch implementation of MobileNetV2, ShuffleNetV2 and GhostNet on CIFAR-10 datasets.


Models are modified as the inputs are all 32*32 RGB images. We only conduct three downsample operations and you can find the modified parts in .py files.

# Accuracy after training:

	Model				Accuracy
	MobileNetV2			93.92%
	ShuffleNetV2			92.60%
	GhostNet			92.57%

1. ![](https://github.com/MonkeyKing-KK/Huaguoshan/blob/master/compare.jpg) 

# requirements:
    -requirements.txt

# quick start:
    python train.py
    
	default settings: batch_size = 128, max_epoch = 190, lr = 0.1. 
	Three arc are available just for now: MobileNetV2, shufflenetv2, ghost_net




