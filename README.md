# Application of Attention Mechanism to Improve YOLOv3 Model for Hand Detection - SEYOLOv3-model

This project adds an attention mechanism and pruning to the YOLOv3 model, using an open-source hand detection dataset [oxford hand](http://www.robots.ox.ac.uk/~vgg/data/hands/) for hand detection,
and model pruning based on that. For this dataset, after channel pruning for YOLOv3, the model's parameter count and model size are reduced by 80%, FLOPs are reduced by 70%, 
and the forward inference speed can reach 200% of the original, while maintaining a roughly unchanged mAP.

## Environment
Python 3.6, Pytorch 1.0 and above

The implementation of SEYOLOv3 refers to eriklindernoren's [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3), so the code dependencies can also refer to its repo.

## Installation
##### Clone and install requirements
    $ git clone （https://github.com/Lidongfeng/SEYOLOv3-model）
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

## Dataset Preparation
1. Download the [dataset](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz) to get the compressed file.
2. Unzip the compressed file into the data directory, obtaining the hand_dataset folder.
3. Run converter.py in the data directory to generate images, labels folders and train.txt, valid.txt files. The training set contains a total of 4,807 images,
 and the test set contains a total of 821 images.
## Normal Training (Baseline)

```bash
python train.py --model_def config/yolov3-hand.cfg
```

## Attention Mechanism Introduction
In this project, we add an attention mechanism, which means that during image processing, more focus will be placed on important features while ignoring unimportant information. 
This can help YOLOv3 more accurately detect objects in images and perform detection faster.

The attention modules mainly include: SE, CBAM modules, information reconstruction and SEYoLo model construction for the YOLOv3 model.
SEnet (Squeeze-and-Excitation Network) takes into account the relationship between feature channels and adds an attention mechanism to the feature channels.
SEnet automatically learns the importance of each feature channel and uses the obtained importance to enhance the features and suppress unimportant features for the current task.

CBAM (Convolutional Block Attention Module) combines the attention mechanism of both feature channels and feature space dimensions. It automatically learns the importance of each feature channel, 
similar to SENet. Additionally, it automatically learns the importance of each feature space in a similar way. It then uses the obtained importance to enhance the features and suppress unimportant features for the current task.
CBAM extracts feature space attention in the following way: after ChannelAttention, the feature map with channel importance selected is sent to the feature space attention module. 
Similar to the channel attention module, space attention performs max-pooling and average-pooling for each channel, and then concatenates the results. Afterward, 
a convolution is used to reduce the feature map to a 1∗w∗h1*w*h1∗w∗h spatial weight, which is then point-multiplied with the input feature to implement the spatial attention mechanism.

## Pruning Algorithm Introduction
This code is based on the paper [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) 
to implement an improved channel pruning algorithm. A similar implementation can be found here [yolov3-network-slimming](https://github.com/talebolano/yolov3-network-slimming). 
The original algorithm in the paper is for classification models and is based on pruning using the gamma coefficient of the BN layer.

## General Steps of the Pruning Algorithm
The following are just general steps of the algorithm. During the specific implementation, the s parameter needs to be tried, or iterative pruning may be required.

1. Perform sparse training
```bash
   python train.py --model_def config/yolov3-hand.cfg -sr --s 0.01
   ```
2. Perform pruning based on the test_prune.py file to obtain the pruned model

3. Fine-tune the pruned model
```bash
   python train.py --model_def config/prune_yolov3-hand.cfg -pre checkpoints/prune_yolov3_ckpt.pth
   ```

### Comparison Before and After Pruning
1. The following image shows the change in the number of channels before and after pruning some convolutional layers:

![](https://raw.githubusercontent.com/Lam1360/md-image/master/img/20190628205342.png)
   > The number of channels in some convolutional layers is significantly reduced


2. Comparison of indicators before and after pruning:

 |                | Number of Parameters|Model Size|Flops | Forward Inference Time (2070 TI) |  mAP   |
   | :------------: | :----------------:| :------: | :---:| :------------------------------: | :----: |
   | Baseline (416) |        61.5M      | 246.4MB  |32.8B |              15.0 ms             | 0.6792 |
   |  Prune (416)   |        10.9M      |  43.6MB  | 9.6B |              7.7 ms              | 0.7213 |
   | Finetune (416) |        Same       |   Same   | Same |              Same                | 0.7250 |


 >After adding a sparse regularization term, the mAP is even higher (it was found during the experiment that actually, mAP fluctuation of 0.02 is normal). Therefore, it can be considered that the mAP obtained by sparse training is almost the same as that of normal training. Finetuning the pruned model does not significantly improve the performance, so the three-step pruning can be simplified into two steps. Before and after pruning, the number of model parameters and model size is reduced to 1/6 of the original, FLOPs is reduced to 1/3, and the forward inference speed can reach twice the original, while maintaining a roughly unchanged mAP. It should be noted that the pruning effect shown in the table above is specific to this dataset and may not necessarily guarantee similar results on other datasets.

3. SEYOLO model testing:

The following code can be executed for testing:

```bash
   python detect.py --image_folder data/imgs/ --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-hand.cfg --class_path data/oxfordhand.names --conf_thres 0.01
   ```

The result of detection would be in output/imgs.