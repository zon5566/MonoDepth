# MonoDepth

<img src="https://github.com/zon5566/MonoDepth/blob/master/img/input_1.png" height="130"/><img src="https://github.com/zon5566/MonoDepth/blob/master/img/light_1.png" height="130"/>

This is a Pytorch implementation of the work from the paper ***_Unsupervised Monocular Depth Estimation with Left-Right Consistency_*** by Clément Godard, Oisin Mac Aodha, and Gabriel J. Brostow. The original paper and works can be found via the links: <a href='https://arxiv.org/abs/1609.03677'>Paper Link</a>, <a href='https://github.com/mrharicot/monodepth'>GitHub Link</a>.

The code contains the architectures mentioned in the paper (monodepth, resnet), and the training/evaluation. However, I modified some parts of the monodepth architecture to make the training smoother, and also make the visual performance better.

## Requirements
The code was written in Python 3.5, Pytorch 0.4.1, cuda 9.0<br>
The model is trained on KITTI dataset (29000 images, organized by paper authors) with 8 batch size, totally 25 hours with 50 epochs, on a single GTX-1080.

## Files Description
* ```main.py```: the main function where training or testing starts.
* ```depthmodel.py```: contains the training propogation process and the loss calculation.
* ```model_monodepth.py/model_resnet.py```: the model architecture
* ```dataloader.py```: data loader object extended from the built in function in Pytorch.
* ```config.py```: parameters setting
* ```evaluate.py```: this program will generate a csv file of numerical errors between the ground truth images and the predicted ones from the npy file, which is generated by main.py with evaluate mode.

## Dataset -  KITTI
The source images are compressed together to zip files with same dates and same labels, and these zip files are stored under the directorty ```kitti_zip_light```, which is also the parameter ```dataset_root``` in ```config.py```.

```
kitti_zip_light
|
|-2011_09_28_drive_0001_sync.zip
|   |
|   |- image_02/data
|   |    |
|   |    |-0000000105.jpg
|   |    |- ...
|   |
|   |- image_03/data
|        |
|        |-0000000105.jpg
|        |- ...
|     
|-2011_09_26_drive_0057_sync.zip
|   |
|   ...
...

```

## Training
To train the model from scratch, just run the command```python3 main.py``` with parameters ```--model_type```, which represents the name of the model. This name will be used as the subdirectory of the checkpoints, images, the tensorboard, and the evaluation things.

## Evaluation
First, run the command ```python3 main.py --mode=evaluation --model_type=<your model_type>```. The model type name is the one you want to evaluate with.

Numerical errors can be calculated by typing ```python3 evaluate.py <your model type>```. The csv result will be saved in the directory ```evaluation/csv/<your model type>/```, and the predicted images will be in ```evaluation/img/<your model type>/```

## Loss Visualization
For each training iteration, the loss is stored in the directory ```runs/<your model type>```. To see the loss trend chart, run the command ```tensorboard --logdir=runs/<your modl type>``` in the root directory. Before running the command, make sure the tensorboardX package is installed.

## More Examples
The below figures from top to bottom are input images, depth images produced from the model based on the paper, and the ones produced from the model with some modifications.

<img src="https://github.com/zon5566/MonoDepth/blob/master/img/input_trafficlight.png" height="130"/><img src="https://github.com/zon5566/MonoDepth/blob/master/img/input_guide.png" height="130"/>
<img src="https://github.com/zon5566/MonoDepth/blob/master/img/depth_trafficlight.png" height="130"/><img src="https://github.com/zon5566/MonoDepth/blob/master/img/depth_guide.png" height="130"/>
<img src="https://github.com/zon5566/MonoDepth/blob/master/img/dp_trafficlight.png" height="130"/><img src="https://github.com/zon5566/MonoDepth/blob/master/img/dp_guide.png" height="130"/>
