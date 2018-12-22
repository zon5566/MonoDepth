# MonoDepth

This is a Pytorch implementation of the work from the paper ***_Unsupervised Monocular Depth Estimation with Left-Right Consistency_*** by Clément Godard, Oisin Mac Aodha, and Gabriel J. Brostow. The original paper and works can be found via the links: <a href='https://arxiv.org/abs/1609.03677'>Paper Link</a>, <a href='https://github.com/mrharicot/monodepth'>GitHub Link</a>.

The code contains the architectures mentioned in the paper (monodepth, resnet), and the training/evaluation. However, I modified some parts of the monodepth architecture to make the training smoother and make the visual performance better.

## Requirements
The code was written in Python 3.5, Pytorch 0.4.1, cuda 9.0<br>
The model is trained on KITTI dataset (29000 images, organized by paper authors) with 8 batch size, totally 25 hours with 50 epochs, on a single GTX-1080.
