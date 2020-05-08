# Shape correspondence using anisotropic Chebyshev spectral CNNs

This is the pytorch implementation for the paper 'Shape correspondence using anisotropic Chebyshev spectral CNNs' by Qinsong Li, Shengjun Liu, Ling Hu and Xinru Liu. accepted by CVPR 2020.

In this paper, we extend the spectral CNN to an anisotropic case based on the anisotropic Laplace-Beltrami Operator (ALBO) which allows to aggregate local features from multiply diffusion directions and achieve state-of-art results on shape correspondence task.

![FAUST_cqc](https://github.com/GCVGroup/ACSCNN/blob/master/figs/FAUST_cqc.png)

![FAUST_cqc](https://github.com/GCVGroup/ACSCNN/blob/master/figs/faust_transfer_errs_low_res.jpg)

# How to use this code

1. run matlab code 'run_data_prec_compute_anisotropic.m' to prepare ALBO matrix.
2. start training 'python ACSCNN_faust.py'.

If you have any questions, please contack me. qinsli.cg@foxmail.com (Qinsong Li)

# License 

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us (shjliu.cg@csu.edu.cn, qinsli.cg@foxmail.com).

# Acknowledgments

FAUST dataset: http://faust.is.tue.mpg.de/

F. Bogo, J. Romero, M. Loper and M. J. Black, "FAUST: Dataset and Evaluation for 3D Mesh Registration," *2014 IEEE Conference on Computer Vision and Pattern Recognition*, Columbus, OH, 2014, pp. 3794-3801, doi: 10.1109/CVPR.2014.491.

Discretization of ALBO matrix:  

S. Melzi, E. Rodolà, U. Castellani and M. M. Bronstein, "Shape Analysis with Anisotropic Windowed Fourier Transform," *2016 Fourth International Conference on 3D Vision (3DV)*, Stanford, CA, 2016, pp. 470-478, doi: 10.1109/3DV.2016.57.
