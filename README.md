# Adversarial point perturbations on 3D objects
New distributional and shape attacks on neural networks that process 3D point cloud data.

# Setup
Initially, part of the code for training/testing neural networks was designed to be ran on a server with GPUs, and the other part for visualizing results was supposed to be ran on a personal computer. The setup instructions below are for running both parts on the same computer. This means that some installation decisions may be kind of weird.

This is the directory structure of the project:
```
- /
  - pointnet/
  - pointnet2/
  - Adversarial-point-perturbations-on-3D-objects/
    - pointnet/
    - pointnet2/
    - src/
    - data/
    - output_save/
    - figures/
```

We will assume that an empty folder is created somewhere, and we will refer to it as the root folder, or `/`. The first step is to clone three different repositories into the root folder: [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/charlesq34/pointnet2), and this project.

Then, copy the modified PointNet files from the `pointnet` directory to the cloned PointNet repo, and do the same for PointNet++. The modified files contain some bug fixes and other nonessential modifications that carried over from my previous [work](https://github.com/Daniel-Liu-c0deb0t/3D-Neural-Network-Adversarial-Attacks). Follow the instructions for training PointNet and PointNet++ from their respective repos. A lowered batch size of 10 was used in single-GPU training for PointNet++, for lower memory requirements. The checkpoint files should be saved in `/pointnet/log/` and `/pointnet2/log/` for PointNet and PointNet++.

Next, assuming that the current working directory is `/`, do the following:
```
cd Adversarial-point-perturbations-on-3D-objects/
mkdir output_save/
mkdir figures/
```
For each experiment, a file with the adversarial examples, clean examples, and other stuff will be saved in the `/output_save/` folder. The `figures` folder is for saving generated visualizations, if necessary.

To download the necessary 3D point clouds sampled from the [ModelNet40](https://modelnet.cs.princeton.edu/) dataset, do
```
cd data/
curl https://github.com/Daniel-Liu-c0deb0t/Adversarial-point-perturbations-on-3D-objects/releases/download/Data/point_clouds.npz
```
The triangle each point was sampled from are included.

In this project, there are two types of runnable scripts: visualization scripts and experiment scripts.

Experiment scripts:
- 

Visualization scripts:
- 
