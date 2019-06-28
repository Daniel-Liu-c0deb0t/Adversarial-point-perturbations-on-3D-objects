import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from alpha_shape import alpha_shape_border
from perturb_proj_tree import PerturbProjTree

np.random.seed(1234)

view_label = "chair"
offset_idx = 0
f = np.load("../output_save/1561684535_pointnet_iter_l2_attack_n_sampling_remove_outliers_defense.npz")
shape_names = [line.rstrip() for line in open("/media/sf_Xubuntu_shared/modelnet40_pc/shape_names.txt")]

x = f["x"]
y_pred = f["y_pred"]
y_pred_idx = np.argmax(y_pred, axis = 1)
x_adv = f["x_adv"]
y_adv_pred = f["y_adv_pred"]
y_adv_pred_idx = np.argmax(y_adv_pred, axis = 1)

print("Shape:", x.shape)
print("Labels:", [shape_names[idx] for idx in np.unique(y_pred_idx)])
print("Successful attacks:", np.count_nonzero(y_pred_idx != y_adv_pred_idx))

avg_zero_grad = np.sum(np.all(np.isclose(x_adv - x, 0.0), axis = 2)) / float(len(x))
print("Average number of points with zero gradients:", avg_zero_grad)

print("Selected label:", view_label)

match_idx = np.where(np.logical_and(y_pred_idx != y_adv_pred_idx, y_pred_idx == shape_names.index(view_label)))[0]
x_view = x[match_idx[offset_idx]]
y_pred_idx_view = y_pred_idx[match_idx[offset_idx]]
x_adv_view = x_adv[match_idx[offset_idx]]
y_adv_pred_idx_view = y_adv_pred_idx[match_idx[offset_idx]]

print("Attack result label:", shape_names[y_adv_pred_idx_view])
print("Clean prediction confidence:", y_pred[match_idx[offset_idx]][y_pred_idx_view])
print("Adversarial prediction confidence:", y_adv_pred[match_idx[offset_idx]][y_adv_pred_idx_view])
print("Shape index:", match_idx[offset_idx])

plt.figure(figsize = (30, 15))

def scale_plot():
    plt.gca().auto_scale_xyz((-1, 1), (-1, 1), (-1, 1))
    plt.gca().view_init(0, 0)
    plt.axis("off")

plt.subplot(121, projection = "3d")

plt.gca().scatter(*x_view.T, zdir = "y", s = 20, c = x_view.T[1], cmap = "winter")

scale_plot()

plt.subplot(122, projection = "3d")

plt.gca().scatter(*x_adv_view.T, zdir = "y", s = 20, c = x_adv_view.T[1], cmap = "winter")

scale_plot()

plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.show()
