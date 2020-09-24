import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1234)

view_label = "airplane"
offset_idx = 1
f = np.load("../output_save/1599983066_pointnet_iter_l2_attack_dropout_none.npz")
shape_names = [line.rstrip() for line in open("../data/shape_names.txt")]
show_grads = False
show_true = False

x = f["x"]
y_pred = f["y_pred"]
y_pred_idx = np.argmax(y_pred, axis = 1)
x_adv = f["x_adv"]
y_adv_pred = f["y_adv_pred"]
y_adv_pred_idx = np.argmax(y_adv_pred, axis = 1)
grad_adv = f["grad_adv"]

print("NAN:", np.any(np.isnan(x)))
print("Shape:", x.shape)
print("Labels:", [shape_names[idx] for idx in np.unique(y_pred_idx)])
print("Successful attacks:", np.count_nonzero(y_pred_idx != y_adv_pred_idx))

avg_zero_grad = np.sum(np.all(np.isclose(grad_adv, 0.0), axis = 2)) / float(len(x))
print("Average number of points with zero gradients:", avg_zero_grad)

print("Selected label:", view_label)

match_idx = np.where(np.logical_and(y_pred_idx != y_adv_pred_idx, y_pred_idx == shape_names.index(view_label)))[0]
x_view = x[match_idx[offset_idx]]
y_pred_idx_view = y_pred_idx[match_idx[offset_idx]]
x_adv_view = x_adv[match_idx[offset_idx]]
y_adv_pred_idx_view = y_adv_pred_idx[match_idx[offset_idx]]
grad_adv_view = grad_adv[match_idx[offset_idx]]

print("Attack result label:", shape_names[y_adv_pred_idx_view])
print("Clean prediction confidence:", y_pred[match_idx[offset_idx]][y_pred_idx_view])
print("Adversarial prediction confidence:", y_adv_pred[match_idx[offset_idx]][y_adv_pred_idx_view])
print("Shape index:", match_idx[offset_idx])

def scale_plot():
    scale = 0.7
    plt.gca().auto_scale_xyz((-scale, scale), (-scale, scale), (-scale, scale))
    plt.gca().view_init(-30, 200)
    plt.axis("off")

if show_true:
    plt.figure(figsize = (30, 15))

    plt.subplot(121, projection = "3d")

    plt.gca().scatter(*x_view.T, zdir = "y", s = 50, c = x_view.T[1], cmap = "winter")

    scale_plot()

    plt.subplot(122, projection = "3d")
else:
    plt.figure(figsize = (15, 15))
    plt.subplot(111, projection = "3d")

if show_grads:
    grad_adv_view = np.linalg.norm(grad_adv_view, axis = 1)
    close_to_zero = np.isclose(grad_adv_view, 0.0)
    point_color = np.logical_not(close_to_zero).astype(float)

    plt.gca().scatter(*x_adv_view.T, zdir = "y", s = 50, c = -point_color, cmap = "winter")
else:
    plt.gca().scatter(*x_adv_view.T, zdir = "y", s = 50, c = x_adv_view.T[1], cmap = "winter")

scale_plot()

plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.tight_layout()
plt.savefig("../figures/adv_sticks_grads.pdf", bbox_inches = "tight")
plt.show()
