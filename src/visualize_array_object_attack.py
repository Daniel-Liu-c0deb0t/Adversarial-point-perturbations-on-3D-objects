from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.unicode"] = True
plt.rcParams["font.family"] = "serif"

# attacks
paths = [
    "../output_save/final/1564113021_pointnet_none_none.npz",
    "../output_save/final/1564130586_pointnet_iter_l2_attack_none.npz",
    "../output_save/final/1564164544_pointnet_iter_l2_attack_n_proj_none.npz",
    "../output_save/final/1564190398_pointnet_iter_l2_attack_n_sampling_none.npz",
    "../output_save/final/1564208936_pointnet_iter_l2_adversarial_sticks_none.npz",
    "../output_save/final/1564384193_pointnet_iter_l2_attack_sinks_none.npz"
]

xlabels = ["None", "Iter. gradient $L_2$", "Distribution", "Perturb. resample", "Adv. sticks", "Adv. sinks"]

models = ["car", "person", "lamp", "chair", "vase"]
offset_idx = [0, 0, 0, 0, 0]
shape_names = [line.rstrip() for line in open("/media/sf_Xubuntu_shared/modelnet40_pc/shape_names.txt")]

files = []

for path in paths:
    files.append(np.load(path))

match_idx = []

for i, model in enumerate(models):
    model_idx = shape_names.index(model)
    idx = np.where(np.argmax(files[0]["y_pred"], axis = 1) == model_idx)[0]
    idx = idx[offset_idx[i]]
    match_idx.append(idx)

plt.figure(figsize = (24, 20))

def scale_plot():
    scale = 0.7
    plt.gca().auto_scale_xyz((-scale, scale), (-scale, scale), (-scale, scale))
    plt.gca().view_init(30, 120)
    plt.axis("off")

for i, idx in enumerate(match_idx):
    for j, f in enumerate(files):
        plt.subplot(len(match_idx), len(files), i * len(files) + j + 1, projection = "3d")
        plt.gca().scatter(*f["x_adv"][idx].T, zdir = "y", s = 5, c = f["x_adv"][idx].T[2], cmap = "winter")
        scale_plot()

for i in range(len(xlabels)):
    plt.gcf().text(i / (float(len(xlabels)) + 0.3) + 0.5 / len(xlabels) + 0.05, 0.93, xlabels[i], fontsize = 30, horizontalalignment = "center")

for i in range(len(models)):
    plt.gcf().text(0.05, i / (float(len(models)) + 0.1) + 0.5 / len(models), models[-i - 1].capitalize(), fontsize = 30, rotation = "vertical", verticalalignment = "center")

plt.gcf().text(0.5, 0.96, "Attacks", fontsize = 40, horizontalalignment = "center")
plt.gcf().text(0.01, 0.5, "Objects", fontsize = 40, rotation = "vertical", verticalalignment = "center")

plt.subplots_adjust(left = 0.05, bottom = 0, right = 1, top = 0.95, wspace = 0, hspace = 0)
plt.savefig("../figures/object_attack.pdf", bbox_inches = "tight")
plt.show()
