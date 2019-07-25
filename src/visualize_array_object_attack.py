from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# attacks
paths = [
    "",
    "",
    "",
    "",
    ""
]

xlabels = ["iter. gradient $L_2$", "distribution", "perturb. resample", "adv. sticks", "adv. sinks"]

models = ["car", "person", "lamp", "chair", "vase"]
offset_idx = [0, 0, 0, 0, 0]
shape_names = [line.rstrip() for line in open("/media/sf_Xubuntu_shared/modelnet40_pc/shape_names.txt")]

files = []

for path in paths:
    files.append(np.load(path))

match_idx = []

for i, model in enumerate(models):
    model_idx = shape_names.index(model)
    idx = np.where(np.argmax(files[0][0]["y_pred"], axis = 1) == model_idx)[0]
    idx = idx[offset_idx[i]]
    match_idx.append(idx)

plt.figure(figsize = (30, 20))

def scale_plot():
    plt.gca().auto_scale_xyz((-1, 1), (-1, 1), (-1, 1))
    plt.gca().view_init(0, 0)
    plt.axis("off")

for i, idx in enumerate(match_idx):
    for j, f in enumerate(files):
        plt.subplot(len(match_idx), len(files), i * len(files) + j, projection = "3d")
        plt.gca().scatter(*f["x_adv"][idx].T, zdir = "y", s = 20, c = f["x_adv"][idx].T[1], cmap = "winter")

        if i == len(match_idx) - 1:
            plt.xlabel(xlabels[j])

        if j == 0:
            plt.ylabel(models[i])

        scale_plot()

plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.tight_layout()
plt.savefig("../figures/object_attack.pdf", bbox_inches = "tight")
plt.show()
