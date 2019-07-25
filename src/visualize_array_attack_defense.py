from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# defenses x attacks
paths = [
    [
        "",
        "",
        "",
        "",
        ""
    ],
    [
        "",
        "",
        "",
        "",
        ""
    ],
    [
        "",
        "",
        "",
        "",
        ""
    ]
]

xlabels = ["iter. gradient $L_2$", "distribution", "perturb. resample", "adv. sticks", "adv. sinks"]
ylabels = ["no defense", "remove outliers", "remove salient"]

model = "airplane"
offset_idx = 0
shape_names = [line.rstrip() for line in open("/media/sf_Xubuntu_shared/modelnet40_pc/shape_names.txt")]

files = []

for attack_paths in paths:
    f = []

    for path in attack_paths:
        f.append(np.load(path))

    files.append(f)

model_idx = shape_names.index(model)
match_idx = np.where(np.argmax(files[0][0]["y_pred"], axis = 1) == model_idx)[0]
match_idx = match_idx[offset_idx]

plt.figure(figsize = (30, 15))

def scale_plot():
    plt.gca().auto_scale_xyz((-1, 1), (-1, 1), (-1, 1))
    plt.gca().view_init(0, 0)
    plt.axis("off")

for i, attack_files in enumerate(files):
    for j, f in enumerate(attack_files):
        plt.subplot(len(files), len(files[0]), i * len(files[0]) + j, projection = "3d")
        plt.gca().scatter(*f["x_adv"][match_idx].T, zdir = "y", s = 20, c = f["x_adv"][match_idx].T[1], cmap = "winter")

        if i == len(files) - 1:
            plt.xlabel(xlabels[j])

        if j == 0:
            plt.ylabel(ylabels[i])

        scale_plot()

plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.tight_layout()
plt.savefig("../figures/attack_defense.pdf", bbox_inches = "tight")
plt.show()
