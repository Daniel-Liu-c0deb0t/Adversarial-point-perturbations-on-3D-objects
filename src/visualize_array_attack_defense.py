from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.unicode"] = True

# defenses x attacks
paths = [
    [
        "../output_save/1582094622_pointnet_none_none.npz",
        "../output_save/1582096639_pointnet_iter_l2_attack_none.npz",
        "../output_save/1582508068_pointnet_chamfer_attack_none.npz",
        "../output_save/1582152554_pointnet_iter_l2_attack_n_proj_none.npz",
        "../output_save/1582182744_pointnet_iter_l2_attack_n_sampling_none.npz",
        "../output_save/1582186116_pointnet_iter_l2_adversarial_sticks_none.npz",
        "../output_save/1582453960_pointnet_iter_l2_attack_sinks_none.npz"
    ],
    [
        "../output_save/1582094981_pointnet_none_remove_outliers_defense.npz",
        "../output_save/1582097003_pointnet_iter_l2_attack_remove_outliers_defense.npz",
        "../output_save/1582508626_pointnet_chamfer_attack_remove_outliers_defense.npz",
        "../output_save/1582152895_pointnet_iter_l2_attack_n_proj_remove_outliers_defense.npz",
        "../output_save/1582183086_pointnet_iter_l2_attack_n_sampling_remove_outliers_defense.npz",
        "../output_save/1582186458_pointnet_iter_l2_adversarial_sticks_remove_outliers_defense.npz",
        "../output_save/1582454309_pointnet_iter_l2_attack_sinks_remove_outliers_defense.npz"
    ],
    [
        "../output_save/1582095976_pointnet_none_remove_salient_defense.npz",
        "../output_save/1582097991_pointnet_iter_l2_attack_remove_salient_defense.npz",
        "../output_save/1582510577_pointnet_chamfer_attack_remove_salient_defense.npz",
        "../output_save/1582153775_pointnet_iter_l2_attack_n_proj_remove_salient_defense.npz",
        "../output_save/1582183969_pointnet_iter_l2_attack_n_sampling_remove_salient_defense.npz",
        "../output_save/1582187336_pointnet_iter_l2_adversarial_sticks_remove_salient_defense.npz",
        "../output_save/1582455365_pointnet_iter_l2_attack_sinks_remove_salient_defense.npz"
    ]
]

xlabels = ["None", "Iter. gradient $L_2$", "Chamfer", "Gradient proj.", "Perturb. resample", "Adv. sticks", "Adv. sinks"]
ylabels = ["None", "Remove outliers", "Remove salient"]

model = "stool"
offset_idx = 0
shape_names = [line.rstrip() for line in open("../data/shape_names.txt")]

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
    scale = 0.85
    plt.gca().auto_scale_xyz((-scale, scale), (-scale, scale), (-scale, scale))
    plt.gca().view_init(30, 60)
    plt.axis("off")

for i, attack_files in enumerate(files):
    for j, f in enumerate(attack_files):
        plt.subplot(len(files), len(files[0]), i * len(files[0]) + j + 1, projection = "3d")
        plt.gca().scatter(*f["x_adv"][match_idx].T, zdir = "y", s = 5, c = f["x_adv"][match_idx].T[2], cmap = "winter")
        scale_plot()

for i in range(len(xlabels)):
    plt.gcf().text(i / (float(len(xlabels)) + 0.35) + 0.5 / len(xlabels) + 0.05, 0.9, xlabels[i], fontsize = 30, horizontalalignment = "center")

for i in range(len(ylabels)):
    plt.gcf().text(0.05, i / (float(len(ylabels)) + 0.1) + 0.5 / len(ylabels), ylabels[-i - 1], fontsize = 30, rotation = "vertical", verticalalignment = "center")

plt.gcf().text(0.5, 0.96, "Attacks", fontsize = 40, horizontalalignment = "center")
plt.gcf().text(0.01, 0.5, "Defenses", fontsize = 40, rotation = "vertical", verticalalignment = "center")

plt.subplots_adjust(left = 0.05, bottom = 0, right = 1, top = 0.95, wspace = 0, hspace = 0)
plt.savefig("../figures/attack_defense.pdf", bbox_inches = "tight")
plt.show()
