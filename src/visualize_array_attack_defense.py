from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# defenses x attacks
paths = [
    [
        "../output_save/final/1564113021_pointnet_none_none.npz",
        "../output_save/final/1564130586_pointnet_iter_l2_attack_none.npz",
        "../output_save/final/1564164544_pointnet_iter_l2_attack_n_proj_none.npz",
        "../output_save/final/1564190398_pointnet_iter_l2_attack_n_sampling_none.npz",
        "../output_save/final/1564208936_pointnet_iter_l2_adversarial_sticks_none.npz",
        "../output_save/final/1564228868_pointnet_iter_l2_attack_sinks_none.npz"
    ],
    [
        "../output_save/final/1564113228_pointnet_none_remove_outliers_defense.npz",
        "../output_save/final/1564130794_pointnet_iter_l2_attack_remove_outliers_defense.npz",
        "../output_save/final/1564164752_pointnet_iter_l2_attack_n_proj_remove_outliers_defense.npz",
        "../output_save/final/1564190606_pointnet_iter_l2_attack_n_sampling_remove_outliers_defense.npz",
        "../output_save/final/1564209144_pointnet_iter_l2_adversarial_sticks_remove_outliers_defense.npz",
        "../output_save/final/1564229079_pointnet_iter_l2_attack_sinks_remove_outliers_defense.npz"
    ],
    [
        "../output_save/final/1564114212_pointnet_none_remove_salient_defense.npz",
        "../output_save/final/1564131746_pointnet_iter_l2_attack_remove_salient_defense.npz",
        "../output_save/final/1564165709_pointnet_iter_l2_attack_n_proj_remove_salient_defense.npz",
        "../output_save/final/1564191563_pointnet_iter_l2_attack_n_sampling_remove_salient_defense.npz",
        "../output_save/final/1564210100_pointnet_iter_l2_adversarial_sticks_remove_salient_defense.npz",
        "../output_save/final/1564230037_pointnet_iter_l2_attack_sinks_remove_salient_defense.npz"
    ]
]

xlabels = ["None", "Iter. gradient $L_2$", "Distribution", "Perturb. resample", "Adv. sticks", "Adv. sinks"]
ylabels = ["None", "Remove outliers", "Remove salient"]

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
    scale = 0.7
    plt.gca().auto_scale_xyz((-scale, scale), (-scale, scale), (-scale, scale))
    plt.gca().view_init(30, 120)
    plt.axis("off")

for i, attack_files in enumerate(files):
    for j, f in enumerate(attack_files):
        plt.subplot(len(files), len(files[0]), i * len(files[0]) + j + 1, projection = "3d")
        plt.gca().scatter(*f["x_adv"][match_idx].T, zdir = "y", s = 5, c = f["x_adv"][match_idx].T[2], cmap = "winter")
        scale_plot()

for i in range(len(xlabels)):
    plt.gcf().text(i / (float(len(xlabels)) + 0.3) + 0.5 / len(xlabels) + 0.05, 0.9, xlabels[i], fontsize = 30, horizontalalignment = "center")

for i in range(len(ylabels)):
    plt.gcf().text(0.05, i / (float(len(ylabels)) + 0.1) + 0.5 / len(ylabels), ylabels[-i - 1], fontsize = 30, rotation = "vertical", verticalalignment = "center")

plt.gcf().text(0.5, 0.96, "Attacks", fontsize = 40, horizontalalignment = "center")
plt.gcf().text(0.01, 0.5, "Defenses", fontsize = 40, rotation = "vertical", verticalalignment = "center")

plt.subplots_adjust(left = 0.05, bottom = 0, right = 1, top = 0.95, wspace = 0, hspace = 0)
plt.savefig("../figures/attack_defense.pdf", bbox_inches = "tight")
plt.show()
