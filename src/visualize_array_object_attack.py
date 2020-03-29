from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_array_object_attack(label_models=["car", "person", "lamp", "chair", "vase"]):
    # TODO: Document this function
    
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.unicode"] = True

    # attacks
#     paths = [
#         "../output_save/1582094622_pointnet_none_none.npz",
#         "../output_save/1582096639_pointnet_iter_l2_attack_none.npz",
#         "../output_save/1582508068_pointnet_chamfer_attack_none.npz",
#         "../output_save/1582152554_pointnet_iter_l2_attack_n_proj_none.npz",
#         "../output_save/1582182744_pointnet_iter_l2_attack_n_sampling_none.npz",
#         "../output_save/1582186116_pointnet_iter_l2_adversarial_sticks_none.npz",
#         "../output_save/1582453960_pointnet_iter_l2_attack_sinks_none.npz"
#     ]
    
    paths =[ 
        "../output_save/pointnet_none_none.npz",
        "../output_save/pointnet_iter_l2_attack_none.npz",
        "../output_save/pointnet_iter_l2_attack_remove_outliers_defense.npz",
        "../output_save/pointnet_iter_l2_attack_remove_salient_defense.npz",
        "../output_save/pointnet_iter_l2_attack_random_perturb_defense.npz",
        "../output_save/pointnet_iter_l2_attack_random_remove_defense.npz"
    ]

    xlabels = ["None", "Iter. gradient $L_2$", "Chamfer", "Distributional", "Perturb. resample", "Adv. sticks", "Adv. sinks"]

    models = label_models
    offset_idx = [1, 0, 0, 1, 0]
    shape_names = [line.rstrip() for line in open("../data/shape_names.txt")]

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

    for i, idx in enumerate(match_idx):
        for j, f in enumerate(files):
            plt.subplot(len(match_idx), len(files), i * len(files) + j + 1, projection = "3d")
            plt.gca().scatter(*f["x_adv"][idx].T, zdir = "y", s = 5, c = f["x_adv"][idx].T[2], cmap = "winter")
            scale_plot()

    for i in range(len(xlabels)):
        plt.gcf().text(i / (float(len(xlabels)) + 0.35) + 0.5 / len(xlabels) + 0.05, 0.93, xlabels[i], fontsize = 30, horizontalalignment = "center")

    for i in range(len(models)):
        plt.gcf().text(0.05, i / (float(len(models)) + 0.1) + 0.5 / len(models), models[-i - 1].capitalize(), fontsize = 30, rotation = "vertical", verticalalignment = "center")

    plt.gcf().text(0.5, 0.96, "Attacks", fontsize = 40, horizontalalignment = "center")
    plt.gcf().text(0.01, 0.5, "Objects", fontsize = 40, rotation = "vertical", verticalalignment = "center")

    plt.subplots_adjust(left = 0.05, bottom = 0, right = 1, top = 0.95, wspace = 0, hspace = 0)
    # save plot to pdf
    plt.savefig("../figures/object_attack.pdf", bbox_inches = "tight")
    plt.show()

def scale_plot():
    scale = 0.7
    plt.gca().auto_scale_xyz((-scale, scale), (-scale, scale), (-scale, scale))
    plt.gca().view_init(30, 120)
    plt.axis("off")
    
# visualize_array_object_attack()