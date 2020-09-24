import numpy as np
import h5py
import adversarial_attacks
import adversarial_defenses
from true_proj import project_points_to_triangles
from pointnet_interface import PointNetInterface
from pointnet2_interface import PointNet2Interface
from dgcnn_interface import DGCNNInterface
import time
import os

start_time = time.time()

np.random.seed(1234)

models = (
        ("pointnet", PointNetInterface),
        ("pointnet2", PointNet2Interface),
        ("dgcnn", DGCNNInterface)
)

test_models = (0,)
#test_models = (1,)
#test_models = (2,)

attacks = (
        ("none", lambda _a, x, _b, _c: x, {}),
        ("iter_l2_attack_1_proj", adversarial_attacks.iter_l2_attack_1_proj, {"epsilon": 1.0, "n": 10, "tau": 0.05}),
        #("iter_l2_attack", adversarial_attacks.iter_l2_attack, {"epsilon": 1.0, "n": 100}),
        ("iter_l2_attack", adversarial_attacks.iter_l2_attack, {"epsilon": 2.0, "n": 100}),
        #("iter_l2_attack", adversarial_attacks.iter_l2_attack, {"epsilon": 3.0, "n": 100}),
        ("mom_l2_attack", adversarial_attacks.mom_l2_attack, {"epsilon": 1.0, "mu": 1.0, "n": 10}),
        ("normal_jitter", adversarial_attacks.normal_jitter, {"epsilon": 1.0, "tau": 0.05}),
        ("iter_l2_attack_n_proj", adversarial_attacks.iter_l2_attack_n_proj, {"epsilon": 1.0, "n": 20, "tau": 0.05}),
        ("mom_l2_attack_n_proj", adversarial_attacks.mom_l2_attack_n_proj, {"epsilon": 1.0, "mu": 1.0, "n": 10, "tau": 0.05}),
        ("iter_l2_attack_1_sampling", adversarial_attacks.iter_l2_attack_1_sampling, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "tri_all_points": True}),
        ("iter_l2_attack_1_sampling_all", adversarial_attacks.iter_l2_attack_1_sampling_all, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "tri_all_points": True}),
        #("iter_l2_attack_n_sampling", adversarial_attacks.iter_l2_attack_n_sampling, {"epsilon": 2.0, "n": 100, "k": 0, "kappa": 10, "tri_all_points": True}),
        #("iter_l2_attack_n_sampling", adversarial_attacks.iter_l2_attack_n_sampling, {"epsilon": 2.0, "n": 100, "k": 250, "kappa": 10, "tri_all_points": True}),
        #("iter_l2_attack_n_sampling", adversarial_attacks.iter_l2_attack_n_sampling, {"epsilon": 1.0, "n": 100, "k": 500, "kappa": 10, "tri_all_points": True}),
        ("iter_l2_attack_n_sampling", adversarial_attacks.iter_l2_attack_n_sampling, {"epsilon": 2.0, "n": 100, "k": 500, "kappa": 10, "tri_all_points": True}),
        #("iter_l2_attack_n_sampling", adversarial_attacks.iter_l2_attack_n_sampling, {"epsilon": 3.0, "n": 100, "k": 500, "kappa": 10, "tri_all_points": True}),
        ("iter_l2_attack_1_sampling_rbf", adversarial_attacks.iter_l2_attack_1_sampling_rbf, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "num_farthest": None, "shape": 5.0}),
        ("iter_l2_attack_n_sampling_rbf", adversarial_attacks.iter_l2_attack_n_sampling_rbf, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "num_farthest": None, "shape": 5.0}),
        ("iter_l2_attack_top_k", adversarial_attacks.iter_l2_attack_top_k, {"epsilon": 3.0, "n": 10, "top_k": 10}),
        ("iter_l2_adversarial_sticks", adversarial_attacks.iter_l2_adversarial_sticks, {"epsilon": 2.0, "n": 100, "top_k": 100, "sigma": 400}),
        #("iter_l2_adversarial_sticks2", adversarial_attacks.iter_l2_adversarial_sticks2, {"eta": 0.1, "alpha": 10000.0, "lambda_": 0.01, "n": 20, "top_k": 100, "sigma": 300, "density": 0.5}),
        ("iter_l2_adversarial_sticks2", adversarial_attacks.iter_l2_adversarial_sticks2, {"eta": 0.1, "alpha": 10000.0, "lambda_": 0.01, "n": 20, "top_k": 100, "sigma": 300, "density": 2.0}),
        #("iter_l2_adversarial_sticks2", adversarial_attacks.iter_l2_adversarial_sticks2, {"eta": 0.1, "alpha": 10000.0, "lambda_": 0.01, "n": 20, "top_k": 100, "sigma": 300, "density": 3.5}),
        ("iter_l2_attack_fft", adversarial_attacks.iter_l2_attack_fft, {"epsilon": 20.0, "n": 10}),
        #("iter_l2_attack_sinks", adversarial_attacks.iter_l2_attack_sinks, {"eta": 0.1, "mu": 2.0, "lambda_": 10000.0, "n": 20, "num_sinks": 30}),
        ("iter_l2_attack_sinks", adversarial_attacks.iter_l2_attack_sinks, {"eta": 0.1, "mu": 7.0, "lambda_": 10000.0, "n": 20, "num_sinks": 30}),
        #("iter_l2_attack_sinks", adversarial_attacks.iter_l2_attack_sinks, {"eta": 0.1, "mu": 12.0, "lambda_": 10000.0, "n": 20, "num_sinks": 30}),
        ("chamfer_attack", adversarial_attacks.chamfer_attack, {"eta": 0.1, "alpha": 10000.0, "lambda_": 0.002, "n": 20}),
        ("iter_l2_attack_dropout", adversarial_attacks.iter_l2_attack_dropout, {"epsilon": 2.0, "n": 100})
)

fft = False
sink = 30
sticks = 100
chamfer = True

test_attacks = (0, 2, 5, 9, 14, 16, 17, 18)

defenses = (
        ("none", lambda _a, x, _b: x, {}),
        #("remove_outliers_defense", adversarial_defenses.remove_outliers_defense, {"top_k": 10, "num_std": 0.0}),
        #("remove_outliers_defense", adversarial_defenses.remove_outliers_defense, {"top_k": 10, "num_std": 0.5}),
        ("remove_outliers_defense", adversarial_defenses.remove_outliers_defense, {"top_k": 10, "num_std": 1.0}),
        #("remove_outliers_defense", adversarial_defenses.remove_outliers_defense, {"top_k": 10, "num_std": 1.5}),
        #("remove_outliers_defense", adversarial_defenses.remove_outliers_defense, {"top_k": 10, "num_std": 2.0}),
        #("remove_salient_defense", adversarial_defenses.remove_salient_defense, {"top_k": 100}),
        ("remove_salient_defense", adversarial_defenses.remove_salient_defense, {"top_k": 200}),
        #("remove_salient_defense", adversarial_defenses.remove_salient_defense, {"top_k": 300}),
        #("remove_salient_defense", adversarial_defenses.remove_salient_defense, {"top_k": 400}),
        #("remove_salient_defense", adversarial_defenses.remove_salient_defense, {"top_k": 500}),
        #("random_perturb_defense", adversarial_defenses.random_perturb_defense, {"std": 0.05}),
        ("random_remove_defense", adversarial_defenses.random_remove_defense, {"num_points": 200})
)

test_defenses = (0, 1, 2, 3)
#test_defenses = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

class_names_path = "Adversarial-point-perturbations-on-3D-objects/data/shape_names.txt"
input_data_path = "Adversarial-point-perturbations-on-3D-objects/data/point_clouds.hdf5"
output_dir = "Adversarial-point-perturbations-on-3D-objects/output_save"
num_point_clouds = 10000
max_points = 1024

try:
    os.makedirs(output_dir)
except OSError:
    pass

class_names = [line.rstrip() for line in open(class_names_path)]

with h5py.File(input_data_path, "r") as file:
    X = file["points"][:][:num_point_clouds, :max_points, :]
    Y = file["labels"][:][:num_point_clouds]
    T = file["faces"][:][:num_point_clouds, :, :3, :]

for model_idx in test_models:
    model_name = models[model_idx][0]
    model_type = models[model_idx][1]
    model = model_type(max_points, fft = fft, sink = sink, sticks = sticks, chamfer = chamfer)

    for attack_idx in test_attacks:
        attack_name = attacks[attack_idx][0]
        attack_fn = attacks[attack_idx][1]
        attack_dict = attacks[attack_idx][2]

        attack_cache = {}

        for defense_idx in test_defenses:
            defense_name = defenses[defense_idx][0]
            defense_fn = defenses[defense_idx][1]
            defense_dict = defenses[defense_idx][2]

            print("Model name\t%s" % model_name)
            print("Attack name\t%s" % attack_name)
            print("Attack parameters\t%s" % attack_dict)
            print("Defense name\t%s" % defense_name)
            print("Defense parameters\t%s" % defense_dict)
            attack_start_time = time.time()

            successfully_attacked = 0
            total_attacked = 0
            all_attacked = []
            #avg_dist = 0.0

            for idx in range(len(X)):
                x = X[idx]
                t = T[idx]
                y_idx = Y[idx] # index of correct output
                y_pred = model.pred_fn(x)
                y_pred_idx = np.argmax(y_pred)

                if y_pred_idx == y_idx: # model makes correct prediction
                    if idx in attack_cache:
                        curr_x_adv = attack_cache[idx]
                    else:
                        curr_x_adv = attack_fn(model, np.copy(x), y_idx, attack_dict)
                        attack_cache[idx] = curr_x_adv

                    x_adv = defense_fn(model, np.copy(curr_x_adv), defense_dict)
                    y_adv_pred = model.pred_fn(x_adv)
                    grad_adv = model.grad_fn(x_adv, y_idx)
                    y_adv_pred_idx = np.argmax(y_adv_pred)

                    #if defense_name == "none":
                    #    x_adv_proj = project_points_to_triangles(x_adv, t)
                    #    dist = np.max(np.linalg.norm(x_adv_proj - x_adv, axis = 1))
                    #    avg_dist += dist

                    if y_adv_pred_idx != y_idx:
                        successfully_attacked += 1

                    total_attacked += 1

                    all_attacked.append((x, y_pred, x_adv, y_adv_pred, grad_adv))

            all_attacked = list(zip(*all_attacked))
            all_attacked = [np.array(a) for a in all_attacked]
            timestamp = int(time.time())
            save_file = "%s/%d_%s_%s_%s.npz" % (output_dir, timestamp, model_name, attack_name, defense_name)
            np.savez_compressed(save_file, x = all_attacked[0], y_pred = all_attacked[1], x_adv = all_attacked[2], y_adv_pred = all_attacked[3], grad_adv = all_attacked[4])

            #avg_dist = avg_dist / float(len(X))

            print("Current time\t%d" % timestamp)
            print("Elapsed time\t%f" % (timestamp - attack_start_time))
            print("Number of attempted attacks\t%d" % total_attacked)
            print("Number of successful attacks\t%d" % successfully_attacked)

            #if defense_name == "none":
            #    print("Average Haussdorf distance\t%f" % avg_dist)

            print("Data saved in\t%s" % save_file)
            print()

    model.clean_up()

print("Total elapsed time\t%f" % (time.time() - start_time))
