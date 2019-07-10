import numpy as np
import adversarial_attacks
import adversarial_defenses
from pointnet_interface import PointNetInterface
from pointnet2_interface import PointNet2Interface
import time
import os

start_time = time.time()

np.random.seed(1234)

models = (
        ("pointnet", PointNetInterface),
        ("pointnet2", PointNet2Interface)
)

test_models = (0,)

attacks = (
        ("none", lambda _a, x, _b, _c: x, {}),
        ("iter_l2_attack_1_proj", adversarial_attacks.iter_l2_attack_1_proj, {"epsilon": 1.0, "n": 10, "tau": 0.05}),
        ("iter_l2_attack", adversarial_attacks.iter_l2_attack, {"epsilon": 1.0, "n": 10}),
        ("mom_l2_attack", adversarial_attacks.mom_l2_attack, {"epsilon": 1.0, "mu": 1.0, "n": 10}),
        ("normal_jitter", adversarial_attacks.normal_jitter, {"epsilon": 1.0, "tau": 0.05}),
        ("iter_l2_attack_n_proj", adversarial_attacks.iter_l2_attack_n_proj, {"epsilon": 1.0, "n": 10, "tau": 0.05}),
        ("mom_l2_attack_n_proj", adversarial_attacks.mom_l2_attack_n_proj, {"epsilon": 1.0, "mu": 1.0, "n": 10, "tau": 0.05}),
        ("iter_l2_attack_1_sampling", adversarial_attacks.iter_l2_attack_1_sampling, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "tri_all_points": True}),
        ("iter_l2_attack_1_sampling_all", adversarial_attacks.iter_l2_attack_1_sampling_all, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "tri_all_points": True}),
        ("iter_l2_attack_n_sampling", adversarial_attacks.iter_l2_attack_n_sampling, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "tri_all_points": True}),
        ("iter_l2_attack_1_sampling_rbf", adversarial_attacks.iter_l2_attack_1_sampling_rbf, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "num_farthest": None, "shape": 5.0}),
        ("iter_l2_attack_n_sampling_rbf", adversarial_attacks.iter_l2_attack_n_sampling_rbf, {"epsilon": 3.0, "n": 10, "k": 500, "kappa": 10, "num_farthest": None, "shape": 5.0}),
        ("iter_l2_attack_top_k", adversarial_attacks.iter_l2_attack_top_k, {"epsilon": 3.0, "n": 10, "top_k": 10}),
        ("iter_l2_adversarial_sticks", adversarial_attacks.iter_l2_adversarial_sticks, {"epsilon": 3.0, "n": 10, "top_k": 10, "sigma": 200}),
        ("iter_l2_attack_fft", adversarial_attacks.iter_l2_attack_fft, {"epsilon": 20.0, "n": 10}),
        ("iter_l2_attack_sinks", adversarial_attacks.iter_l2_attack_sinks, {"eta": 0.001, "epsilon": 1.0, "epsilon_rbf": 0.2, "lambda_": 3.0, "n": 100, "num_sinks": 30})
)

fft = False
sink = True

test_attacks = (15,)

defenses = (
        ("none", lambda _a, x, _b: x, {}),
        ("remove_outliers_defense", adversarial_defenses.remove_outliers_defense, {"top_k": 10, "num_std": 1.0}),
        ("remove_salient_defense", adversarial_defenses.remove_salient_defense, {"top_k": 100})
)

test_defenses = (0,)

class_names_path = "pointnet/data/modelnet40_ply_hdf5_2048/shape_names.txt"
input_data_path = "pointnet/point_clouds.npz"
output_dir = "output_save"
num_point_clouds = 10000
max_points = 1024

try:
    os.makedirs(output_dir)
except OSError:
    pass

class_names = [line.rstrip() for line in open(class_names_path)]

with np.load(input_data_path) as file:
    X = file["points"][:num_point_clouds, :max_points, :]
    Y = file["labels"][:num_point_clouds]
    T = file["faces"][:num_point_clouds, :max_points, :3, :]

for model_idx in test_models:
    model_name = models[model_idx][0]
    model_type = models[model_idx][1]
    model = model_type(max_points, fft = fft, sink = sink)

    for attack_idx in test_attacks:
        attack_name = attacks[attack_idx][0]
        attack_fn = attacks[attack_idx][1]
        attack_dict = attacks[attack_idx][2]

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

            for idx in range(len(X)):
                x = X[idx]
                t = T[idx]
                y_idx = Y[idx] # index of correct output
                y_pred = model.pred_fn(x)
                y_pred_idx = np.argmax(y_pred)

                if y_pred_idx == y_idx: # model makes correct prediction
                    x_adv = defense_fn(model, attack_fn(model, np.copy(x), y_idx, attack_dict), defense_dict)
                    y_adv_pred = model.pred_fn(x_adv)
                    y_adv_pred_idx = np.argmax(y_adv_pred)

                    if y_adv_pred_idx != y_idx:
                        successfully_attacked += 1

                    total_attacked += 1

                    all_attacked.append((x, y_pred, x_adv, y_adv_pred, t))

            all_attacked = list(zip(*all_attacked))
            all_attacked = [np.array(a) for a in all_attacked]
            timestamp = int(time.time())
            save_file = "%s/%d_%s_%s_%s.npz" % (output_dir, timestamp, model_name, attack_name, defense_name)
            np.savez_compressed(save_file, x = all_attacked[0], y_pred = all_attacked[1], x_adv = all_attacked[2], y_adv_pred = all_attacked[3], t = all_attacked[4])

            print("Current time\t%d" % timestamp)
            print("Elapsed time\t%f" % (timestamp - attack_start_time))
            print("Number of attempted attacks\t%d" % total_attacked)
            print("Number of successful attacks\t%d" % successfully_attacked)
            print("Data saved in\t%s" % save_file)
            print()

    model.clean_up()

print("Total elapsed time\t%f" % (time.time() - start_time))
