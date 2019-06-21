import numpy as np
import adversarial_attacks
from pointnet_interface import PointNetInterface
from pointnet2_interface import PointNet2Interface
import time
import os

np.random.seed(1234)

models = (
        ("pointnet", PointNetInterface),
        ("pointnet2", PointNet2Interface)
)

test_models = (0,)

attacks = (
        ("iter_l2_attack_1_proj", adversarial_attacks.iter_l2_attack_1_proj, {"epsilon": 0.1, "n": 10, "tau": 0.05}),
        ("iter_l2_attack_n_proj", adversarial_attacks.iter_l2_attack_n_proj, {"epsilon": 0.1, "n": 10, "tau": 0.05})
)

test_attacks = (0,)

class_names_path = "pointnet/data/modelnet40_ply_hdf5_2048/shape_names.txt"
input_data_path = "pointnet/point_clouds.npz"
output_dir = "output_save"
num_point_clouds = 100
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
    model = model_type(max_points)

    for attack_idx in test_attacks:
        attack_name = attacks[attack_idx][0]
        attack_fn = attacks[attack_idx][1]
        attack_dict = attacks[attack_idx][2]

        print("Model name\t%s" % model_name)
        print("Attack name\t%s" % attack_name)
        print("Attack parameters\t%s" % attack_dict)

        successfully_attacked = 0
        total_attacked = 0
        all_attacked = []

        for idx in range(len(X)):
            x = X[idx]
            t = T[idx]
            y_idx = Y[idx] # index of correct output
            y = np.zeros(len(num_classes))
            y[y_idx] = 1.0
            y_pred = model.pred_fn(x)
            y_pred_idx = np.argmax(y_pred)

            if y_pred_idx == y_idx: # model makes correct prediction
                x_adv = attack_fn(model, x, y, attack_dict)
                y_adv_pred = model.pred_fn(x_adv)
                y_adv_pred_idx = np.argmax(y_adv_pred)

                if y_adv_pred_idx != y_idx:
                    successfully_attacked += 1

                total_attacked += 1

                all_attacked.append((x, y_pred, x_adv, y_adv_pred, t))

        all_attacked = list(zip(*all_attacked))
        all_attacked = [np.array(a) for a in all_attacked]
        timestamp = int(time.time())
        save_file = "%s/%d_%s_%s.npz" % (output_dir, timestamp, model_name, attack_name)
        np.savez_compressed(save_file, x = all_attacked[0], y_pred = all_attacked[1], x_adv = all_attacked[2], y_adv_pred = all_attacked[3], t = all_attacked[4])

        print("Time\t%d" % timestamp)
        print("Number of attempted attacks\t%d" % total_attacked)
        print("Number of successful attacks\t%d" % successfully_attacked)
        print("Data saved in\t%s" % save_file)
        print()

    model.clean_up()
