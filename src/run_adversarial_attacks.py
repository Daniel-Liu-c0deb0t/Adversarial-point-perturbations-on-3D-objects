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
import argparse

start_time = time.time()

np.random.seed(1234)

models = {
        "pointnet": PointNetInterface,
        "pointnet2": PointNet2Interface,
        "dgcnn": DGCNNInterface
}

attacks = {
        "none": lambda _a, x, _b, _c: x,
        "iter_l2_attack": adversarial_attacks.iter_l2_attack,
        "normal_jitter": adversarial_attacks.normal_jitter,
        "iter_l2_attack_n_proj": adversarial_attacks.iter_l2_attack_n_proj,
        "iter_l2_attack_n_sampling": adversarial_attacks.iter_l2_attack_n_sampling,
        "iter_l2_adversarial_sticks": adversarial_attacks.iter_l2_adversarial_sticks,
        "iter_l2_attack_sinks": adversarial_attacks.iter_l2_attack_sinks,
        "chamfer_attack": adversarial_attacks.chamfer_attack
}

defenses = {
        "none": lambda _a, x, _b: x,
        "remove_outliers_defense": adversarial_defenses.remove_outliers_defense,
        "remove_salient_defense": adversarial_defenses.remove_salient_defense,
        "random_perturb_defense": adversarial_defenses.random_perturb_defense,
        "random_remove_defense": adversarial_defenses.random_remove_defense
}

parser = argparse.ArgumentParser(description = "Adversarial attacks and defenses on PointNet and PointNet++.")
parser.add_argument("--model", required = True, choices = models)
parser.add_argument("--attack", required = True, choices = attacks)
parser.add_argument("--defense", required = True, choices = defenses)
parser.add_argument("--attack-args", required = True, nargs = "*", type = lambda key_value: key_value.split("=", 1))
parser.add_argument("--defense-args", required = True, nargs = "*", type = lambda key_value: key_value.split("=", 1))
args = parser.parse_args()
print(args)

test_model = args.model
test_attack = args.attack
test_defense = args.defense
attack_args = dict(args.attack_args)
defense_args = dict(args.defense_args)

fft = test_attack == "iter_l2_attack_fft"
sink = int(attack_args["num_sinks"]) if test_attack == "iter_l2_attack_sinks" else None
chamfer = test_attack == "chamfer_attack"

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

model_name = test_model
model_type = models[test_model]
model = model_type(max_points, fft = fft, sink = sink, chamfer = chamfer)

attack_name = test_attack
attack_fn = attacks[test_attack]
attack_dict = attack_args

defense_name = test_defense
defense_fn = defenses[test_defense]
defense_dict = defense_args

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
        x_adv = defense_fn(model, attack_fn(model, np.copy(x), y_idx, attack_dict), defense_dict)
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
