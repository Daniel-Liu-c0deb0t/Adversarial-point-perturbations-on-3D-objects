import numpy as np
import adversarial_attacks
from pointnet_interface import PointNetInterface
from pointnet2_interface import PointNet2Interface

np.random.seed(1234)

models = (
    ("pointnet", PointNetInterface),
    ("pointnet2", PointNet2Interface)
)

test_models = (0,)

attacks = (
    ("iter_l2_1_proj", adversarial_attacks.iter_l2_1_proj, {}),
    ("iter_l2_n_proj", adversarial_attacks.iter_l2_n_proj, {})
)

test_attacks = (0,)

for model_idx in test_models:
    model_name = models[model_idx][0]
    model_type = models[model_idx][1]
    model = model_type()

    for idx in range(len(X)):
        x = X[idx]
        y_idx = Y[idx] # index of correct output
        y = y_idx # TODO: one hot
        y_pred = model.pred_fn(x)
        y_pred_idx = np.argmax(y_pred)

        if y_pred_idx == y_idx: # model makes correct prediction
            for attack_idx in test_attacks:
                attack_name = attacks[attack_idx][0]
                attack_fn = attacks[attack_idx][1]
                attack_dict = attacks[attack_idx][2]

                x_adv = attack_fn(model.grad_fn, x, y, **attack_dict)
                y_adv_pred = model.pred_fn(x_adv)
                y_adv_pred_idx = np.argmax(y_adv_pred)

                
