import tensorflow as tf
import importlib
import sys

class PointNetInterface:
    def __init__(self, max_points):
        sys.path.append("pointnet/models")
        model = importlib.import_module("pointnet_cls")

        self.x_pl, self.y_pl = model.placeholder_inputs(1, max_points)
        self.is_training = tf.placeholder(tf.bool, shape = ())

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            logits, end_points = model.get_model(self.x_pl, is_training)

        self.loss = model.get_loss(logits, self.y_pl, end_points)

        # session code
