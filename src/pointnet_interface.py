import tensorflow as tf
import importlib
import sys

class PointNetInterface:
    def __init__(self, max_points):
        checkpoint_path = "pointnet/log/model.ckpt"

        sys.path.append("pointnet/models")
        model = importlib.import_module("pointnet_cls")

        self.x_pl, self.y_pl = model.placeholder_inputs(1, max_points)
        self.is_training = tf.placeholder(tf.bool, shape = ())

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            logits, end_points = model.get_model(self.x_pl, self.is_training)

        self.y_pred = tf.nn.softmax(logits)
        loss = model.get_loss(logits, self.y_pl, end_points)
        self.grad_loss_wrt_x = tf.gradients(loss, self.x_pl)[0]

        self.grad_out_wrt_x = []

        for i in range(40):
            self.grad_out_wrt_x.append(tf.gradients(logits[:, i], self.x_pl)[0])

        # load saved parameters
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.sess = tf.Session(config = config)
        saver.restore(self.sess, checkpoint_path)
        print("Model restored!")

    def clean_up(self):
        self.sess.close()

    def pred_fn(self, x):
        return self.sess.run(self.y_pred, feed_dict = {self.x_pl: [x], self.is_training: False})[0]

    def grad_fn(self, x, y):
        return self.sess.run(self.grad_loss_wrt_x, feed_dict = {self.x_pl: [x], self.y_pl: [y], self.is_training: False})[0]

    def output_grad_fn(self, x):
        res = []

        for i in range(len(self.grad_out_wrt_x)):
            res.append(self.sess.run(self.grad_out_wrt_x[i], feed_dict = {self.x_pl: [x], self.is_training: False})[0])

        return np.array(res)
