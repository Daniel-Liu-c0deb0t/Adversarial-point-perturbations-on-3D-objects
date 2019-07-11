import tensorflow as tf
import numpy as np
import importlib
import sys

class PointNetInterface:
    def __init__(self, max_points, fft = False, sink = None):
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
        self.sess = tf.Session(config = config)
        saver.restore(self.sess, checkpoint_path)
        print("Model restored!")

        if fft:
            self.x_freq = tf.placeholder(tf.complex64, shape = self.x_pl.shape.as_list())
            self.x_time = tf.real(tf.ifft2d(self.x_freq))

            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_time, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            self.grad_loss_wrt_x_freq = tf.gradients(loss, self.x_freq)[0]

        if sink is not None:
            self.x_clean = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.init_sink_pl = tf.placeholder(tf.float32, shape = (1, sink, 3))
            self.sink_sources = tf.placeholder(tf.float32, shape = (1, sink, 3))
            self.epsilon = tf.placeholder(tf.float32, shape = ())
            self.lambda_ = tf.placeholder(tf.float32, shape = ())
            self.eta = tf.placeholder(tf.float32, shape = ())

            sinks = tf.get_variable("sinks", dtype = tf.float32, shape = (1, sink, 3))
            self.init_sinks = tf.assign(sinks, self.init_sink_pl)

            dist = tf.linalg.norm(self.sink_sources[:, :, tf.newaxis, :] - self.x_clean[:, tf.newaxis, :, :], axis = 3)
            rbf = tf.exp(-((dist / self.epsilon) ** 2))[:, :, :, tf.newaxis]
            perturb = rbf * (sinks[:, :, tf.newaxis, :] - self.x_clean[:, tf.newaxis, :, :])
            perturb = tf.where(tf.is_finite(perturb), perturb, tf.zeros_like(perturb))
            self.x_perturb = self.x_clean + tf.reduce_sum(perturb, axis = 1)

            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_perturb, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            loss_dist = tf.sqrt(tf.reduce_sum((self.x_perturb - self.x_clean) ** 2, axis = (1, 2), keepdims = True))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.eta)
            self.train = optimizer.minimize(-loss + self.lambda_ * loss_dist, var_list = [sinks])
            self.init_optimizer = tf.variables_initializer(optimizer.variables())

    def clean_up(self):
        self.sess.close()

    def pred_fn(self, x):
        return self.sess.run(self.y_pred, feed_dict = {self.x_pl: [x], self.is_training: False})[0]

    def reset_sink_fn(self, sinks):
        self.sess.run(self.init_optimizer)
        self.sess.run(self.init_sinks, feed_dict = {self.init_sink_pl: [sinks]})

    def x_perturb_sink_fn(self, x, sink_sources, epsilon, lambda_):
        return self.sess.run(self.x_perturb, feed_dict = {self.x_clean: [x], self.sink_sources: [sink_sources], self.epsilon: epsilon, self.lambda_: lambda_, self.is_training: False})[0]

    def grad_fn(self, x, y):
        return self.sess.run(self.grad_loss_wrt_x, feed_dict = {self.x_pl: [x], self.y_pl: [y], self.is_training: False})[0]

    def grad_freq_fn(self, x, y):
        return self.sess.run(self.grad_loss_wrt_x_freq, feed_dict = {self.x_freq: [x], self.y_pl: [y], self.is_training: False})[0]

    def train_sink_fn(self, x, y, sink_sources, epsilon, lambda_, eta):
        self.sess.run(self.train, feed_dict = {self.x_clean: [x], self.y_pl: [y], self.sink_sources: [sink_sources], self.epsilon: epsilon, self.lambda_: lambda_, self.eta: eta, self.is_training: False})

    def output_grad_fn(self, x):
        res = []

        for i in range(len(self.grad_out_wrt_x)):
            res.append(self.sess.run(self.grad_out_wrt_x[i], feed_dict = {self.x_pl: [x], self.is_training: False})[0])

        return np.array(res)
