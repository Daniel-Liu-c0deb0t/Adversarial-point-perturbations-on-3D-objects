import tensorflow as tf
import numpy as np
import importlib
import sys

class PointNetInterface:
    def __init__(self, max_points, fft = False, sink = None, chamfer = False):
        tf.reset_default_graph()

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
            self.sink_source = tf.placeholder(tf.float32, shape = (1, sink, 3))
            self.epsilon = tf.placeholder(tf.float32, shape = ())
            self.lambda_ = tf.placeholder(tf.float32, shape = ())
            self.eta = tf.placeholder(tf.float32, shape = ())

            sinks = tf.get_variable("sinks", dtype = tf.float32, shape = (1, sink, 3))
            self.init_sinks = tf.assign(sinks, self.init_sink_pl)

            dist = tf.linalg.norm(self.sink_source[:, :, tf.newaxis, :] - self.x_clean[:, tf.newaxis, :, :], axis = 3)
            rbf = tf.exp(-((dist / self.epsilon) ** 2))[:, :, :, tf.newaxis]
            perturb = rbf * (sinks[:, :, tf.newaxis, :] - self.x_clean[:, tf.newaxis, :, :])
            self.x_perturb = self.x_clean + tf.tanh(tf.reduce_sum(perturb, axis = 1))

            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_perturb, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            loss_dist = tf.sqrt(tf.reduce_sum((self.x_perturb - self.x_clean) ** 2, axis = (1, 2), keep_dims = True))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.eta)
            self.train = optimizer.minimize(-loss + self.lambda_ * loss_dist, var_list = [sinks])
            self.init_optimizer = tf.variables_initializer(optimizer.variables())
        
        if chamfer:
            self.x_clean_chamfer = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.lambda_chamfer = tf.placeholder(tf.float32, shape = ())
            self.alpha_chamfer = tf.placeholder(tf.float32, shape = ())
            self.eta_chamfer = tf.placeholder(tf.float32, shape = ())
            self.x_chamfer = tf.get_variable("x_chamfer", dtype = tf.float32, shape = self.x_pl.shape.as_list())

            self.init_x_chamfer = tf.assign(self.x_chamfer, self.x_clean_chamfer)
            
            dist = tf.linalg.norm(self.x_chamfer[:, :, tf.newaxis, :] - self.x_clean_chamfer[:, tf.newaxis, :, :], axis = 3)
            dist = tf.where(tf.eye(self.x_pl.shape.as_list()[1], batch_shape = (1,), dtype = tf.bool), tf.fill(dist.shape.as_list(), float("inf")), dist)
            dist = tf.reduce_min(dist, axis = 2, keep_dims = True)
            loss_chamfer = tf.reduce_mean(dist, axis = 1, keep_dims = True)
            
            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_chamfer, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            
            loss_l2 = tf.sqrt(tf.reduce_sum((self.x_chamfer - self.x_clean_chamfer) ** 2, axis = (1, 2), keep_dims = True))
            optimizer_chamfer = tf.train.AdamOptimizer(learning_rate = self.eta_chamfer)
            self.train_chamfer = optimizer_chamfer.minimize(-loss + self.alpha_chamfer * (loss_chamfer + self.lambda_chamfer * loss_l2), var_list = [self.x_chamfer])
            self.init_optimizer_chamfer = tf.variables_initializer(optimizer_chamfer.variables())

    def clean_up(self):
        self.sess.close()

    def pred_fn(self, x):
        return self.sess.run(self.y_pred, feed_dict = {self.x_pl: [x], self.is_training: False})[0].astype(float)

    def reset_sink_fn(self, sinks):
        self.sess.run(self.init_optimizer)
        self.sess.run(self.init_sinks, feed_dict = {self.init_sink_pl: [sinks]})
    
    def reset_chamfer_fn(self, x):
        self.sess.run(self.init_optimizer_chamfer)
        self.sess.run(self.init_x_chamfer, feed_dict = {self.x_clean_chamfer: [x]})

    def x_perturb_sink_fn(self, x, sink_source, epsilon, lambda_):
        return self.sess.run(self.x_perturb, feed_dict = {self.x_clean: [x], self.sink_source: [sink_source], self.epsilon: epsilon, self.lambda_: lambda_, self.is_training: False})[0].astype(float)

    def x_perturb_chamfer_fn(self):
        return self.sess.run(self.x_chamfer)[0].astype(float)
    
    def grad_fn(self, x, y):
        return self.sess.run(self.grad_loss_wrt_x, feed_dict = {self.x_pl: [x], self.y_pl: [y], self.is_training: False})[0].astype(float)

    def grad_freq_fn(self, x, y):
        return self.sess.run(self.grad_loss_wrt_x_freq, feed_dict = {self.x_freq: [x], self.y_pl: [y], self.is_training: False})[0].astype(float)

    def train_sink_fn(self, x, y, sink_source, epsilon, lambda_, eta):
        self.sess.run(self.train, feed_dict = {self.x_clean: [x], self.y_pl: [y], self.sink_source: [sink_source], self.epsilon: epsilon, self.lambda_: lambda_, self.eta: eta, self.is_training: False})

    def train_chamfer_fn(self, x, y, alpha_chamfer, lambda_chamfer, eta_chamfer):
        self.sess.run(self.train_chamfer, feed_dict = {self.x_clean_chamfer: [x], self.y_pl: [y], self.alpha_chamfer: alpha_chamfer, self.lambda_chamfer: lambda_chamfer, self.eta_chamfer: eta_chamfer, self.is_training: False})
    
    def output_grad_fn(self, x):
        res = []

        for i in range(len(self.grad_out_wrt_x)):
            res.append(self.sess.run(self.grad_out_wrt_x[i], feed_dict = {self.x_pl: [x], self.is_training: False})[0].astype(float))

        return np.array(res)
