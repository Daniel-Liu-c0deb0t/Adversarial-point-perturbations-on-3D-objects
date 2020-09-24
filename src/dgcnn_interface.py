import tensorflow as tf
import numpy as np
import importlib
import sys

class DGCNNInterface:
    def __init__(self, max_points, fft = False, sink = None, sticks = None, dropout = None, chamfer = False):
        tf.reset_default_graph()

        checkpoint_path = "dgcnn/tensorflow/log/model.ckpt"

        sys.path.append("dgcnn/tensorflow/models")
        model = importlib.import_module("dgcnn")

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
            self.x_perturb = self.x_clean + 1.0 * tf.tanh(tf.reduce_sum(perturb, axis = 1))

            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_perturb, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            loss_dist = tf.sqrt(tf.reduce_sum((self.x_perturb - self.x_clean) ** 2, axis = (1, 2), keep_dims = True))
            sink_dist = tf.linalg.norm(1e-6 + sinks[:, :, tf.newaxis, :] - sinks[:, tf.newaxis, :, :], axis = 3)
            diag = tf.eye(sink, batch_shape = (1,))
            loss_sink_dist = tf.reduce_min(tf.where(diag > 0.0, tf.fill((1, sink, sink), float("inf")), sink_dist), axis = (1, 2))[:, tf.newaxis, tf.newaxis]
            norm_loss = tf.reduce_max(tf.linalg.norm(1e-6 + sinks - self.sink_source, axis = 2), axis = 1)[:, tf.newaxis, tf.newaxis]
            optimizer = tf.train.AdamOptimizer(learning_rate = self.eta)
            self.train = optimizer.minimize(-loss + self.lambda_ * (loss_dist - 1.0 * loss_sink_dist + 5.0 * norm_loss), var_list = [sinks])
            self.init_optimizer = tf.variables_initializer([optimizer.get_slot(sinks, name) for name in optimizer.get_slot_names()] + list(optimizer._get_beta_accumulators()))

        if sticks is not None:
            self.x_clean_sticks = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.x_mask_sticks = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.x_init_sticks = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.lambda_sticks = tf.placeholder(tf.float32, shape = ())
            self.alpha_sticks = tf.placeholder(tf.float32, shape = ())
            self.eta_sticks = tf.placeholder(tf.float32, shape = ())
            x_sticks_raw = tf.get_variable("x_sticks_raw", dtype = tf.float32, shape = self.x_pl.shape.as_list())

            self.init_x_sticks = tf.assign(x_sticks_raw, self.x_init_sticks)
            
            self.x_sticks = self.x_clean_sticks + 0.5 * tf.tanh(x_sticks_raw - self.x_clean_sticks) * self.x_mask_sticks
            dist = tf.linalg.norm(1e-6 + self.x_sticks[:, :, tf.newaxis, :] - self.x_clean_sticks[:, tf.newaxis, :, :], axis = 3)
            dist = tf.reduce_min(dist, axis = 2, keep_dims = True)
            loss_sticks = tf.reduce_sum(dist, axis = 1, keep_dims = True) / float(sticks)
            
            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_sticks, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            
            loss_l2 = tf.sqrt(tf.reduce_sum((self.x_sticks - self.x_clean_sticks) ** 2, axis = (1, 2), keep_dims = True))
            optimizer_sticks = tf.train.AdamOptimizer(learning_rate = self.eta_sticks)
            self.train_sticks = optimizer_sticks.minimize(-loss + self.alpha_sticks * (loss_sticks + self.lambda_sticks * loss_l2), var_list = [x_sticks_raw])
            self.init_optimizer_sticks = tf.variables_initializer([optimizer_sticks.get_slot(x_sticks_raw, name) for name in optimizer_sticks.get_slot_names()] + list(optimizer_sticks._get_beta_accumulators()))

        if dropout is not None:
            self.x_clean_dropout = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.x_prev_dropout = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.x_mask_dropout = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.x_init_dropout = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.lambda_dropout = tf.placeholder(tf.float32, shape = ())
            self.alpha_dropout = tf.placeholder(tf.float32, shape = ())
            self.eta_dropout = tf.placeholder(tf.float32, shape = ())
            x_dropout_raw = tf.get_variable("x_dropout_raw", dtype = tf.float32, shape = self.x_pl.shape.as_list())

            self.init_x_dropout = tf.assign(x_dropout_raw, self.x_init_dropout)
            
            self.x_dropout = (self.x_prev_dropout + 1.0 * tf.tanh(x_dropout_raw - self.x_prev_dropout)) * self.x_mask_dropout
            dist = tf.linalg.norm(1e-6 + self.x_dropout[:, :, tf.newaxis, :] - self.x_clean_dropout[:, tf.newaxis, :, :] * self.x_mask_dropout[:, tf.newaxis, :, :], axis = 3)
            dist = tf.reduce_min(dist, axis = 2, keep_dims = True)
            loss_dropout = tf.reduce_sum(dist, axis = 1, keep_dims = True) / float(dropout)
            
            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_dropout, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            
            loss_l2 = tf.sqrt(tf.reduce_sum((self.x_dropout - self.x_clean_dropout * self.x_mask_dropout) ** 2, axis = (1, 2), keep_dims = True))
            optimizer_dropout = tf.train.AdamOptimizer(learning_rate = self.eta_dropout)
            self.train_dropout = optimizer_dropout.minimize(-loss + self.alpha_dropout * (loss_dropout + self.lambda_dropout * loss_l2), var_list = [x_dropout_raw])
            self.init_optimizer_dropout = tf.variables_initializer([optimizer_dropout.get_slot(x_dropout_raw, name) for name in optimizer_dropout.get_slot_names()] + list(optimizer_dropout._get_beta_accumulators()))
        
        if chamfer:
            self.x_clean_chamfer = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.x_init_chamfer = tf.placeholder(tf.float32, shape = self.x_pl.shape.as_list())
            self.lambda_chamfer = tf.placeholder(tf.float32, shape = ())
            self.alpha_chamfer = tf.placeholder(tf.float32, shape = ())
            self.eta_chamfer = tf.placeholder(tf.float32, shape = ())
            x_chamfer_raw = tf.get_variable("x_chamfer_raw", dtype = tf.float32, shape = self.x_pl.shape.as_list())

            self.init_x_chamfer = tf.assign(x_chamfer_raw, self.x_init_chamfer)
            
            self.x_chamfer = self.x_clean_chamfer + 1.0 * tf.tanh(x_chamfer_raw - self.x_clean_chamfer)
            dist = tf.linalg.norm(self.x_chamfer[:, :, tf.newaxis, :] - self.x_clean_chamfer[:, tf.newaxis, :, :], axis = 3)
            dist = tf.reduce_min(dist, axis = 2, keep_dims = True)
            loss_chamfer = tf.reduce_mean(dist, axis = 1, keep_dims = True)
            
            with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
                logits, end_points = model.get_model(self.x_chamfer, self.is_training)

            loss = model.get_loss(logits, self.y_pl, end_points)
            
            loss_l2 = tf.sqrt(tf.reduce_sum((self.x_chamfer - self.x_clean_chamfer) ** 2, axis = (1, 2), keep_dims = True))
            optimizer_chamfer = tf.train.AdamOptimizer(learning_rate = self.eta_chamfer)
            self.train_chamfer = optimizer_chamfer.minimize(-loss + self.alpha_chamfer * (loss_chamfer + self.lambda_chamfer * loss_l2), var_list = [x_chamfer_raw])
            self.init_optimizer_chamfer = tf.variables_initializer([optimizer_chamfer.get_slot(x_chamfer_raw, name) for name in optimizer_chamfer.get_slot_names()] + list(optimizer_chamfer._get_beta_accumulators()))

    def clean_up(self):
        self.sess.close()

    def pred_fn(self, x):
        return self.sess.run(self.y_pred, feed_dict = {self.x_pl: [x], self.is_training: False})[0].astype(float)

    def reset_sink_fn(self, sinks):
        self.sess.run(self.init_optimizer)
        self.sess.run(self.init_sinks, feed_dict = {self.init_sink_pl: [sinks]})
    
    def reset_chamfer_fn(self, x):
        self.sess.run(self.init_optimizer_chamfer)
        self.sess.run(self.init_x_chamfer, feed_dict = {self.x_init_chamfer: [x]})

    def reset_sticks_fn(self, x):
        self.sess.run(self.init_optimizer_sticks)
        self.sess.run(self.init_x_sticks, feed_dict = {self.x_init_sticks: [x]})

    def reset_dropout_fn(self, x):
        self.sess.run(self.init_optimizer_dropout)
        self.sess.run(self.init_x_dropout, feed_dict = {self.x_init_dropout: [x]})

    def x_perturb_sink_fn(self, x, sink_source, epsilon, lambda_):
        return self.sess.run(self.x_perturb, feed_dict = {self.x_clean: [x], self.sink_source: [sink_source], self.epsilon: epsilon, self.lambda_: lambda_, self.is_training: False})[0].astype(float)

    def x_perturb_chamfer_fn(self, x):
        return self.sess.run(self.x_chamfer, feed_dict = {self.x_clean_chamfer: [x]})[0].astype(float)

    def x_perturb_sticks_fn(self, x, mask):
        return self.sess.run(self.x_sticks, feed_dict = {self.x_clean_sticks: [x], self.x_mask_sticks: [mask]})[0].astype(float)

    def x_perturb_dropout_fn(self, x, mask):
        return self.sess.run(self.x_dropout, feed_dict = {self.x_prev_dropout: [x], self.x_mask_dropout: [mask]})[0].astype(float)
    
    def grad_fn(self, x, y):
        return self.sess.run(self.grad_loss_wrt_x, feed_dict = {self.x_pl: [x], self.y_pl: [y], self.is_training: False})[0].astype(float)

    def grad_freq_fn(self, x, y):
        return self.sess.run(self.grad_loss_wrt_x_freq, feed_dict = {self.x_freq: [x], self.y_pl: [y], self.is_training: False})[0].astype(float)

    def train_sink_fn(self, x, y, sink_source, epsilon, lambda_, eta):
        self.sess.run(self.train, feed_dict = {self.x_clean: [x], self.y_pl: [y], self.sink_source: [sink_source], self.epsilon: epsilon, self.lambda_: lambda_, self.eta: eta, self.is_training: False})

    def train_chamfer_fn(self, x, y, alpha_chamfer, lambda_chamfer, eta_chamfer):
        self.sess.run(self.train_chamfer, feed_dict = {self.x_clean_chamfer: [x], self.y_pl: [y], self.alpha_chamfer: alpha_chamfer, self.lambda_chamfer: lambda_chamfer, self.eta_chamfer: eta_chamfer, self.is_training: False})

    def train_sticks_fn(self, x, y, mask, alpha_sticks, lambda_sticks, eta_sticks):
        self.sess.run(self.train_sticks, feed_dict = {self.x_clean_sticks: [x], self.y_pl: [y], self.x_mask_sticks: [mask], self.alpha_sticks: alpha_sticks, self.lambda_sticks: lambda_sticks, self.eta_sticks: eta_sticks, self.is_training: False})

    def train_dropout_fn(self, x, x_prev, y, mask, alpha_dropout, lambda_dropout, eta_dropout):
        self.sess.run(self.train_dropout, feed_dict = {self.x_clean_dropout: [x], self.x_prev_dropout: [x_prev], self.y_pl: [y], self.x_mask_dropout: [mask], self.alpha_dropout: alpha_dropout, self.lambda_dropout: lambda_dropout, self.eta_dropout: eta_dropout, self.is_training: False})
    
    def output_grad_fn(self, x):
        res = []

        for i in range(len(self.grad_out_wrt_x)):
            res.append(self.sess.run(self.grad_out_wrt_x[i], feed_dict = {self.x_pl: [x], self.is_training: False})[0].astype(float))

        return np.array(res)
