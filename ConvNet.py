import numpy as np
import tensorflow as tf
import DataIO
import os
import datetime

# this file defines a class for building and training a conv network
class ConvNet:
    def __init__(self, net_config_in, train_config_in, net_id):
        self.net_config = net_config_in
        self.train_config = train_config_in
        self.conv_filter_name = {}
        self.bias_name = {}
        self.conv_filter = {}
        self.bias = {}
        self.best_conv_filter = {}
        self.best_bias = {}
        self.assign_best_conv_filter = {}
        self.assign_best_bias = {}
        self.net_id = net_id
        self.res_noise_power_dict = {}
        self.res_noise_pdf_dict = {}

    def build_network(self, built_for_training=False):  # build a network similar with SRCNN, a fully-convolutional netowrk for channel noise estimation
        # built_for_train: denote whether the network is build for training or test. If for test, xavier initialization is not needed since the model will be loaded later.
        x_in = tf.placeholder(tf.float32, [None, self.net_config.feature_length])  # input data
        x_in_reshape = tf.reshape(x_in, (-1, self.net_config.feature_length, 1, 1))
        layer_output = {}

        for layer in range(self.net_config.total_layers):  # construct layers
            self.conv_filter_name[layer] = format("conv_layer%d" % (layer))
            self.bias_name[layer] = format("b%d" % (layer))

            if layer == 0:
                x_input = x_in_reshape
                in_channels = 1
            else:
                x_input = layer_output[layer - 1]
                in_channels = self.net_config.feature_map_nums[layer - 1]
            out_channels = self.net_config.feature_map_nums[layer]

            if built_for_training:
                # xavier initialization for training
                self.conv_filter[layer] = tf.get_variable(name=self.conv_filter_name[layer], shape=[self.net_config.filter_sizes[layer], 1, in_channels, out_channels],
                                                          dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                self.bias[layer] = tf.get_variable(name=self.bias_name[layer], shape=[out_channels],
                                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                self.best_conv_filter[layer] = tf.Variable(tf.ones([self.net_config.filter_sizes[layer], 1, in_channels, out_channels], tf.float32), dtype=tf.float32)
                self.best_bias[layer] = tf.Variable(tf.ones([out_channels], tf.float32), dtype=tf.float32)
                self.assign_best_conv_filter[layer] = self.best_conv_filter[layer].assign(self.conv_filter[layer])
                self.assign_best_bias[layer] = self.best_bias[layer].assign(self.bias[layer])
            else:
                # just build tensors for testing and their values will be loaded later.
                self.conv_filter[layer] = tf.Variable(tf.random_normal([self.net_config.filter_sizes[layer], 1, in_channels, out_channels], 0, 1, tf.float32), dtype=tf.float32,
                                                      name=self.conv_filter_name[layer])
                self.bias[layer] = tf.Variable(tf.random_normal([out_channels], 0, 1, tf.float32), dtype=tf.float32, name=self.bias_name[layer])

            if layer == self.net_config.total_layers - 1:
                layer_output[layer] = tf.nn.conv2d(x_input, self.conv_filter[layer], [1, 1, 1, 1], 'SAME') + self.bias[layer]
            else:
                # Activation Function
                layer_output[layer] = tf.nn.relu(tf.nn.conv2d(x_input, self.conv_filter[layer], [1, 1, 1, 1], 'SAME') + self.bias[layer])

        y_out = layer_output[self.net_config.total_layers - 1]
        y_out = tf.reshape(y_out, (-1, self.net_config.label_length))

        return x_in, y_out

    # restore_network_with_model_id: to restore network from the folder with model id. This is used to load model from the specified model folder.
    def restore_network_with_model_id(self, sess_in, restore_layers_in, model_id):
        # restore some layers
        save_dict = {}
        if restore_layers_in > 0:
            for layer in range(restore_layers_in):
                save_dict[self.conv_filter_name[layer]] = self.conv_filter[layer]
                save_dict[self.bias_name[layer]] = self.bias[layer]
            model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str)-1)]
            model_folder = format("%snetid%d_model%s" % (self.net_config.model_folder, self.net_id, model_id_str))
            restore_model_name = format("%s/model.ckpt" % model_folder)
            saver_restore = tf.train.Saver(save_dict)
            saver_restore.restore(sess_in, restore_model_name)
            print("Restore the first %d layers.\n" % restore_layers_in)

    def save_network_temporarily(self, sess_in):
        for layer in range(self.net_config.save_layers):
            sess_in.run(self.assign_best_conv_filter[layer])
            sess_in.run(self.assign_best_bias[layer])

    def save_network(self, sess_in, model_id):
        # save network
        save_dict = {}
        for layer in range(self.net_config.save_layers):
            save_dict[self.conv_filter_name[layer]] = self.best_conv_filter[layer]
            save_dict[self.bias_name[layer]] = self.best_bias[layer]

        model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str) - 1)]
        save_model_folder = format("%snetid%d_model%s" % (self.net_config.model_folder, self.net_id, model_id_str))
        # model(a_b_c) means the 3rd network model indexed by c is trained based on the 1st network model indexed by a and the 2nd network model indexed by b. a,b,
        # c could be the same.

        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)
        save_model_name = format("%s/model.ckpt" % (save_model_folder))
        saver_save = tf.train.Saver(save_dict)
        saver_save.save(sess_in, save_model_name)
        print("Save %d layers.\n" % self.net_config.save_layers)

    def test_network_online(self, dataio, x_in, y_label, orig_loss, loss_after_training, calc_org_loss, sess_in):
        # this function is used to test the network loss online when training network
        remain_samples = self.train_config.test_sample_num
        load_batch_size = self.train_config.test_minibatch_size
        ave_loss_after_train = 0.0
        ave_org_loss = 0.0
        while remain_samples > 0:
            if remain_samples < self.train_config.test_minibatch_size:
                load_batch_size = remain_samples

            batch_xs, batch_ys = dataio.load_batch_for_test(load_batch_size)  # features, labels
            if calc_org_loss:
                loss_after_training_value, orig_loss_value = sess_in.run([loss_after_training, orig_loss], feed_dict={x_in: batch_xs, y_label: batch_ys})
                ave_org_loss += orig_loss_value * load_batch_size
            else:
                loss_after_training_value = sess_in.run(loss_after_training, feed_dict={x_in: batch_xs, y_label: batch_ys})
            remain_samples -= load_batch_size
            ave_loss_after_train += loss_after_training_value * load_batch_size

        if calc_org_loss:
            ave_org_loss /= np.double(self.train_config.test_sample_num)
        ave_loss_after_train /= np.double(self.train_config.test_sample_num)
        if calc_org_loss:
            print("Test loss: %f, orig loss: %f" % (ave_loss_after_train, ave_org_loss))
        else:
            print(ave_loss_after_train)
        return ave_loss_after_train, ave_org_loss

    ## normality test
    def calc_normality_test(self, residual_noise, batch_size, batch_size_for_norm_test):
        groups = int(batch_size // batch_size_for_norm_test)
        residual_noise = tf.reshape(residual_noise, [groups, self.net_config.label_length * batch_size_for_norm_test])

        mean = tf.reduce_mean(residual_noise, axis=1)
        mean = tf.reshape(mean, [groups, 1])
        variance = tf.reduce_mean(tf.square(residual_noise - mean), axis=1)
        variance = tf.reshape(variance, [groups, 1])
        moment_3rd = tf.reduce_mean(tf.pow(residual_noise - mean, 3), axis=1)
        moment_3rd = tf.reshape(moment_3rd, [groups, 1])
        moment_4th = tf.reduce_mean(tf.pow(residual_noise - mean, 4), axis=1)
        moment_4th = tf.reshape(moment_4th, [groups, 1])
        skewness = tf.divide(moment_3rd, tf.pow(variance, 3 / 2.0) + 1e-10)
        kurtosis = tf.divide(moment_4th, tf.square(variance) + 1e-10)
        norm_test = tf.reduce_mean(tf.square(skewness) + 0.25 * tf.square(kurtosis-3))

        return norm_test

    def train_network(self, model_id):
        start = datetime.datetime.now()
        dataio = DataIO.TrainingDataIO(self.train_config.training_feature_file, self.train_config.training_label_file,
                                       self.train_config.training_sample_num, self.net_config.feature_length,
                                       self.net_config.label_length)  # construct class for loading data
        dataio_test = DataIO.TestDataIO(self.train_config.test_feature_file, self.train_config.test_label_file,
                                   self.train_config.test_sample_num, self.net_config.feature_length,
                                   self.net_config.label_length)  # construct class for loading data

        x_in, y_out = self.build_network(True)  # build the network for training

        # define loss function
        y_label = tf.placeholder(tf.float32, [None, self.net_config.label_length])

        ## normality test
        if self.train_config.normality_test_enabled:
            training_loss =  self.calc_normality_test(y_label - y_out, self.train_config.training_minibatch_size, 1)
            orig_loss_for_test = self.calc_normality_test(y_label - x_in, self.train_config.test_minibatch_size, 1)
            test_loss = self.calc_normality_test(y_label - y_out, self.train_config.test_minibatch_size, 1)
            if self.train_config.normality_lambda != np.inf:
                training_loss = tf.reduce_mean(tf.square(y_out - y_label)) + training_loss * self.train_config.normality_lambda
                orig_loss_for_test = tf.reduce_mean(tf.square(y_label - x_in)) + orig_loss_for_test * self.train_config.normality_lambda
                test_loss = tf.reduce_mean(tf.square(y_label - y_out)) + test_loss * self.train_config.normality_lambda
        else:
            training_loss = tf.reduce_mean(tf.square(y_out - y_label))
            orig_loss_for_test = tf.reduce_mean(tf.square(y_label - x_in))
            test_loss = training_loss

        # SGD_Adam
        train_step = tf.train.AdamOptimizer().minimize(training_loss)

        # init operation
        init = tf.global_variables_initializer()

        # create a session
        sess = tf.Session()
        sess.run(init)

        self.restore_network_with_model_id(sess, self.net_config.restore_layers, model_id)

        # calculate the loss before training and assign it to min_loss
        min_loss, ave_org_loss = self.test_network_online(dataio_test, x_in, y_label, orig_loss_for_test, test_loss, True, sess)

        self.save_network_temporarily(sess)
        # Train
        count = 0
        epoch = 0
        print('Iteration\tLoss')

        while epoch < self.train_config.epoch_num:
            epoch += 1
            batch_xs, batch_ys = dataio.load_next_mini_batch(self.train_config.training_minibatch_size)
            sess.run([train_step], feed_dict={x_in: batch_xs, y_label: batch_ys})
            if epoch % 500 == 0 or epoch == self.train_config.epoch_num:
                print(epoch)
                ave_loss_after_train, _ = self.test_network_online(dataio_test, x_in, y_label, orig_loss_for_test, test_loss, False, sess)
                if ave_loss_after_train < min_loss:
                    min_loss = ave_loss_after_train
                    self.save_network_temporarily(sess)
                    count = 0
                else:
                    count += 1
                    if count >= 8:  # no patience
                        break
        self.save_network(sess, model_id)
        sess.close()
        end = datetime.datetime.now()
        print('Final minimum loss: %f' % min_loss)
        print('Used time for training: %ds'% (end-start).seconds)

    def get_res_noise_power(self, model_id, SNRset=np.zeros(0)):
        if self.res_noise_power_dict.__len__() == 0:
            # if len(model_id) > self.net_id+1, discard redundant parts.
            model_id_str = np.array2string(model_id[0:(self.net_id+1)], separator='_', formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str)-1)]
            residual_noise_power_file = format("%sresidual_noise_property_netid%d_model%s.txt" % (self.net_config.residual_noise_property_folder, self.net_id, model_id_str))
            data = np.loadtxt(residual_noise_power_file, dtype=np.float32)
            shape_data = np.shape(data)
            if np.size(shape_data) == 1:
                self.res_noise_power_dict[data[0]] = data[1:shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_power_dict[data[i, 0]] = data[i, 1:shape_data[1]]
        return self.res_noise_power_dict

    def get_res_noise_pdf(self, model_id):
        if self.res_noise_pdf_dict.__len__() == 0:
            # if len(model_id) > self.net_id+1, discard redundant parts.
            model_id_str = np.array2string(model_id[0:(self.net_id+1)], separator='_', formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str)-1)]
            residual_noise_pdf_file = format("%sresidual_noise_property_netid%d_model%s.txt" % (self.net_config.residual_noise_property_folder, self.net_id, model_id_str))
            data = np.loadtxt(residual_noise_pdf_file, dtype=np.float32)
            shape_data = np.shape(data)
            if np.size(shape_data) == 1:
                self.res_noise_pdf_dict[data[0]] = data[1:shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_pdf_dict[data[i, 0]] = data[i, 1:shape_data[1]]
        return self.res_noise_pdf_dict


