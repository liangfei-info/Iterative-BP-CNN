import numpy as np
import tensorflow as tf

# this file defines classes for data io
class TrainingDataIO:
    def __init__(self, feature_filename, label_filename, total_trainig_samples, feature_length, label_length):
        print("Construct the data IO class for training!\n")
        self.fin_label = open(label_filename, "rb")
        self.fin_feature = open(feature_filename, "rb")
        self.total_trainig_samples = total_trainig_samples
        self.feature_length = feature_length
        self.label_length = label_length

    def __del__(self):
        print("Delete the data IO class!\n")
        self.fin_feature.close()
        self.fin_label.close()

    def load_next_mini_batch(self, mini_batch_size, factor_of_start_pos=1):
        # the function is to load the next batch where the datas in the batch are from a continuous memory block
        # the start position for reading data must be a multiple of factor_of_start_pos
        remain_samples = mini_batch_size
        sample_id = np.random.randint(self.total_trainig_samples)   # output a single value which is less than total_trainig_samples
        features = np.zeros((0))
        labels = np.zeros((0))
        if mini_batch_size > self.total_trainig_samples:
            print("Mini batch size should not be larger than total sample size!\n")
        self.fin_feature.seek((self.feature_length * 4) * (sample_id//factor_of_start_pos*factor_of_start_pos), 0)  # float32 = 4 bytes = 32 bits
        self.fin_label.seek((self.label_length * 4) * (sample_id//factor_of_start_pos*factor_of_start_pos), 0)

        while 1:
            new_feature = np.fromfile(self.fin_feature, np.float32, self.feature_length * remain_samples)
            new_label = np.fromfile(self.fin_label, np.float32, self.label_length * remain_samples)
            features = np.concatenate((features, new_feature))
            labels = np.concatenate((labels, new_label))
            remain_samples -= len(new_feature) // self.feature_length
            if remain_samples == 0:
                break
            self.fin_feature.seek(0, 0)
            self.fin_label.seek(0, 0)
        features = features.reshape((mini_batch_size, self.feature_length))
        labels = labels.reshape((mini_batch_size, self.label_length))
        return features, labels


class TestDataIO:
    def __init__(self, feature_filename, label_filename, test_sample_num, feature_length, label_length):
        self.fin_label = open(label_filename, "rb")
        self.fin_feature = open(feature_filename, "rb")
        self.test_sample_num = test_sample_num
        self.feature_length = feature_length
        self.label_length = label_length
        self.all_features = np.zeros(0)
        self.all_labels = np.zeros(0)
        self.data_position = 0

    def __del__(self):
        self.fin_feature.close()
        self.fin_label.close()

    def seek_file_to_zero(self):  # reset the file pointer to the start of the file
        self.fin_feature.seek(0, 0)
        self.fin_label.seek(0, 0)

    def load_batch_for_test(self, batch_size):
        if batch_size > self.test_sample_num:
            print("Batch size should not be larger than total sample size!\n")
        if np.size(self.all_features) == 0:
            self.all_features = np.fromfile(self.fin_feature, np.float32, self.feature_length * self.test_sample_num)
            self.all_labels = np.fromfile(self.fin_label, np.float32, self.label_length * self.test_sample_num)
            self.all_features = np.reshape(self.all_features, [self.test_sample_num, self.feature_length])
            self.all_labels = np.reshape(self.all_labels, [self.test_sample_num, self.label_length])

        features = self.all_features[self.data_position:(self.data_position + batch_size), :]
        labels = self.all_labels[self.data_position:(self.data_position + batch_size), :]
        self.data_position += batch_size
        if self.data_position >= self.test_sample_num:
            self.data_position = 0
        return features, labels


class NoiseIO:
    def __init__(self, blk_len, read_from_file, noise_file, cov_1_2_mat_file_gen_noise, rng_seed=None):
        self.read_from_file = read_from_file
        self.blk_len = blk_len
        self.rng_seed = rng_seed
        if read_from_file:
            self.fin_noise = open(noise_file, 'rb')
        else:
            self.rng = np.random.RandomState(rng_seed)
            fin_cov_file = open(cov_1_2_mat_file_gen_noise, 'rb')
            cov_1_2_mat = np.fromfile(fin_cov_file, np.float32, blk_len*blk_len)
            cov_1_2_mat = np.reshape(cov_1_2_mat, [blk_len, blk_len])
            fin_cov_file.close()
            ## output parts of the correlation function for check
            cov_func = np.matmul(cov_1_2_mat, cov_1_2_mat)
            print('Correlation function of channel noise: ')
            print(cov_func[0,0:10])
            self.awgn_noise = tf.placeholder(dtype=tf.float32, shape=[None, blk_len])
            self.noise_tf = tf.matmul(self.awgn_noise, cov_1_2_mat)
            self.sess = tf.Session()


    def __del__(self):
        if self.read_from_file:
            self.fin_noise.close()
        else:
            self.sess.close()

    def reset_noise_generator(self): # this function resets the file pointer or the rng generator to generate the same noise data
        if self.read_from_file:
            self.fin_noise.seek(0, 0)
        else:
            self.rng = np.random.RandomState(self.rng_seed)


    def generate_noise(self, batch_size):
        if self.read_from_file:
            noise = np.fromfile(self.fin_noise, np.float32, batch_size * self.blk_len)
            noise = np.reshape(noise, [batch_size, self.blk_len])
        else:
            noise_awgn = self.rng.randn(batch_size, self.blk_len)
            noise_awgn = noise_awgn.astype(np.float32)
            noise = self.sess.run(self.noise_tf, feed_dict={self.awgn_noise: noise_awgn})

        return noise