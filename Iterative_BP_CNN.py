# This file contains some functions used to implement an iterative BP and denoising system for channel decoding.
# The denoising is implemented through CNN.
# The system architecture can be briefly denoted as BP-CNN-BP-CNN-BP...

import numpy as np
import datetime
import BP_Decoder
import tensorflow as tf
import ConvNet
import LinearBlkCodes as lbc
import DataIO


# emprical distribution
def stat_prob(x, prob):
    qstep = 0.01
    min_v = -10
    x = np.reshape(x, [1, np.size(x)])
    [hist, _] = np.histogram(x, np.int32(np.round(2*(-min_v) / qstep)),[min_v,-min_v])
    if np.size(prob) == 0:
        prob = hist
    else:
        prob = prob + hist

    return prob

# denoising and calculate LLR for next decoding
def denoising_and_calc_LLR_awgn(res_noise_power, y_receive, output_pre_decoder, net_in, net_out, sess):
    # estimate noise with cnn denoiser
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    noise_after_cnn = sess.run(net_out, feed_dict={net_in: noise_before_cnn})
    # calculate the LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
    return LLR

def calc_LLR_epdf(prob, s_mod_plus_res_noise):
    qstep = 0.01
    min_v = -10
    id = ((s_mod_plus_res_noise - 1 - min_v) / qstep).astype(np.int32)
    id[id < 0] = 0
    id[id > np.size(prob) - 1] = np.size(prob) - 1
    p0 = prob[id]
    id = ((s_mod_plus_res_noise + 1 - min_v) / qstep).astype(np.int32)
    id[id < 0] = 0
    id[id > np.size(prob) - 1] = np.size(prob) - 1
    p1 = prob[id]
    LLR = np.log(np.divide(p0 + 1e-7, p1 + 1e-7))
    return LLR

def denoising_and_calc_LLR_epdf(prob, y_receive, output_pre_decoder, net_in, net_out, sess):
    # estimate noise with cnn denoiser
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    noise_after_cnn = sess.run(net_out, feed_dict={net_in: noise_before_cnn})
    # calculate the LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = calc_LLR_epdf(prob, s_mod_plus_res_noise)
    return LLR


# simulation
def simulation_colored_noise(linear_code, top_config, net_config, simutimes_range, target_err_bits_num, batch_size):
# target_err_bits_num: the simulation stops if the number of bit errors reaches the target.
# simutimes_range: [min_simutimes, max_simutimes]

    ## load configurations from top_config
    SNRset = top_config.eval_SNRs
    bp_iter_num = top_config.BP_iter_nums_simu
    noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file_simu, rng_seed=0)
    denoising_net_num = top_config.cnn_net_number
    model_id = top_config.model_id

    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    K, N = np.shape(G_matrix)

    ## build BP decoding network
    if np.size(bp_iter_num) != denoising_net_num + 1:
        print('Error: the length of bp_iter_num is not correct!')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size)

    ## build denoising network
    conv_net = {}
    denoise_net_in = {}
    denoise_net_out = {}
    # build network for each CNN denoiser,
    for net_id in range(denoising_net_num):
        if top_config.same_model_all_nets and net_id > 0:
            conv_net[net_id] = conv_net[0]
            denoise_net_in[net_id] = denoise_net_in[0]
            denoise_net_out[net_id] = denoise_net_out[0]
        else:
            conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id)
            denoise_net_in[net_id], denoise_net_out[net_id] = conv_net[net_id].build_network()
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    # restore denoising network
    for net_id in range(denoising_net_num):
        if top_config.same_model_all_nets and net_id > 0:
            break
        conv_net[net_id].restore_network_with_model_id(sess, net_config.total_layers, model_id[0:(net_id+1)])

    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(bp_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (net_config.model_folder, N, K, bp_str))

    if top_config.corr_para != top_config.corr_para_simu:  # this means we are testing the model robustness to correlation level.
        ber_file = format('%s_SimuCorrPara%.2f' % (ber_file, top_config.corr_para_simu))
    if top_config.same_model_all_nets:
        ber_file = format('%s_SameModelAllNets' % ber_file)
    if top_config.update_llr_with_epdf:
        ber_file = format('%s_llrepdf' % ber_file)
    if denoising_net_num > 0:
        model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str)-1)]
        ber_file = format('%s_model%s' % (ber_file, model_id_str))
    if np.size(SNRset) == 1:
        ber_file = format('%s_%.1fdB' % (ber_file, SNRset[0]))

    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')

    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNRset:
        real_batch_size = batch_size
        # simulation part
        bit_errs_iter = np.zeros(denoising_net_num + 1, dtype=np.int32)
        actual_simutimes = 0
        rng = np.random.RandomState(0)
        noise_io.reset_noise_generator()
        for ik in range(0, max_batches):
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
            x_bits, _, s_mod, ch_noise, y_receive, LLR = lbc.encode_and_transmission(G_matrix, SNR, real_batch_size, noise_io, rng)
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0))
            print('Practical EbN0: %.2f' % practical_snr)

            for iter in range(0, denoising_net_num+1):
                # BP decoding
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter])

                if iter < denoising_net_num:
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_epdf(prob, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)
                    else:
                        res_noise_power = conv_net[iter].get_res_noise_power(model_id, SNRset).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_awgn(res_noise_power, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)
                output_x = linear_code.dec_src_bits(u_BP_decoded)
                bit_errs_iter[iter] += np.sum(output_x != x_bits)

            actual_simutimes += real_batch_size
            if bit_errs_iter[denoising_net_num] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))

        ber_iter = np.zeros(denoising_net_num+1, dtype=np.float64)
        fout_ber.write(str(SNR) + '\t')
        for iter in range(0, denoising_net_num+1):
            ber_iter[iter] = bit_errs_iter[iter] / float(K * actual_simutimes)
            fout_ber.write(str(ber_iter[iter]) + '\t')
        fout_ber.write('\n')

    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')


def generate_noise_samples(linear_code, top_config, net_config, train_config, bp_iter_num, net_id_data_for, generate_data_for, noise_io, model_id):
# net_id_data_for: the id of the CNN network this function generates data for. Start from zero.
# model_id is to designate the specific model folder
    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix

    SNRset_for_generate_training_data = train_config.SNR_set_gen_data
    if generate_data_for == 'Training':
        batch_size_each_SNR = int(train_config.training_minibatch_size // np.size(train_config.SNR_set_gen_data))
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
    elif generate_data_for == 'Test':
        batch_size_each_SNR = int(train_config.test_minibatch_size // np.size(train_config.SNR_set_gen_data))
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
    else:
        print('Invalid objective of data generation!')
        exit(0)


    # build BP decoding network
    if np.size(bp_iter_num) != net_id_data_for + 1:
        print('Error: the length of bp_iter_num is not correct!')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size_each_SNR)

    conv_net = {}
    denoise_net_in = {}
    denoise_net_out = {}
    for net_id in range(net_id_data_for):
        conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id)
        denoise_net_in[net_id], denoise_net_out[net_id] = conv_net[net_id].build_network()

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # restore cnn networks before the target CNN
    for net_id in range(net_id_data_for):
        conv_net[net_id].restore_network_with_model_id(sess, net_config.total_layers, model_id[0:(net_id+1)])

    start = datetime.datetime.now()

    if generate_data_for == 'Training':
        fout_est_noise = open(train_config.training_feature_file, 'wb')
        fout_real_noise = open(train_config.training_label_file, 'wb')
    elif generate_data_for == 'Test':
        fout_est_noise = open(train_config.test_feature_file, 'wb')
        fout_real_noise = open(train_config.test_label_file, 'wb')
    else:
        print('Invalid objective of data generation!')
        exit(0)

    # generating data
    for ik in range(0, total_batches):  # number of batches
        for SNR in SNRset_for_generate_training_data:
            x_bits, _, _, channel_noise, y_receive, LLR = lbc.encode_and_transmission(G_matrix, SNR, batch_size_each_SNR, noise_io)

            for iter in range(0, net_id_data_for + 1):
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter])

                if iter != net_id_data_for:
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_epdf(prob, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)
                    else:
                        res_noise_power = conv_net[iter].get_res_noise_power(model_id).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_awgn(res_noise_power, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], sess)

            # reconstruct noise
            noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
            noise_before_cnn = noise_before_cnn.astype(np.float32)
            noise_before_cnn.tofile(fout_est_noise)  # write features to file
            channel_noise.tofile(fout_real_noise)  # write labels to file

    fout_real_noise.close()
    fout_est_noise.close()

    sess.close()
    end = datetime.datetime.now()

    print("Time: %ds" % (end - start).seconds)
    print("end")


## calculate the resdual noise power or its empirical distribution
def analyze_residual_noise(linear_code, top_config, net_config, simutimes, batch_size):

    ## load some configurations from top_config
    net_id_tested = top_config.currently_trained_net_id
    model_id = top_config.model_id
    bp_iter_num = top_config.BP_iter_nums_gen_data[0:(net_id_tested + 1)]
    noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file)
    SNRset = top_config.eval_SNRs

    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    _, N = np.shape(G_matrix)

    max_batches, residual_times = np.array(divmod(simutimes, batch_size), np.int32)
    print('Real simutimes: %d' % simutimes)
    if residual_times != 0:
        max_batches += 1

    # build BP decoding network
    if np.size(bp_iter_num) != net_id_tested + 1:
        print('Error: the length of bp_iter_num is not correct!')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size)

    # build denoising network
    conv_net = {}
    denoise_net_in = {}
    denoise_net_out = {}

    # build network for each CNN denoiser,
    for net_id in range(net_id_tested+1):
        conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id)
        denoise_net_in[net_id], denoise_net_out[net_id] = conv_net[net_id].build_network()

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # restore denoising network
    for net_id in range(net_id_tested + 1):
        conv_net[net_id].restore_network_with_model_id(sess, net_config.total_layers, model_id[0:(net_id+1)])

    model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
    model_id_str = model_id_str[1:(len(model_id_str) - 1)]
    loss_file_name = format("%sresidual_noise_property_netid%d_model%s.txt" % (net_config.residual_noise_property_folder, net_id_tested, model_id_str))
    fout_loss = open(loss_file_name, 'wt')

    start = datetime.datetime.now()
    for SNR in SNRset:
        noise_io.reset_noise_generator()
        real_batch_size = batch_size
        # simulation part
        loss = 0.0
        prob = np.ones(0)
        for ik in range(0, max_batches):
            print("Batch id: %d" % ik)
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
            x_bits, _, s_mod, channel_noise, y_receive, LLR = lbc.encode_and_transmission(G_matrix, SNR, real_batch_size, noise_io)

            for iter in range(0, net_id_tested+1):
                # BP decoding
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter])
                noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
                noise_after_cnn = sess.run(denoise_net_out[iter], feed_dict={denoise_net_in[iter]: noise_before_cnn})
                s_mod_plus_res_noise = y_receive - noise_after_cnn
                if iter < net_id_tested:  # calculate the LLR for next BP decoding
                    if top_config.update_llr_with_epdf:
                        prob_tmp = conv_net[iter].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = calc_LLR_epdf(prob_tmp, s_mod_plus_res_noise)
                    else:
                        res_noise_power = conv_net[iter].get_res_noise_power(model_id).get(np.float32(SNR))
                        LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
            if top_config.update_llr_with_epdf:
                prob = stat_prob(s_mod_plus_res_noise - s_mod, prob)
            else:
                loss += np.sum(np.mean(np.square(s_mod_plus_res_noise-s_mod), 1))

        # each SNR
        if top_config.update_llr_with_epdf:
            fout_loss.write(str(SNR) + '\t')
            for i in range(np.size(prob)):
                fout_loss.write(str(prob[i]) + '\t')
            fout_loss.write('\n')
        else:
            loss /= np.double(simutimes)
            fout_loss.write(str(SNR) + '\t' + str(loss) + '\n')

    fout_loss.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()