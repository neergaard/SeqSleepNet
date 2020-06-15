import os
import resource
# os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import shutil
import sys
from datetime import datetime
import h5py

from seqsleepnet import SeqSleepNet
from seqsleepnet_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_wrapper import DataGeneratorWrapper

from scipy.io import loadmat, savemat
from tensorflow.python import pywrap_tensorflow

import time

import wandb
wandb.init(project='seqsleepnet')


resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "./data_processing/tf_data/seqsleepnet_eeg/train_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "./data_processing/tf_data/seqsleepnet_eeg/eval_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "./data_processing/tf_data/seqsleepnet_eeg/test_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "./data_processing/tf_data/seqsleepnet_eog/train_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "./data_processing/tf_data/seqsleepnet_eog/eval_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "./data_processing/tf_data/seqsleepnet_eog/test_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "./data_processing/tf_data/seqsleepnet_emg/train_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "./data_processing/tf_data/seqsleepnet_emg/eval_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "./data_processing/tf_data/seqsleepnet_emg/test_list_n1.txt", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./out/seqsleepnet_sleep_nfilter32_seq30_dropout0.75_nhidden64_att64_3chan/holdout/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
# tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
# tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
# tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 10)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 32)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size1", 64, "Sequence length (default: 64)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

# flag for early stopping
tf.app.flags.DEFINE_boolean("early_stopping", False, "whether to apply early stopping (default: False)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):  # python3
    print("{}={}".format(attr.upper(), value.value))
print("")

# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path, FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)):
    os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)):
    os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.dropout_keep_prob_rnn = FLAGS.dropout_keep_prob_rnn
config.epoch_seq_len = FLAGS.seq_len
config.epoch_step = FLAGS.seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.nhidden2 = FLAGS.nhidden2
config.attention_size1 = FLAGS.attention_size1

eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

# 1 channel case
if (not eog_active and not emg_active):
    print("eeg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             num_fold=config.num_fold_training_data,
                                             data_shape=[config.frame_seq_len, config.ndim],
                                             seq_len=config.epoch_seq_len,
                                             shuffle=True)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                             num_fold=config.num_fold_testing_data,
                                             data_shape=[config.frame_seq_len, config.ndim],
                                             seq_len=config.epoch_seq_len,
                                             shuffle=False)
    # test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    #                                         num_fold=config.num_fold_testing_data,
    #                                         data_shape=[config.frame_seq_len, config.ndim],
    #                                         seq_len = config.epoch_seq_len,
    #                                         shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    valid_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    #test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    nchannel = 1

elif(eog_active and not emg_active):
    print("eeg and eog active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             num_fold=config.num_fold_training_data,
                                             data_shape=[config.frame_seq_len, config.ndim],
                                             seq_len=config.epoch_seq_len,
                                             shuffle=True)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
                                             num_fold=config.num_fold_testing_data,
                                             data_shape=[config.frame_seq_len, config.ndim],
                                             seq_len=config.epoch_seq_len,
                                             shuffle=False)
    # test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    #                                        eog_filelist=os.path.abspath(FLAGS.eog_test_data),
    #                                         num_fold=config.num_fold_testing_data,
    #                                         data_shape=[config.frame_seq_len, config.ndim],
    #                                         seq_len = config.epoch_seq_len,
    #                                         shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    valid_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    valid_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    #test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    #test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    nchannel = 2
elif(eog_active and emg_active):
    print("eeg, eog, and emg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             emg_filelist=os.path.abspath(FLAGS.emg_train_data),
                                             num_fold=config.num_fold_training_data,
                                             data_shape=[config.frame_seq_len, config.ndim],
                                             seq_len=config.epoch_seq_len,
                                             shuffle=True)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
                                             emg_filelist=os.path.abspath(FLAGS.emg_eval_data),
                                             num_fold=config.num_fold_testing_data,
                                             data_shape=[config.frame_seq_len, config.ndim],
                                             seq_len=config.epoch_seq_len,
                                             shuffle=False)
    # test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    #                                              eog_filelist=os.path.abspath(FLAGS.eog_test_data),
    #                                              emg_filelist=os.path.abspath(FLAGS.emg_test_data),
    #                                         num_fold=config.num_fold_testing_data,
    #                                         data_shape=[config.frame_seq_len, config.ndim],
    #                                         seq_len = config.epoch_seq_len,
    #                                         shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    train_gen_wrapper.compute_emg_normalization_params()
    valid_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    valid_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    valid_gen_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    #test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    #test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    #test_gen_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    nchannel = 3

config.nchannel = nchannel
wandb.config.update(vars(config))

# variable to keep track of best accuracy on validation set for model selection
best_acc = 0.0

# Training
# ==================================================
early_stop_count = 0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = SeqSleepNet(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # initialize all variables
        sess.run(tf.global_variables_initializer())
        print("Model initialized")
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len, dtype=int) * config.frame_seq_len
            epoch_seq_len = np.ones(len(x_batch), dtype=int) * config.epoch_seq_len
            feed_dict = {
                net.input_x: x_batch,
                net.input_y: y_batch,
                net.dropout_keep_prob_rnn: config.dropout_keep_prob_rnn,
                net.epoch_seq_len: epoch_seq_len,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 1
            }
            _, step, output_loss, total_loss, accuracy = sess.run(
                [train_op, global_step, net.output_loss, net.loss, net.accuracy],
                feed_dict)
            return step, output_loss, total_loss, accuracy

        def dev_step(x_batch, y_batch):
            """
            A single evaluation step
            """
            frame_seq_len = np.ones(len(x_batch)*config.epoch_seq_len, dtype=int) * config.frame_seq_len
            epoch_seq_len = np.ones(len(x_batch), dtype=int) * config.epoch_seq_len
            feed_dict = {
                net.input_x: x_batch,
                net.input_y: y_batch,
                net.dropout_keep_prob_rnn: 1.0,
                net.epoch_seq_len: epoch_seq_len,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 0
            }
            output_loss, total_loss, yhat = sess.run(
                [net.output_loss, net.loss, net.predictions], feed_dict)
            return output_loss, total_loss, yhat

        def evaluate(gen_wrapper, log_filename):
            output_loss = 0
            total_loss = 0

            N = int(np.sum(gen_wrapper.file_sizes) - (config.epoch_seq_len - 1)*len(gen_wrapper.file_sizes))
            yhat = np.zeros([config.epoch_seq_len, N])
            y = np.zeros([config.epoch_seq_len, N])

            count = 0
            gen_wrapper.new_subject_partition()
            for data_fold in range(config.num_fold_training_data):
                # load data of the current fold
                gen_wrapper.next_fold()
                yhat_, output_loss_, total_loss_ = _evaluate(gen_wrapper.gen)

                output_loss += output_loss_
                total_loss += total_loss_

                yhat[:, count: count + len(gen_wrapper.gen.data_index)] = yhat_

                # groundtruth
                for n in range(config.epoch_seq_len):
                    y[n, count: count + len(gen_wrapper.gen.data_index)] =\
                        gen_wrapper.gen.label[gen_wrapper.gen.data_index - (config.epoch_seq_len - 1) + n]
                count += len(gen_wrapper.gen.data_index)

            acc = 0
            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} ".format(output_loss, total_loss))
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(yhat[n, :], y[n, :])  # due to zero-indexing
                    if n == config.epoch_seq_len - 1:
                        text_file.write("{:g} \n".format(acc_n))
                    else:
                        text_file.write("{:g} ".format(acc_n))
                    acc += acc_n
            acc /= config.epoch_seq_len

            return acc, yhat, output_loss, total_loss

        def _evaluate(gen):
            # Validate the model on the entire data stored in gen variable
            output_loss = 0
            total_loss = 0
            yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            # test with minibatch of factor times training minibatch to speed up, , you can reduce it if having problem with memory
            factor = 20
            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*factor*config.batch_size: test_step*factor*config.batch_size] = yhat_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*factor*config.batch_size: len(gen.data_index)] = yhat_[n]
                output_loss += output_loss_
                total_loss += total_loss_
            yhat = yhat + 1  # python index from 0
            return yhat, output_loss, total_loss

        def evaluate_reduce(gen_wrapper=None, n_folds=10, log_filename=None):
            output_loss = 0
            total_loss = 0

            N = int(np.sum(gen_wrapper.file_sizes) - (config.epoch_seq_len - 1)*len(gen_wrapper.file_sizes))
            yhat = np.zeros([config.epoch_seq_len, N])
            y = np.zeros([config.epoch_seq_len, N])

            count = 0
            gen_wrapper.new_subject_partition()
            for data_fold in range(n_folds):
                # load data of the current fold
                gen_wrapper.next_fold()
                yhat_, output_loss_, total_loss_ = _evaluate_reduce(gen_wrapper.gen)

                output_loss += output_loss_
                total_loss += total_loss_

                yhat[:, count: count + len(gen_wrapper.gen.reduce_data_index)] = yhat_

                # groundtruth
                for n in range(config.epoch_seq_len):
                    y[n, count: count + len(gen_wrapper.gen.reduce_data_index)] =\
                        gen_wrapper.gen.label[gen_wrapper.gen.reduce_data_index - (config.epoch_seq_len - 1) + n]
                count += len(gen_wrapper.gen.reduce_data_index)

            yhat = yhat[:, :count]
            yhat = np.reshape(yhat, count*config.epoch_seq_len)
            y = y[:, :count]
            y = np.reshape(y, count*config.epoch_seq_len)

            acc = 0
            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} ".format(output_loss, total_loss))
                acc = accuracy_score(yhat, y)  # due to zero-indexing
                text_file.write("{:g} \n".format(acc))

            return acc, yhat, output_loss, total_loss

        def _evaluate_reduce(gen):
            # Validate the model on the entire data stored in gen variable
            output_loss = 0
            total_loss = 0
            yhat = np.zeros([config.epoch_seq_len, len(gen.reduce_data_index)])
            # test with minibatch of factor times training minibatch to speed up, , you can reduce it if having problem with memory
            factor = 20
            num_batch_per_epoch = np.floor(len(gen.reduce_data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch_reduce(factor*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*factor*config.batch_size: test_step*factor*config.batch_size] = yhat_[n]
                test_step += 1
            if(gen.reduce_pointer < len(gen.reduce_data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch_reduce(factor*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*factor*config.batch_size: len(gen.reduce_data_index)] = yhat_[n]
                output_loss += output_loss_
                total_loss += total_loss_
            yhat = yhat + 1

            return yhat, output_loss, total_loss

        # test off the pretrained model, using subsampling version to speed up testing
        print("{} Start off validation".format(datetime.now()))
        eval_acc, eval_yhat, eval_output_loss, eval_total_loss = \
            evaluate_reduce(gen_wrapper=valid_gen_wrapper, n_folds=config.num_fold_testing_data, log_filename="eval_result_log.txt")
        wandb.log({'epoch': 0, 'step': 0, 'eval_output_loss': eval_output_loss, 'eval_total_loss': eval_total_loss, 'eval_acc': eval_acc})
        # test_acc, test_yhat, test_output_loss, test_total_loss = \
        #    evaluate_reduce(gen_wrapper=test_gen_wrapper, log_filename="test_result_log.txt")
        # test_gen_wrapper.gen.reset_reduce_pointer()
        valid_gen_wrapper.gen.reset_reduce_pointer()

        start_time = time.time()
        # Loop over number of epochs

        print('')
        print("{} Starting training procedure".format(datetime.now()))
        total_step = -1
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            train_gen_wrapper.new_subject_partition()

            for data_fold in range(config.num_fold_training_data):
                train_gen_wrapper.next_fold()
                '''
                IMPORTANT HERE: the number of epoch is reduced by a factor of config.seq_len to encourage scanning through the subjects quicker
                then the number of training epoch need to increased by a factor of config.seq_len accordingly (in the config file)
                '''
                train_batches_per_epoch = np.floor(len(train_gen_wrapper.gen.data_index) / config.batch_size / config.epoch_seq_len).astype(np.uint32)
                print(train_batches_per_epoch)
                step = 1
                while step < train_batches_per_epoch:
                    step += 1
                    # Get a batch
                    x_batch, y_batch, label_batch = train_gen_wrapper.gen.next_batch(config.batch_size)
                    train_step_, train_output_loss_, train_total_loss_, train_acc_ = train_step(x_batch, y_batch)
                    time_str = datetime.now().isoformat()

                    # average acc over sequences
                    acc_ = 0
                    for n in range(config.epoch_seq_len):
                        acc_ += train_acc_[n]
                    acc_ /= config.epoch_seq_len

                    wandb.log({'step': total_step, 'train_data_fold': data_fold, 'train_step': total_step,
                               'train_output_loss': train_output_loss_, 'train_total_loss': train_total_loss_, 'train_acc': acc_})
                    print("{}: step {}, output_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, acc_))
                    with open(os.path.join(out_dir, "train_log.txt"), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g}\n".format(train_step_, train_output_loss_, train_total_loss_, acc_))
                    step += 1

                    current_step = tf.train.global_step(sess, global_step)
                    if step == train_batches_per_epoch - 1:
                        # if current_step % config.evaluate_every == 0:
                        # Validate the model on the validation and test sets
                        # using subsampling version to speed up testing
                        print("{} Start validation".format(datetime.now()))
                        eval_acc, eval_yhat, eval_output_loss, eval_total_loss = \
                            evaluate_reduce(gen_wrapper=valid_gen_wrapper, log_filename="eval_result_log.txt")
                        # test_acc, test_yhat, test_output_loss, test_total_loss = \
                        #    evaluate_reduce(gen_wrapper=test_gen_wrapper, log_filename="test_result_log.txt")
                        wandb.log({'step': total_step, 'eval_output_loss': eval_output_loss, 'eval_total_loss': eval_total_loss, 'eval_acc': eval_acc})

                        early_stop_count += 1
                        if(eval_acc >= best_acc):
                            early_stop_count = 0  # reset
                            best_acc = eval_acc
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) + '.ckpt')
                            save_path = saver.save(sess, checkpoint_name)

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best.txt"), "a") as text_file:
                                #text_file.write("Validation accuracy {:g} \n".format(eval_acc))
                                #text_file.write("Test accuracy {:g} \n".format(test_acc))
                                #text_file.write("{:g} {:g}\n".format(eval_acc, test_acc))
                                text_file.write("{:g}\n".format(eval_acc))

                        valid_gen_wrapper.gen.reset_reduce_pointer()
                        # test_gen_wrapper.gen.reset_reduce_pointer()

                        if(FLAGS.early_stopping == True):
                            #print('EARLY STOPPING enabled!')
                            # early stop after 10 training steps without improvement.
                            if(early_stop_count >= config.early_stop_count):
                                end_time = time.time()
                                with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                                    text_file.write("{:g}\n".format((end_time - start_time)))
                                quit()

        end_time = time.time()
        with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
