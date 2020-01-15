# author:fighting
import argparse
import tensorflow as tf
import pickle
import math
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
FLAGS = None

# One-hot encoding for category labels
def getOne_hot(labels):
    f_labels = []
    for lab in labels:
        f_labels.append(lab-1)
    one_hot_index = np.arange(len(f_labels)) * 31 + f_labels
    one_hot = np.zeros((len(f_labels), 31))
    one_hot.flat[one_hot_index] = 1
    return one_hot



def load_data():
    """
    Load data from pickle
    :return: Arrays
    """
    with open(FLAGS.source_data, 'rb') as f:
        train_label = pickle.load(f)
        dev_label = pickle.load(f)
        test_label = pickle.load(f)
        train_vectors = pickle.load(f)
        dev_vectors = pickle.load(f)
        test_vectors = pickle.load(f)
        train_nums = pickle.load(f)
        dev_nums = pickle.load(f)
        test_nums = pickle.load(f)
        train_distance = pickle.load(f)
        dev_distance = pickle.load(f)
        test_distance = pickle.load(f)
        test_texts = pickle.load(f)
        return train_label, dev_label, test_label, train_vectors, dev_vectors, test_vectors, \
               train_nums, dev_nums, test_nums, train_distance, dev_distance, test_distance,test_texts

# initial weight
def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


# initial offset
def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)

# initial lstm
def lstm_cell(num_units, keep_prob=1):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)





# 将数组中字符串元素转化为Int类型
def strTransInt(arr):
    arr1 = [int(float(val)) for val in arr]
    return arr1



def main():
    # Load data
    train_label, dev_label, test_label, train_vectors, dev_vectors, test_vectors, train_nums, dev_nums, \
    test_nums, train_distance, dev_distance, test_distance, test_texts = load_data()
    print('test_label:', test_label)

    # train_distance = normalization(train_distance)  # 将距离标准化
    # dev_distance = normalization(dev_distance)
    # test_distance = normalization(test_distance)

    # 将标签one-hot编码之前先进行数组内元素类型的转化（str-int）
    train_label_oh = getOne_hot(train_label)
    print("train_vectors_size:", train_vectors.shape)
    dev_label_oh = getOne_hot(dev_label)
    # test_label = strTransInt(test_label)
    test_label_oh = getOne_hot(test_label)
    # Split data
    # train_x, train_y, dev_x, dev_y= get_data(data_x, data_y)
    print("test_vectors:", type(test_vectors), 'shape:', test_vectors.shape)   #(20, 25, 128)
    # Steps  计算一个epoch计算完有多少个batch_size   train_x.shape[0]:训练数据的总条数     train_batch_size:一个batch_size包含多少条数据
    train_steps = math.ceil(train_vectors.shape[0] / FLAGS.train_batch_size)

    print("train.shape[0]:", train_vectors.shape[0], "train_steps:", train_steps, "FLAGS.train_batch_size",
          FLAGS.train_batch_size)
    dev_steps = math.ceil(dev_vectors.shape[0] / FLAGS.dev_batch_size)
    print("dev.shape[0]:", dev_vectors.shape[0], "dev_steps:", dev_steps, "FLAGS.dev_batch_size",
          FLAGS.dev_batch_size)
    test_steps = math.ceil(test_vectors.shape[0] / FLAGS.test_batch_size)
    print("test.shape[0]:", test_vectors.shape[0], "test_steps:", test_steps, "FLAGS.test_batch_size",
          FLAGS.test_batch_size)
    global_step = tf.Variable(-1, trainable=True, name='global_step')
    # Train and dev dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_vectors, train_label_oh))
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)
    train_lab = tf.data.Dataset.from_tensor_slices(train_label)
    train_lab = train_lab.batch(FLAGS.train_batch_size)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_vectors, dev_label_oh))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
    dev_lab = tf.data.Dataset.from_tensor_slices(dev_label)
    dev_lab = dev_lab.batch(FLAGS.dev_batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_vectors, test_label_oh))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)
    test_lab = tf.data.Dataset.from_tensor_slices(test_label)
    test_lab = test_lab.batch(FLAGS.test_batch_size)

    train_distance_dataset = tf.data.Dataset.from_tensor_slices(train_distance)
    train_distance_dataset = train_distance_dataset.batch(FLAGS.train_batch_size)

    test_distance_dataset = tf.data.Dataset.from_tensor_slices(test_distance)
    test_distance_dataset = test_distance_dataset.batch(FLAGS.test_batch_size)

    dev_distance_dataset = tf.data.Dataset.from_tensor_slices(dev_distance)
    dev_distance_dataset = dev_distance_dataset.batch(FLAGS.dev_batch_size)

    # A reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    iterator_distance = tf.data.Iterator.from_structure(train_distance_dataset.output_types, train_distance_dataset.output_shapes)
    iterator_lab = tf.data.Iterator.from_structure(train_lab.output_types, train_lab.output_shapes)
    with tf.name_scope('input_data'):
        with tf.name_scope('train_dataset_initial'):
            train_initializer = iterator.make_initializer(train_dataset)
        with tf.name_scope('dev_dataset_initial'):
            dev_initializer = iterator.make_initializer(dev_dataset)
        with tf.name_scope('test_dataset_initial'):
            test_initializer = iterator.make_initializer(test_dataset)
        with tf.name_scope('train_weight_dataset_initial'):
            tw_initializer = iterator_distance.make_initializer(train_distance_dataset)
        with tf.name_scope('test_weight_dataset_initial'):
            tew_initializer = iterator_distance.make_initializer(test_distance_dataset)
        with tf.name_scope('dev_weight_dataset_initial'):
            dw_initializer = iterator_distance.make_initializer(dev_distance_dataset)

    train_lab_initializer = iterator_lab.make_initializer(train_lab)
    dev_lab_initializer = iterator_lab.make_initializer(dev_lab)
    test_lab_initializer = iterator_lab.make_initializer(test_lab)


    # Input Layer
    with tf.variable_scope('inputs'):
        # 此处的y_label是将原始标签经one-hot编码后的数据
        x, y_label = iterator.get_next()
        tw = iterator_distance.get_next()
        # 此处的y_lab是原始的标签，数据形式为：[[1, 2], [2,4], ......]
        y_lab = iterator_lab.get_next()

    # x = tf.cast(x, dtype=tf.float32)
    inputs = x

    keep_prob = tf.placeholder(tf.float64, [])
    is_train = tf.placeholder(tf.bool)
    # RNN Layer
    cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)
    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float64)
    output = tf.stack(output, axis=1)
    print('Output', output)
    output = tf.reshape(output, [-1, FLAGS.num_units * 2])
    print('Output Reshape', output)


    with tf.variable_scope('combination'):
        w = weight([FLAGS.num_units*2 , FLAGS.category_num])
        b = bias([FLAGS.category_num])
        w = tf.cast(w, tf.float64)
        b = tf.cast(b, tf.float64)
        y = tf.matmul(output, w) + b
        y = tf.reshape(y, [-1,25, 31])
        # multiply需要保持两个元素的类型一致
        tw = tf.cast(tw, tf.float64)
        tw = tf.reshape(tw, [-1,25, 31])
        # tw = tf.nn.tanh(tw)                 # 将距离权重归一化
        y = tf.cast(y, tf.float64)
        y = tf.add(y, tw)     # tw为每个词在7类里面的距离向量
        print("Y:", y)
        y_hidden = tf.reduce_sum(y, 1)
        y_hidden = tf.layers.batch_normalization(y_hidden, training=is_train)
        y_ = tf.nn.softmax(y_hidden)
        y_predict = tf.cast(tf.argmax(y_, axis=1), tf.int32)
        print('Output Y', y_predict)
    tf.summary.histogram('y_predict', y_predict)
    E2 = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_hidden)
    cross_entropy = tf.reduce_mean(E2)
    tf.summary.scalar('loss', cross_entropy)

    # Train  反向传递误差，调节网络参数
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)

    # Saver
    saver = tf.train.Saver()

    # Iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Global step
    gstep = 0

    # Summaries
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
                                   sess.graph)

    if FLAGS.train:
        # 训练的时候不会生成可视化界面，因为在训练的起始会首先执行删除操作
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

        for epoch in range(FLAGS.epoch_num):
            print('epoch:', epoch, "epoch_num:", FLAGS.epoch_num)
            tf.train.global_step(sess, global_step_tensor=global_step)
            # Train
            sess.run(train_initializer)
            sess.run(tw_initializer)
            sess.run(train_lab_initializer)
            for step in range(int(train_steps)):
                # tw = sess.run([tw],  feed_dict={keep_prob: FLAGS.keep_prob, is_train:True})
                # print("tw:", tw)
                smrs, loss_,  gstep, _,  y_labs = sess.run([summaries, cross_entropy, global_step, train,  y_lab],
                                                     feed_dict={keep_prob: 1, is_train:True})
                # acc = get_acc(y_predict_rr, y_labs, step)


                if step % FLAGS.steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss_)
                    # if loss_ < 0.015:
                    #     saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
                    #     return



                # Summaries for tensorboard
                if gstep % FLAGS.steps_per_summary == 0:
                    writer.add_summary(smrs, gstep)
                    print('Write summaries to', FLAGS.summaries_dir)
            #
            if epoch % FLAGS.epochs_per_dev == 0:
                # Dev
                sess.run(dev_initializer)
                sess.run(dw_initializer)
                sess.run(dev_lab_initializer)
                for step in range(int(dev_steps)):
                    if step % FLAGS.steps_per_print == 0:
                        print('Dev loss', sess.run(cross_entropy, feed_dict={keep_prob: 1, is_train:True}), 'Step', step)
            # Save model
            if epoch % FLAGS.epochs_per_save == 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
        # plot_learning_curves(accuracy)


    else:
        # Load model
        ckpt = tf.train.get_checkpoint_state('ckptIntetion1')
        # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)
        sess.run(test_initializer)
        sess.run(tew_initializer)
        sess.run(test_lab_initializer)
        for step in range(int(test_steps)):
            # y_predict_rr是筛选出概率大于0.5的预测值
            # tw = sess.run([tw],  feed_dict={keep_prob: FLAGS.keep_prob, is_train:True})
            # print("tw:", tw)
            y_predict_results, y_labs = sess.run([y_predict, y_lab], feed_dict={keep_prob: 1, is_train:True})
            print('y_predict', y_predict_results)
            print("test_label:", test_label)
            n = 0   # Count the number of correct identification
            i = 0  # record the index of y_predict_results
            TF1, TF2, TF3, TF4, TF5, TF6, TF7, TF8, TF9, TF10, TF11, TF12, TF13, TF14, TF15, TF16, TF17, TF18, TF19, TF20, TF21, TF22\
                , TF23, TF24, TF25, TF26, TF27, TF28, TF29, TF30, TF31= 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            FP1, FP2, FP3, FP4, FP5, FP6, FP7, FP8, FP9, FP10, FP11, FP12, FP13, \
            FP14, FP15, FP16, FP17, FP18, FP19, FP20, FP21, FP22, FP23, FP24, FP25, FP26, FP27, FP28, FP29, FP30, FP31= 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, TP19, TP20, TP21, \
            TP22, TP23, TP24, TP25, TP26, TP27, TP28, TP29, TP30, TP31= 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

            for test_lab in test_label:
                result = y_predict_results[i] + 1
                if result == 1:
                    FP1 += 1
                if result == 2:
                    FP2 += 1
                if result == 3:
                    FP3 += 1
                if result == 4:
                    FP4 += 1
                if result == 5:
                    FP5 += 1
                if result == 6:
                    FP6 += 1
                if result == 7:
                    FP7 += 1
                if result == 8:
                    FP8 += 1
                if result == 9:
                    FP9 += 1
                if result == 10:
                    FP10 += 1
                if result == 11:
                    FP11 += 1
                if result == 12:
                    FP12 += 1
                if result == 13:
                    FP13 += 1
                if result == 14:
                    FP14 += 1
                if result == 15:
                    FP15+= 1
                if result == 16:
                    FP16 += 1
                if result == 17:
                    FP17 += 1
                if result == 18:
                    FP18 += 1
                if result == 19:
                    FP19 += 1
                if result == 20:
                    FP20 += 1
                if result == 21:
                    FP21 += 1
                if result == 22:
                    FP22 += 1
                if result == 23:
                    FP23 += 1
                if result == 24:
                    FP24 += 1
                if result == 25:
                    FP25 += 1
                if result == 26:
                    FP26 += 1
                if result == 27:
                    FP27 += 1
                if result == 28:
                    FP28 += 1
                if result == 29:
                    FP29 += 1
                if result == 30:
                    FP30 += 1
                if result == 31:
                    FP31 += 1

                if test_lab == 1:
                    TF1 += 1
                if test_lab == 2:
                    TF2 += 1
                if test_lab == 3:
                    TF3 += 1
                if test_lab == 4:
                    TF4 += 1
                if test_lab == 5:
                    TF5 += 1
                if test_lab == 6:
                    TF6 += 1
                if test_lab == 7:
                    TF7 += 1
                if test_lab == 8:
                    TF8 += 1
                if test_lab == 9:
                    TF9 += 1
                if test_lab == 10:
                    TF10 += 1
                if test_lab == 11:
                    TF11 += 1
                if test_lab == 12:
                    TF12 += 1
                if test_lab == 13:
                    TF13 += 1
                if test_lab == 14:
                    TF14 += 1
                if test_lab == 15:
                    TF15 += 1
                if test_lab == 16:
                    TF16 += 1
                if test_lab == 17:
                    TF17 += 1
                if test_lab == 18:
                    TF18 += 1
                if test_lab == 19:
                    TF19 += 1
                if test_lab == 20:
                    TF20 += 1
                if test_lab == 21:
                    TF21 += 1
                if test_lab == 22:
                    TF22 += 1
                if test_lab == 23:
                    TF23 += 1
                if test_lab == 24:
                    TF24 += 1
                if test_lab == 25:
                    TF25 += 1
                if test_lab == 26:
                    TF26 += 1
                if test_lab == 27:
                    TF27 += 1
                if test_lab == 28:
                    TF28 += 1
                if test_lab == 29:
                    TF29 += 1
                if test_lab == 30:
                    TF30 += 1
                if test_lab == 31:
                    TF31 += 1
                if result == test_lab:
                    if result == 1:
                        TP1 += 1
                    if result == 2:
                        TP2 += 1
                    if result == 3:
                        TP3 += 1
                    if result == 4:
                        TP4 += 1
                    if result == 5:
                        TP5 += 1
                    if result == 6:
                        TP6 += 1
                    if result == 7:
                        TP7 += 1
                    if result == 8:
                        TP8 += 1
                    if result == 9:
                        TP9 += 1
                    if result == 10:
                        TP10 += 1
                    if result == 11:
                        TP11 += 1
                    if result == 12:
                        TP12 += 1
                    if result == 13:
                        TP13 += 1
                    if result == 14:
                        TP14 += 1
                    if result == 15:
                        TP15 += 1
                    if result == 16:
                        TP16 += 1
                    if result == 17:
                        TP17 += 1
                    if result == 18:
                        TP18 += 1
                    if result == 19:
                        TP19 += 1
                    if result == 20:
                        TP20 += 1
                    if result == 21:
                        TP21 += 1
                    if result == 22:
                        TP22 += 1
                    if result == 23:
                        TP23 += 1
                    if result == 24:
                        TP24 += 1
                    if result == 25:
                        TP25 += 1
                    if result == 26:
                        TP26 += 1
                    if result == 27:
                        TP27 += 1
                    if result == 28:
                        TP28 += 1
                    if result == 29:
                        TP29 += 1
                    if result == 30:
                        TP30 += 1
                    if result == 31:
                        TP31 += 1
                    n += 1
                print(test_texts[i])
                print("predict:", result, "label:", test_lab, n)
                i += 1
            all_test_sens = len(test_label)
            # average precision
            p = (TP1/TF1 + TP2/TF2 + TP3/TF3 + TP4/TF4 + TP5/TF5 + TP6/TF6 + TP7/TF7 + TP8/TF8 + TP9/TF9 + TP10/TF10 + \
                TP11/TF11 + TP12/TF12 + TP13/TF13 + TP14/TF14 + TP15/TF15 + TP16/TF16 + TP17/TF17 + TP18/TF18 + \
                TP19/TF19 + TP20/TF20 + TP21/TF21 + TP22/TF22 + TP23/TF23 + TP24/TF24 + TP25/TF25 + TP26/TF26 + TP27/TF27 + TP28/TF28+ TP29/TF29 + TP30/TF30 +TP31/TF31)/31
            # average recall
            r = (TP1/FP1 + TP2/FP2 + TP3/FP3 + TP4/FP4 + TP5/FP5 + TP6/FP6 + TP7/FP7 + TP8/FP8 + TP9/FP9 + TP10/FP10 + TP11/FP11
                 + TP12/FP12 + TP13/FP13 + TP14/FP14 + TP15/FP15 + TP16/FP16 + TP17/FP17 + TP18/FP18 + TP19/FP19 + TP20/FP20 +
                 TP21/FP21 + TP22/FP22 + TP23/FP23 + TP24/FP24 + TP25/FP25 + TP26/FP26 + TP27/FP27 + TP28/FP28 + TP29/FP29 + TP30/FP30+ TP31/FP31)/31
            # f1-score
            f1 = (2*p*r)/(p+r)

            print('accuracy:', n / all_test_sens)
            print('Precision:', p)
            print('recall:', r)
            print('f1:', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI LSTM')
    parser.add_argument('--train_batch_size', help='train batch size', default=50)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=20)
    parser.add_argument('--test_batch_size', help='test batch size', default=667)  # 模型每次测试的样本
    parser.add_argument('--source_data', help='source_size', default='./data/intention/data.pkl')
    # parser.add_argument('--source_data', help='source size', default='/Users/fighting/PycharmProjects/BiLSTM_MClassification/data/intention/data.pkl')
    parser.add_argument('--num_layer', help='num of layer', default=2, type=int)
    parser.add_argument('--num_units', help='num of units', default=128, type=int)
    # parser.add_argument('--time_step', help='time steps', default=32, type=int)
    parser.add_argument('--time_step', help='time steps', default=25, type=int)   # 输入一句话的长度
    parser.add_argument('--embedding_size', help='time steps', default=128, type=int)
    # parser.add_argument('--category_num', help='category num', default=5, type=int)
    parser.add_argument('--category_num', help='category num', default=31, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)
    # parser.add_argument('--epoch_num', help='num of epoch', default=1000, type=int)
    parser.add_argument('--epoch_num', help='num of epoch', default=50, type=int)
    parser.add_argument('--epochs_per_test', help='epochs per test', default=20, type=int)
    # parser.add_argument('--epochs_per_test', help='epochs per test', default=200, type=int)
    parser.add_argument('--epochs_per_dev', help='epochs per dev', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=100, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=1, type=float)
    # parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.6, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckptIntetion1/model.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries/', type=str)
    parser.add_argument('--train', help='train', default=False, type=bool)
    # parser.add_argument('--train', help='train', default=True, type=bool)
    FLAGS, args = parser.parse_known_args()
    main()
