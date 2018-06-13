import numpy as np
import tensorflow as tf
import logging

from CNN_input import read_dataset

logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

DATASET_DIR = r'C:\Users\PycharmProjects\Cifar'
N_FEATURES = 3072
N_CLASSES = 7
N_FC1 = 512
N_FC2 = 256
BATCH_SIZE = 128
TEST_BATCH_SIZE = 5000
TRAINING_EPOCHS = 10000
DISPLAY_STEP = 50
SAVE_STEP = 1000
BASEDIR = './New_LOG/'
BETA = 0.01

cifar10 = read_dataset(DATASET_DIR, onehot_encoding=True)
logging.info('TRAIN: {}\nEVAL: {}'.format(cifar10.train.images.shape, cifar10.eval.images.shape))


def conv_layer(inpt, k, s, channels_in, channels_out, name='CONV'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(
            [k, k, channels_in, channels_out], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='b')
        conv = tf.nn.conv2d(inpt, W, strides=[1, s, s, 1], padding='SAME')
        act = tf.nn.relu(conv)
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return act


def pool_layer(inpt, k, s, pool_type='mean'):
    if pool_layer is 'mean':
        return tf.nn.avg_pool(inpt,
                              ksize=[1, k, k, 1],
                              strides=[1, s, s, 1],
                              padding='SAME',
                              name='POOL')
    else:
        return tf.nn.max_pool(inpt,
                              ksize=[1, k, k, 1],
                              strides=[1, s, s, 1],
                              padding='SAME',
                              name='POOL')


def fc_layer(inpt, neurons_in, neurons_out, last=False, name='FC'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(
            [neurons_in, neurons_out]), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[neurons_out]), name='b')
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        if last:
            act = tf.add(tf.matmul(inpt, W), b)
        else:
            act = tf.nn.relu(tf.add(tf.matmul(inpt, W), b))
        tf.summary.histogram('activations', act)
        return act


def cifar10_model(learning_rate, batch_size):
    tf.reset_default_graph()
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, N_FEATURES], name='x')
        x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), perm=[0, 2, 3, 1])
        tf.summary.image('input', x_image, max_outputs=3)
        y = tf.placeholder(tf.float32, [None, N_CLASSES], name='labels')

        phase = tf.placeholder(tf.bool, name='PHASE')

        conv1 = conv_layer(x_image, 5, 1, channels_in=3, channels_out=64)
        with tf.name_scope('BN'):
            norm1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=phase)
        pool1 = pool_layer(norm1, 3, 2, pool_type='mean')
        conv2 = conv_layer(pool1, 5, 1, channels_in=64, channels_out=64)
        with tf.name_scope('BN'):
            norm2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=phase)
        pool2 = pool_layer(norm2, 3, 2, pool_type='mean')

        flattend = tf.reshape(pool2, shape=[-1, 8 * 8 * 64])
        fc1 = fc_layer(flattend, neurons_in=8 * 8 * 64, neurons_out=N_FC1)
        with tf.name_scope('BN'):
            norm3 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=phase)
        fc1_dropout = tf.nn.dropout(norm3,N_dp)

		fc2 = fc_layer(fc1_dropout, neurons_in=N_FC1, neurons_out=N_FC2, last=True)
        with tf.name_scope('BN'):
            norm4 = tf.contrib.layers.batch_norm(fc2, center=True, scale=True, is_training=phase)
        fc2_dropout = tf.nn.dropout(norm4,N_dp)

        logits = fc_layer(fc2_dropout, neurons_in=N_FC2, neurons_out=N_CLASSES, last=True)

        trainable_vars = tf.trainable_variables()

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)) + \
                BETA * tf.add_n([tf.nn.l2_loss(v)
                                for v in trainable_vars if not 'b' in v.name])

            tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        merged_summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess.run(init)

        LOGDIR = BASEDIR + 'lr={:.0E},bs={}'.format(learning_rate, batch_size)
        summary_writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)
        eval_writer = tf.summary.FileWriter(LOGDIR + '/eval')

        for i in range(TRAINING_EPOCHS):
            batch_x, batch_y = cifar10.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, phase: 1})
            if i % DISPLAY_STEP == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                s, lss, acc , _ = sess.run([merged_summary, loss, accuracy, train_step], 
                                    feed_dict={x: batch_x, y: batch_y, phase: 1},
                                    options=run_options,
                                    run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, 'step{}'.format(i))
                summary_writer.add_summary(s, i)
                for batch in range(cifar10.eval.num_exzamples // TEST_BATCH_SIZE):
                    test_acc = []
                    batch_x, batch_y = cifar10.eval.next_batch(TEST_BATCH_SIZE)
                    test_acc.append(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, phase: 0}))
                eval_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='eval_accuracy', simple_value=np.mean(test_acc))]), i)
                logging.info('Iter={}, loss={}, trainging_accuracy={}, test_accuracy={}'.format(i+1, lss, acc, np.mean(test_acc)))
        LOGDIR = saver.save(sess, LOGDIR + '/model.ckpt')
        logging.info('Model saved in file: {}'.format(LOGDIR))


def main():
    for lr in [1e-2]:#, 1e-3, 1e-4]:
        for bs in [64]:#, 128]:
            logging.info('learing rate = {:.0E}, batch size = {}'.format(lr, bs))
            cifar10_model(lr, bs)

if __name__ == '__main__':
    main()
