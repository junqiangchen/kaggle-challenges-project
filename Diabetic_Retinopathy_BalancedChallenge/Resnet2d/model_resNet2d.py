'''

'''
from Resnet2d.layer import (conv2d, normalizationlayer, max_pool2d, resnet_Add, weight_xavier_init, bias_variable,
                            dense_to_one_hot)
import tensorflow as tf
import numpy as np
import cv2
import os


def conv_relu_drop(x, kernal, drop, phase, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv2d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, norm_type='group', scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def full_connected_relu_drop(x, kernal, drop, activefunction='relu', scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        FC = tf.matmul(x, W) + B
        if activefunction == 'relu':
            FC = tf.nn.relu(FC)
            FC = tf.nn.dropout(FC, drop)
        elif activefunction == 'softmax':
            FC = tf.nn.softmax(FC)
        return FC


def _create_conv_net(X, image_width, image_height, image_channel, drop, phase, n_class=1):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_relu_drop(x=inputX, kernal=(3, 3, image_channel, 16), drop=drop, phase=phase, scope='layer0')
    layer1 = conv_relu_drop(x=layer0, kernal=(3, 3, 16, 16), drop=drop, phase=phase, scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = max_pool2d(x=layer1)
    # layer2->convolution
    layer2 = conv_relu_drop(x=down1, kernal=(3, 3, 16, 32), drop=drop, phase=phase, scope='layer2_1')
    layer2 = conv_relu_drop(x=layer2, kernal=(3, 3, 32, 32), drop=drop, phase=phase, scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = max_pool2d(x=layer2)
    # layer3->convolution
    layer3 = conv_relu_drop(x=down2, kernal=(3, 3, 32, 64), drop=drop, phase=phase, scope='layer3_1')
    layer3 = conv_relu_drop(x=layer3, kernal=(3, 3, 64, 64), drop=drop, phase=phase, scope='layer3_2')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = max_pool2d(x=layer3)
    # layer4->convolution
    layer4 = conv_relu_drop(x=down3, kernal=(3, 3, 64, 128), drop=drop, phase=phase, scope='layer4_1')
    layer4 = conv_relu_drop(x=layer4, kernal=(3, 3, 128, 128), drop=drop, phase=phase, scope='layer4_2')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = max_pool2d(x=layer4)
    # layer5->convolution
    layer5 = conv_relu_drop(x=down4, kernal=(3, 3, 128, 256), drop=drop, phase=phase, scope='layer5_1')
    layer5 = conv_relu_drop(x=layer5, kernal=(3, 3, 256, 256), drop=drop, phase=phase, scope='layer5_2')
    layer5 = resnet_Add(x1=down4, x2=layer5)
    # down sampling5
    down5 = max_pool2d(x=layer5)
    # layer6->convolution
    layer6 = conv_relu_drop(x=down5, kernal=(3, 3, 256, 512), drop=drop, phase=phase, scope='layer6_1')
    layer6 = conv_relu_drop(x=layer6, kernal=(3, 3, 512, 512), drop=drop, phase=phase, scope='layer6_2')
    layer6 = resnet_Add(x1=down5, x2=layer6)
    # down sampling5
    down6 = max_pool2d(x=layer6)

    gap = tf.reduce_mean(down6, axis=(1, 2))
    # layer7->FC1
    layer7 = tf.reshape(gap, [-1, 512])  # shape=(?, 512)
    layer7 = full_connected_relu_drop(x=layer7, kernal=(512, 512), drop=drop, activefunction='relu',
                                      scope='fc1')
    # layer7->output
    output = full_connected_relu_drop(x=layer7, kernal=(512, n_class), drop=drop, activefunction='regression',
                                      scope='output')
    return output


# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class ResNet2dModule(object):
    """
        A ResNet2d implementation
        :param image_height: number of height in the input image
        :param image_width: number of width in the input image
        :param image_depth: number of depth in the input image
        :param channels: number of channels in the input image
        :param costname: name of the cost function.Default is "dice coefficient"
    """

    def __init__(self, image_height, image_width, channels=1, n_class=2, costname="cross_entropy",
                 inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.n_class = n_class

        self.X = tf.placeholder("float", shape=[None, self.image_height, self.image_width, self.channels])
        self.Y_gt = tf.placeholder("float", shape=[None, self.n_class])
        self.lr = tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)
        self.drop = tf.placeholder('float')
        self.alpha = [1, 1, 1, 1, 1]
        self.gamma = 4
        self.Y_pred_logits = _create_conv_net(self.X, self.image_width, self.image_height, self.channels, self.drop,
                                              self.phase, n_class=n_class)

        self.Y_pred = tf.nn.softmax(self.Y_pred_logits)
        if costname == "focal_loss":
            self.cost = self.__get_cost(costname, self.Y_pred)
        else:
            self.cost = self.__get_cost(costname, self.Y_pred_logits)
        self.predict = tf.argmax(self.Y_pred, 1)
        self.accuracy = self.__get_accuracy(self.Y_pred)
        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)
        else:
            self.sess = tf.InteractiveSession(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    def __get_cost(self, cost_name, Y_pred):
        if cost_name == "cross_entropy":
            # the function first calculate softmax then calculate the
            # cross_entropy(-tf.reduce_sum(self.Y_gt*tf.log(Y_pred))),
            # logits don't through the tf.nn,softmax function
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_gt, logits=Y_pred))
            return cost
        if cost_name == "focal_loss":
            weight_loss = np.array(self.alpha)
            epsilon = 1.e-5
            # Scale predictions so that the class probas of each sample sum to 1
            output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
            # Clip the prediction value to prevent NaN's and Inf's
            output = tf.clip_by_value(output, epsilon, 1. - epsilon)
            # Calculate Cross Entropy
            cross_entropy = -self.Y_gt * tf.log(output)
            # Calculate Focal Loss
            loss = tf.pow(1 - output, self.gamma) * cross_entropy
            loss = tf.reduce_mean(loss, axis=0)
            loss = tf.reduce_mean(weight_loss * loss)
            return loss

    def __get_accuracy(self, Y_pred):
        correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(self.Y_gt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
        return accuracy

    def train(self, train_images, train_lanbels, val_images, val_labels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=5,
              batch_size=1):
        label_counts = np.unique(train_lanbels).shape[0]
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(logs_path + "model\\"):
            os.makedirs(logs_path + "model\\")
        model_path = logs_path + "model\\" + model_path
        # update the moving average of batch norm before finish the training step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ensures that we execute the update_ops before performing the train_step
            train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        self.sess.run(init)

        ckpt = tf.train.get_checkpoint_state(logs_path + "model\\")
        if ckpt and ckpt.model_checkpoint_path:
            print('Checkpoint file: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        all_data_number = train_images.shape[0]
        DISPLAY_STEP = all_data_number // batch_size
        index_in_epoch = 0

        train_epochsall = all_data_number * train_epochs
        val_epochs = val_images.shape[0]
        for i in range(train_epochsall):
            # get new batch for training
            batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(train_images, train_lanbels, batch_size,
                                                                       index_in_epoch)
            # label one_hot coding
            train_labels_onehot = dense_to_one_hot(batch_ys_path, label_counts)
            batch_ys = train_labels_onehot.astype(np.float)
            train_images_data = np.empty((len(batch_xs_path), self.image_height, self.image_width, self.channels))
            for num in range(len(batch_xs_path)):
                train_image = cv2.imread(batch_xs_path[num], 0)
                train_image = cv2.resize(train_image, (self.image_height, self.image_width))
                # Normalize from [0:255] => [0.0:1.0]
                train_image = (train_image - np.mean(train_image)) / np.std(train_image)
                batchimage = np.reshape(train_image, (self.image_height, self.image_width, self.channels))
                train_images_data[num, :, :, :] = batchimage
            batch_xs = train_images_data.astype(np.float)
            # Extracting images and labels from given data
            batch_ys = batch_ys.astype(np.float)
            # check progress on every step
            if i % DISPLAY_STEP == 0:
                train_loss, train_accuracy = self.sess.run([self.cost, self.accuracy],
                                                           feed_dict={self.X: batch_xs,
                                                                      self.Y_gt: batch_ys,
                                                                      self.lr: learning_rate,
                                                                      self.drop: 1,
                                                                      self.phase: 1})
                print('epochs %d training_loss ,training_accuracy=> %.5f,%.5f ' % (i, train_loss, train_accuracy))
                # for validation process add validation loss and accuracy into summary
                # get new batch for validation
                val_index_in_epoch = 0
                validataion_loss_list = []
                validataion_accuracy_list = []
                for _ in range(0, val_epochs, batch_size):
                    valbatch_xs_path, valbatch_ys_path, val_index_in_epoch = _next_batch(val_images, val_labels,
                                                                                         batch_size, val_index_in_epoch)
                    # label one_hot coding
                    val_labels_onehot = dense_to_one_hot(valbatch_ys_path, label_counts)
                    valbatch_ys = val_labels_onehot.astype(np.float)
                    val_images_data = np.empty(
                        (len(valbatch_xs_path), self.image_height, self.image_width, self.channels))
                    for num in range(len(valbatch_xs_path)):
                        val_image = cv2.imread(valbatch_xs_path[num], 0)
                        val_image = cv2.resize(val_image, (self.image_height, self.image_width))
                        # Normalize from [0:255] => [0.0:1.0]
                        val_image = (val_image - np.mean(val_image)) / np.std(val_image)
                        valbatchimage = np.reshape(val_image, (self.image_height, self.image_width, self.channels))
                        val_images_data[num, :, :, :] = valbatchimage
                    valbatch_xs = val_images_data.astype(np.float)
                    # Extracting images and labels from given data
                    valbatch_ys = valbatch_ys.astype(np.float)
                    # get validation loss and validation accuracy from validation dataset
                    validataion_loss, validataion_accuracy = self.sess.run([self.cost, self.accuracy],
                                                                           feed_dict={self.X: valbatch_xs,
                                                                                      self.Y_gt: valbatch_ys,
                                                                                      self.lr: learning_rate,
                                                                                      self.drop: 1,
                                                                                      self.phase: 1})
                    validataion_loss_list.append(validataion_loss)
                    validataion_accuracy_list.append(validataion_accuracy)
                validataion_loss = np.mean(validataion_loss_list)
                validataion_accuracy = np.mean(validataion_accuracy_list)
                print('epochs %d valid_loss,valid_accuracy => %5f,%5f ' % (i, validataion_loss, validataion_accuracy))
                validationloss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="valid_loss", simple_value=validataion_loss)])
                validationaccuracy_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="valid_accuracy", simple_value=validataion_accuracy)])
                summary_writer.add_summary(validationloss_summary, i)
                summary_writer.add_summary(validationaccuracy_summary, i)

                save_path = saver.save(self.sess, model_path, global_step=i)
                print("Model saved in file:", save_path)
                if DISPLAY_STEP <= all_data_number:
                    DISPLAY_STEP *= 2
                if DISPLAY_STEP > all_data_number:
                    DISPLAY_STEP = all_data_number
            # train operation
            _, summary = self.sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                                 self.Y_gt: batch_ys,
                                                                                 self.lr: learning_rate,
                                                                                 self.drop: dropout_conv,
                                                                                 self.phase: 1})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(self.sess, model_path)
        print("Model saved in file:", save_path)
        self.sess.close()

    def prediction(self, test_images):
        test_images = (test_images - np.mean(test_images)) / np.std(test_images)
        test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], 1))
        y_dummy = np.empty((test_images.shape[0], self.n_class))
        predictvaluetmp, predict_probvaluetmp = self.sess.run([self.predict, self.Y_pred],
                                                              feed_dict={self.X: test_images,
                                                                         self.Y_gt: y_dummy,
                                                                         self.drop: 1,
                                                                         self.phase: 1})
        predictvalue, predict_probvalue = predictvaluetmp[0], predict_probvaluetmp[0][predictvaluetmp[0]]
        return predictvalue, predict_probvalue

    def closs_session(self):
        self.sess.close()
        tf.reset_default_graph()
