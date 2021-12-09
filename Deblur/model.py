from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from SRNDeblur.util.util import *
from SRNDeblur.util.BasicConvLSTMCell import *

BASE_DIR = 'Deblur/SRNDeblur/checkpoints'

class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels
        self.crop_size = 256
        self.train_dir = os.path.join(BASE_DIR, args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        tf.reset_default_graph() 
        
    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in xrange(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                    # encoder
                    conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                    conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                    conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                    conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                    conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                    conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                    conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                    conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                    conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                    conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                    if self.args.model == 'lstm':
                        deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                    else:
                        deconv3_4 = conv3_4

                    # decoder
                    deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                    deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                    deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                    deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                    cat2 = deconv2_4 + conv2_4
                    deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                    deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                    deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                    deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                    deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                    deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                    inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

            return x_unwrap

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width,image):
        #if not os.path.exists(output_path):
        #    os.makedirs(output_path)
        #imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, step=523000)

        #blur = scipy.misc.imread(os.path.join(input_path, imgName))
        blur = image
        h, w, c = blur.shape
        # make sure the width is larger than the height
        rot = False
        if h > w:
            blur = np.transpose(blur, [1, 0, 2])
            rot = True
        h = int(blur.shape[0])
        w = int(blur.shape[1])
        resize = False
        if h > H or w > W:
            scale = min(1.0 * H / h, 1.0 * W / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
            resize = True
            blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
        else:
            blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
        blurPad = np.expand_dims(blurPad, 0)
        if self.args.model != 'color':
            blurPad = np.transpose(blurPad, (3, 1, 2, 0))

        start = time.time()
        deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
        #duration = time.time() - start
        #print('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgName), duration))
        res = deblur[-1]
        if self.args.model != 'color':
            res = np.transpose(res, (3, 1, 2, 0))
        res = im2uint8(res[0, :, :, :])
        # crop the image into original size
        if resize:
            res = res[:new_h, :new_w, :]
            res = scipy.misc.imresize(res, [h, w], 'bicubic')
        else:
            res = res[:h, :w, :]

        if rot:
            res = np.transpose(res, [1, 0, 2])
        #scipy.misc.imsave(os.path.join(output_path, imgName), res)
        return res
