import os
import argparse
import tensorflow as tf
import torch
import numpy as np
import streamlit as st
import sys
sys.path.append(os.path.join(os.getcwd(),'Deblur/SRNDeblur'))
from SRNDeblur.run_model import parse_args
from SRNDeblur.models.model import DEBLUR
from SRNDeblur.util.util import *
from SRNDeblur.util.BasicConvLSTMCell import *

class Deblur():
    def __init__(self):
        self.args = parse_args()
        file = open(self.args.datalist, 'w')
        file.close()

    def predict(self,image):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tf.reset_default_graph()
        self.deblur = DEBLUR(self.args)
        self.deblur.train_dir = os.path.join(os.path.join(os.getcwd(),'Deblur/SRNDeblur'),self.deblur.train_dir)
        output = self.test(self.args.height, self.args.width,np.array(image))
        return output
        
    @st.cache
    def test(self, height, width,image):
        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.deblur.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.deblur.saver = tf.train.Saver()
        self.deblur.load(sess, self.deblur.train_dir, step=523000)

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

        deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
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

        return res

    