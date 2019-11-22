"""  TP: Face Recogntion Inference @ Summer School 04/07/2019
"""
# Zuheng Ming
# zuheng.ming@univ-lr.fr

import argparse
import os
import sys
import numpy as np
import cv2
from scipy import misc
import re
import time
import tensorflow as tf



class args_model():
    def __init__(self):
        self.images_placeholder = None
        self.embeddings = None
        self.keep_probability_placeholder = None
        self.phase_train_placeholder = None
        self.logits = None
        self.phase_train_placeholder_expression = None

def load_model(model_dir):
    with tf.device('/cpu:0'):

        # Load the model of face verification
        print('Model directory: %s' % model_dir)
        meta_file, ckpt_file = get_model_filenames(os.path.expanduser(model_dir))

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        model_dir_exp = os.path.expanduser(model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
        saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))

        args_model.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        args_model.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        args_model.keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
        args_model.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

    return sess, args_model

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_images(dir_images, image_size, do_prewhiten=True):
    image_paths = os.listdir(dir_images)
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i, img_path in enumerate(image_paths):
        img = misc.imread(os.path.join(dir_images, img_path))
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = cv2.resize(img, (image_size, image_size))
        images[i, :, :, :] = img
    return images

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def face_embeddings(images, sess, args_model):

    if len(images.shape)==3:
        images = np.expand_dims(images, axis=0)

    feed_dict = {args_model.phase_train_placeholder: False, args_model.images_placeholder: images, args_model.keep_probability_placeholder: 1.0}

    t2 = time.time()
    emb_array = sess.run([args_model.embeddings], feed_dict=feed_dict)
    embeddings = emb_array[0]
    t3 = time.time()
    #print('Embedding calculation FPS:%d' % (int(1 / (t3 - t2))))

    return embeddings

def main(args):

    sess, args_model = load_model(args.model_dir)
    
    face_imgs = load_images(args.img_ref, args.image_size)

    embeddings = face_embeddings(face_imgs, sess, args_model)

    # Caculate the distance of embeddings and verification the two face
    diff = np.subtract(embeddings[0], embeddings[1])
    dist = np.sum(np.square(diff))

    predict_issame = np.less(dist, args.threshold)

    if predict_issame:
        print('>>>>>>>>>>>>>>>>>> The identities of the two faces are same! @Threshold %f | The distance of embeddings is %f'%(args.threshold, dist))
    else:
        print('XXXXXXXXXXXXXXXXXX  The identities of the two faces are different!  @Threshold %f | The distance of embeddings is %f'%(args.threshold, dist))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='The device used for computing, the default device is CPU', default='CPU')
    parser.add_argument('--img_ref', type=str, help='Directory with unaligned image 1.', default='../data/images/test')

    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='/mnt/hgfs/share/models/20190629-011223/')
    parser.add_argument('--threshold', type=float,
                        help='The threshold for the face verification',default=0.9)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))