import glob
import re
import tensorflow as tf
import random
import numpy as np
import os
from skimage import transform


#build filenames -> labels
def build_filenames_labels(path,mode):
    path=path+'/'+mode
    label_dict={}
    filenames_labels=[]
    files=os.listdir(path)#file names in path
    i=0
    for file in files:
        if os.path.isdir(path+'/'+file):
            label_dict[file]=i
            imgnames=os.listdir(path+'/'+file)
            for imgname in imgnames:
                if 'jpeg' in imgname or 'jpg' in imgname:
                    filenames_labels.append((path+'/'+file+'/'+imgname,str(i)))
            i+=1
    print("#####"+str(len(filenames_labels))+"#####")
    return label_dict,filenames_labels


def read_image(filename_q, mode):
  """Load next jpeg file from filename / label queue
  Randomly applies distortions if mode == 'train' (including a 
  random crop to [56, 56, 3]). Standardizes all images.

  Args:
    filename_q: Queue with 2 columns: filename string and label string.
     filename string is relative path to jpeg file. label string is text-
     formatted integer between '0' and '199'
    mode: 'train' or 'val'

  Returns:
    [img, label]: 
      img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
      label = tf.unit8 target class label: {0 .. 199}
  """
  item = filename_q.dequeue()
  filename = item[0]
  label = item[1]
  file = tf.read_file(filename)
  img = tf.image.decode_image(file, channels=3)
 
  #img = tf.decode_raw(file, channels=3)
  # image distortions: left/right, random hue, random color saturation
  if mode == 'train':
    img = tf.random_crop(img, np.array([56, 56, 3]))
    img = tf.image.random_flip_left_right(img)
    # val accuracy improved without random hue
    # img = tf.image.random_hue(img, 0.05)
    img = tf.image.random_saturation(img, 0.5, 2.0)
  else:
    img = tf.image.crop_to_bounding_box(img, 4, 4, 56, 56)

  label = tf.string_to_number(label, tf.int32)
  label = tf.cast(label, tf.uint8)

  return [img, label]


def batch_q(mode, config):
  """Return batch of images using filename Queue

  Args:
    mode: 'train' or 'val'
    config: training configuration object

  Returns:
    imgs: tf.uint8 tensor [batch_size, height, width, channels]
    labels: tf.uint8 tensor [batch_size,]

  """
  label_dict, filenames_labels = build_filenames_labels(config.path, mode)
  random.shuffle(filenames_labels)
  filename_q = tf.train.input_producer(filenames_labels,
                                       num_epochs=config.num_epochs,
                                       shuffle=True)

  # 2 read_image threads to keep batch_join queue full:
  return tf.train.batch_join([read_image(filename_q, mode) for i in range(2)],
                             config.batch_size, shapes=[(56, 56, 3), ()],
                             capacity=2048)


label_dict, filenames_labels = build_filenames_labels('flowers', 'train')
#print(filenames_labels)