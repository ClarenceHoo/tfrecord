# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:20:25 2017

@author: hufei
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import tensorflow as tf 
from PIL import Image
from binvox_rw import read_as_3d_array
import numpy as np



image_dir = ""
voxel_dir = ""
image_format = ".png"
n_vox = 32
train_number = 2
img_num = 8
'''
数据目录如下：
图片目录：picture -- category  --  model_id -- rendring -- 0.png

label目录：model -- category  --  model_id -- model.binvox

'''
#这里是找寻所有该目录下的第一级子文件夹
import glob  
import os.path
from voxel import voxel2obj
def first_subdir(path):  
    list = []  
    if (os.path.exists(path)):  
    #获取该目录下的所有文件或文件夹目录路径  
        files = glob.glob(path + '/*' )  
        for file in files:  
            #判断该路径下是否是文件夹  
            if (os.path.isdir(file)):  
                #分成路径和文件的二元元组  
                h = os.path.split(file)    
                list.append(h[1])  
        return list

def second_subdir(path):  
    list = []  
    if (os.path.exists(path)):  
    #获取该目录下的所有文件或文件夹目录路径  
        files = glob.glob(path + '/*' )  
        for file in files:  
            #判断该路径下是否是文件夹  
            if (os.path.isdir(file)):  
                #分成路径和文件的二元元组
                h = os.path.split(file)  
                files = glob.glob(file + '/*' )
                for file2 in files: 
                    if (os.path.isdir(file2)):
                        h2 = os.path.split(file2)
                        list.append((h[1],h2[1]))
        return list        
        
def get_path_pire(image_path,voxel_path): 
#该函数主要实现得到一一匹配的图像文件夹路径和模型路径
    pair_list=[]
    category_object =second_subdir(image_path)
    for (category,object_id) in category_object:
        get_image_path= os.path.join(image_path,category+"/"+object_id+"/rendering")
        get_voxel_path= os.path.join(voxel_path,category+"/"+object_id)
        pair_list.append((get_image_path,get_voxel_path))
    return pair_list    

def load_img(image_dir,image_id,image_format=".png"):
    #实现功能：给定图像目录和图像id读取单张图片
    image_id = image_id 
    if image_id<10:
        image_id="0"+str(image_id)
    else:
        image_id=str(image_id)
    image_path = os.path.join(image_dir,image_id+image_format)
    im = Image.open(image_path)
    im = add_random_color_background(im)
    ims_c = crop_center(im,127,127)
    return ims_c
    
def load_imgs(image_path,img_num = img_num):
    #实现批量读取images
    image_ids = np.random.randint(20,size=img_num)
    im_matrix = []
    for id in image_ids:
        im_id = load_img(image_path,id,image_format=".png")
        im_matrix.append(im_id)
    return np.array(im_matrix)

def load_label(voxel_dir):
    voxel_fn = os.path.join(voxel_dir,"model.binvox")
    with open(voxel_fn, 'rb') as f:
        voxel = read_as_3d_array(f)
    return voxel

def tfrecode(image_dir,voxel_dir):
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    path_pire =  get_path_pire(image_dir,voxel_dir) 
    for (image_path,voxel_path) in path_pire:
        voxel = load_label(voxel_path)
        voxel_data = voxel.data
        voxel_raw = voxel_data.tobytes()
        for i in range(train_number):
            print("这是第%.4f个模型"%i)
            im = load_imgs(image_path)
            img_raw = im.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[voxel_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()
    
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    imgs = tf.decode_raw(features['img_raw'], tf.float64)
    imgs = tf.reshape(imgs,[127,127,img_num,3])
    imgs = tf.cast(imgs, tf.float32) * (1. / 255)
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [n_vox,n_vox,n_vox,1])
    label = tf.cast(label, tf.int32) 
    return imgs, label
    

def read_test():
    img, lab = read_and_decode("single_train.tfrecords")#使用shuffle_batch可以随机打乱输入
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(2):
            image,label= sess.run([img, lab])
            max,max2 = sess.run([tf.reduce_max(image),tf.reduce_max(label)])
            print(image.shape, label.shape)
            print(max,max2)
        coord.request_stop() 
        coord.join(threads)

def crop_center(im, new_height, new_width):
    height = im.shape[0]  # Get dimen;sions
    width = im.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return im[top:bottom, left:right,:]                

def add_random_color_background(im, color_range=[[225, 255], [225, 255], [225, 255]]):
    #给背景色加上随机颜色，颜色范围如下所示
    r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]

    if isinstance(im, Image.Image):
        im = np.array(im)

    if im.shape[2] > 3:
        # If the image has the alpha channel, add the background
        alpha = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float32)
		#这里处理首先是将im的alpha通道拿出来，为[137,137]，并加上一个新的轴变为[137,137,1]
		#然后判断矩阵值是否为0，若为0这替换为1，若不为0则替换为0
        im = im[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        im = alpha * bg_color + (1 - alpha) * im#这里是将图像简化，若alpha通道给出的透明值为0，那么在rgb加上全部赋值为255,256，通过alpha通道来极大的简化数据

    return im
            
if __name__ == '__main__':
    read_test()
    tfrecode(image_dir,voxel_dir)
