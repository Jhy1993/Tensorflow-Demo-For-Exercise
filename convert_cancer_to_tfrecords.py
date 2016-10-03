# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 14:31:10 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
translate csv into tfrecoder
Reference:
https://github.com/tobegit3hub/deep_recommend_system/blob/master/data/convert_cancer_to_tfrecords.py
"""
import tensorflow as tf 
import os

def conver_tfrecords(input_filename, output_filename):
    current_path = os.getcwd()#return current path
    input_file = os.path.join(current_path, input_filename)
    output_file = os.path.join(current_path, output_filename)
    print("Start to conver {} to {}".format(input_file, output_file))

    writer =tf.python_io.TFRecordWriter(output_file)
    
    for line in open(input_file, "r"):
        data = line.split(",")
        label = float(data[9])
        features = [float(i) for i in data[0:9]]
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
            tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "features":
            tf.train.Feature(float_list=tf.train.FlaotList(value=features)),
            }))
        writer.write(example.SerializeToString())

    writer.close()
    print("Sucessfully conver {} to {}".format(input_file, output_file))

current_path = os.getcwd()
for file in os.listdir(current_path):
    if file.startswith("cancer_train") and file.endwith(".csv") and not file.endwith(".tfrecords"):
        conver_tfrecords(file, file + ".tfrecords")
        
