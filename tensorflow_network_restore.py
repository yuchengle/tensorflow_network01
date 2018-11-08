# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import json

#构建计算的神经网络，预测类别的得分
def multilayer_perceptron(X,W1,W2,W,b1,b2,bias):
    layer_1 = tf.add(tf.matmul(X, W1), b1)
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.add(tf.matmul(layer_2, W), bias)
    return out_layer
#归一化函数
def autoNorm(dataSet):
    minVals = min(dataSet)
    maxVals = max(dataSet)
    ranges = maxVals - minVals
    if ranges == 0:
        ranges = 1
    normDataSet = []
    for i in range(len(dataSet)):
        data = np.round((dataSet[i]-minVals)/ranges,6)
        normDataSet.append(data)
    return normDataSet
def getresult(file):
    #数据准备
    inputfile=open(file,'r')
    datas = inputfile.readlines()
    test_xs = []
    mid = []
    for data in datas:
        dt = data.strip().split(',')
        test_xs.append(dt[1:])
        mid.append(dt[0])
    test_xs = np.array(test_xs) #数组类型转换
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../networkmodel/networkmodel.meta')
        saver.restore(sess,tf.train.latest_checkpoint('../networkmodel/'))
        
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X_placeholder:0")
        W1 = graph.get_tensor_by_name("W1:0")
        W2 = graph.get_tensor_by_name("W2:0")
        W = graph.get_tensor_by_name("W:0")
        b1 = graph.get_tensor_by_name("b1:0")
        b2 = graph.get_tensor_by_name("b2:0")
        bias = graph.get_tensor_by_name("bias:0")
    
        pred = multilayer_perceptron(X,W1,W2,W,b1,b2,bias)
        pred_class_index=tf.argmax(pred, 1)
        pred_value = sess.run(pred_class_index, feed_dict={X: test_xs})
        pred_probs = sess.run(pred,feed_dict = {X: test_xs})
        jsonList = []
        for i in range(len(mid)):
            pred_prob = autoNorm(pred_probs[i])
            confidence = max(pred_prob)/sum(pred_prob)
            Item = {}
            Item["id"] = mid[i]
            Item["pred_value"] = float(pred_value[i])
            Item["pred_prob"] = float(confidence)
            jsonList.append(Item)
        jsonArr = json.dumps(jsonList, sort_keys=True, ensure_ascii=False)
        return jsonArr
'''
if __name__ == '__main__':
    file='test0.txt'
    print(getresult(file))
'''