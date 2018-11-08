# -*- coding: utf-8 -*-
import tensorflow as tf
import os.path
import numpy as np

def network(model_version,num_labels,n_hidden_1,n_hidden_2,inputfile1,inputfile2,learning_rate,training_epochs,display_step):
    #训练好的模型分类器保存位置
    MODEL_DIR = "../networkmodel"
    MODEL_NAME = "networkmodel"
    if not tf.gfile.Exists(MODEL_DIR):
        tf.gfile.MakeDirs(MODEL_DIR)
    output_path = os.path.join(tf.compat.as_bytes(MODEL_DIR),tf.compat.as_bytes(str(model_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(output_path)
    #训练样本读入
    data1 = inputfile1.readlines()
    xs = []
    labels = []
    for data11 in data1:
        dt1 = data11.strip().split(',')
        p = int(len(dt1) - num_labels)
        xs.append(dt1[1:p])
        labels.append(dt1[p:])
    #测试样本读入
    data2 = inputfile2.readlines()
    test_xs = []
    test_labels = []
    for data21 in data2:
        dt2 = data21.strip().split(',')
        q = int(len(dt2) - num_labels)
        test_xs.append(dt2[1:q])
        test_labels.append(dt2[q:])
    #数组类型转换
    xs = np.array(xs)
    labels = np.array(labels)
    test_xs = np.array(test_xs)
    test_labels = np.array(test_labels)
    #打乱训练样本顺序
    arr = np.arange(xs.shape[0])
    np.random.shuffle(arr) 
    xs = xs[arr, :] 
    labels = labels[arr, :] 
    # 网络参数
    n_input = int(len(xs[0])) # 样本特征数
    #准备好placeholder
    X = tf.placeholder(tf.float32, [None, n_input], name='X_placeholder')
    Y = tf.placeholder(tf.int32, [None, num_labels], name='Y_placeholder')
    #两个隐层和输出层的权重
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_labels]), name='W')
    }
    #偏置在隐层的时候和节点个数一致，在输出层和输出类别个数一致
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
        'out': tf.Variable(tf.random_normal([num_labels]), name='bias')
    }
    #构建网络计算graph
    #此函数是用来构建计算的神经网络，得出预测类别的得分
    def multilayer_perceptron(x, weights, biases):
        # 第1个隐层，使用relu激活函数
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='fc_1')
        layer_1 = tf.nn.relu(layer_1, name='relu_1')
        # 第2个隐层，使用relu激活函数
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name='fc_2')
        layer_2 = tf.nn.relu(layer_2, name='relu_2')
        # 输出层
        out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='fc_3')
        return out_layer
    #拿到预测类别score
    pred = multilayer_perceptron(X, weights, biases)
    #计算损失函数值并初始化optimizer
    loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y, name='cross_entropy_loss')
    loss = tf.reduce_mean(loss_all, name='avg_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #初始化变量
    init = tf.global_variables_initializer()
    #把计算图写入事件文件
    writer = tf.summary.FileWriter(logdir='../tensorboard_network', graph=tf.get_default_graph())
    writer.close()
    #声明tf.train.Saver用于保存模型
    saver = tf.train.Saver()
    #在session中执行graph定义的运算
    with tf.Session() as sess:
        sess.run(init)
        # 训练
        for epoch in range(training_epochs):
            avg_loss = 0.
            # 使用optimizer进行优化
            _, l = sess.run([optimizer, loss], feed_dict={X: xs, Y: labels})
            # 求平均的损失
            avg_loss += l / int(len(data1))
            '''
            # 每一步都展示信息
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_loss))
        print("Optimization Finished!")
        '''
        # 在测试集上评估
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", sess.run(accuracy,feed_dict = {X: test_xs, Y: test_labels}))
        saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME))
        builder.save()
    
if __name__ == '__main__':    
    model_version=1
    num_labels = 3 #分类数
    n_hidden_1 = 10 # 第1个隐层节点数
    n_hidden_2 = 10 # 第2个隐层节点数
    inputfile1=open('train.txt', 'r') #训练样本
    inputfile2=open('test.txt', 'r') #测试样本
    learning_rate = 0.001
    training_epochs = 1000 #训练总轮数
    display_step = 10 #信息展示的频度
    network(model_version,num_labels,n_hidden_1,n_hidden_2,inputfile1,inputfile2,learning_rate,training_epochs,display_step)
    