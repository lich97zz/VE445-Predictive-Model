import numpy as np
import tensorflow as tf
import os
import random
import time
LR = 0.01
BatchSize = 50
EPOCH = 1
def load_data(dir_path=os.path.dirname(os.path.realpath(__file__)), balance=False,simple=False):
##load the data from specified path
##if balance=True, adjust the data to be a balanced set, resample to minority class
##if simple=True, only load 1000 data
    label_file_name = dir_path + "\\names_labels.csv"
    smiles_file_name = dir_path + "\\names_smiles.csv"
    onehot_file_name = dir_path + "\\names_onehots.npy"
    data = np.load(onehot_file_name, allow_pickle=True).item()
    data = data['onehots']
    file_name = label_file_name
    if dir_path[-4:] == 'test':
        file_name = dir_path + "\\output_sample.csv"
    label_list = []
    with open(file_name,'r') as f:
        header = f.readline().replace('\n','').split(',')
        index = 0
        positive_count = 0
        negative_count = 0
        if header[0] != 'Label':
            index += 1
        for each in f.readlines():
            each = each.replace('\n','').split(',')
            if int(each[index]) == 1:
                positive_count += 1
            else:
                negative_count += 1
            label_list.append(int(each[index]))
    label_list = np.array(label_list)
    if balance:
        delta = negative_count - positive_count
        pair = []
        positive_index = []
        count = 0
        for i in range(len(label_list)):
            if label_list[i]==1:
                positive_index.append(i)  
            pair.append((data[i],label_list[i]))
        random.shuffle(positive_index)
        for i in range(delta):
            j = random.choice(positive_index)
            pair.append((data[j],label_list[j]))
        random.shuffle(pair)
        data = [each[0] for each in pair]
        label_list = [each[1] for each in pair]    
    if simple:
        data = data[:500]
        label_list = label_list[:500]    
##        print("info:",len(label_list), label_list.count(1))
##        for i in range(20):
##            print(label_list[i],' ')            
    data = np.array(data)
    label_list = np.array(label_list)

    return data, label_list
        
def model(input_shape,lr_):
##construction of the training model
##shape_ is 2D input data
##lr_ is learning rate
    conv_num = 32
    input_shape = list(input_shape)
    input = tf.placeholder(tf.float32, [None] + input_shape, name='input')
    input = tf.reshape(input, [-1] + input_shape + [1])
    label = tf.placeholder(tf.int32, [None], name='label')
    label = tf.one_hot(label, 2)

    conv1 = tf.keras.layers.Conv2D(conv_num, 5, 1, 'same', activation=tf.nn.relu)(input)
    pool1 = tf.keras.layers.MaxPool2D(2, 2)(conv1)
    conv2 = tf.keras.layers.Conv2D(conv_num, 3, (1, 2), padding='same', activation=tf.nn.relu)(pool1)
    pool2 = tf.keras.layers.MaxPool2D(2, 2)(conv2)
    conv3 = tf.keras.layers.Conv2D(conv_num, 3, 3, padding='same', activation=tf.nn.relu)(pool2)
    conv4 = tf.keras.layers.Conv2D(conv_num, 3, 3, padding='same', activation=tf.nn.relu)(conv3)    
    pool3 = tf.keras.layers.MaxPool2D(2, 2)(conv4)

    flat = tf.reshape(pool3, [-1, 1*3*conv_num])
    output1 = tf.keras.layers.Dense(16, name='output1')(flat)
    output2 = tf.keras.layers.Dense(16, name='output2')(output1)
    output = tf.keras.layers.Dense(2, name='output')(output2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=output) 

    train_op = tf.train.AdamOptimizer(lr_).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1), )[1]
    recall = tf.metrics.recall(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1), )[1]
    precision = tf.metrics.precision(labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1), )[1]
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    return init_op, train_op, loss, accuracy, recall, precision

def train_model(balance = True,simple=False, lr_ = LR):
##Train the model with learning rate=lr_
##load data as a balanced set or simplified set
    path = os.path.split(os.path.realpath(__file__))[0][:-12]
    train_x, train_y = load_data(path+'\\train', balance,simple)
    valid_x, valid_y = load_data(path+'\\validation')
    test_x, test_y = load_data(path+'\\test')
    init_op, train_op, loss, accuracy, recall, precision = model(train_x.shape[1:],lr_)
    sess = tf.Session()
    sess.run(init_op)
    saver = tf.train.Saver()
    train_size = train_x.shape[0]
    print("Training...")
    for epoch in range(EPOCH):
        time1=0
        if epoch<1:
            time1 = time.time()
        for i in range(0, train_size, BatchSize):
            b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
            _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})
        if epoch % 1 == 0:
            accuracy_ = 0
            recall_ = 0
            precision_ = 0
            for i in range(0, valid_x.shape[0], BatchSize):
                b_x, b_y = valid_x[i:i + BatchSize], valid_y[i:i + BatchSize]
                accuracy_ += sess.run(accuracy, {'input:0': b_x, 'label:0': b_y})
                recall_ += sess.run(recall, {'input:0': b_x, 'label:0': b_y})
                precision_ += sess.run(precision, {'input:0': b_x, 'label:0': b_y})
            accuracy_ = accuracy_ * BatchSize / valid_x.shape[0]
            print('INFO: Epoch ', epoch, ': Loss: %.4f' % loss_, ', Accuracy: %.2f' % accuracy_)
##            recall_ = recall_ * BatchSize / valid_x.shape[0]
##            precision_ = precision_ * BatchSize / valid_x.shape[0]
##            print('epoch:', epoch, '| train loss: %.4f' % loss_, '| valid accuracy: %.2f' % accuracy_,\
##                  '| valid recall: %.2f' % recall_,'| valid precision: %.2f' % precision_)
        if epoch < 1:
            dt = round(time.time()-time1,2)
            print("EPOCH TIME(sec):", dt)
            print("ESTIMATED TIME(min):", round(dt*EPOCH/60,1))
        if epoch % 10 == 0:
            lr_ = lr_/1.4
    accuracy_ = 0
    recall_ = 0
    precision_ = 0
    b_x,b_y = valid_x, valid_y
    accuracy_ = sess.run(accuracy, {'input:0': b_x, 'label:0': b_y})
    recall_ = sess.run(recall, {'input:0': b_x, 'label:0': b_y})
    precision_ = sess.run(precision, {'input:0': b_x, 'label:0': b_y})
    print("Validation accuracy,recall,precision:",(accuracy_,recall_,precision_))
    
    saver.save(sess, path+'\\weights\\model')
    sess.close()


train_model(balance=True,simple=True)


