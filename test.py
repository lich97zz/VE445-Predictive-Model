import numpy as np
import tensorflow as tf
import os
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

'''
def evaluate():
##load a trained model
    tf.reset_default_graph()
    path = os.path.split(os.path.realpath(__file__))[0][:-12] 
    data = np.load(path + '\\test\\names_onehots.npy', allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    data_size = onehots.shape[0]

    d_x, d_y = load_data(path+'\\train',balance=False)
    e_x, e_y = load_data(path+'\\validation',balance=False)
    b_x = np.concatenate((d_x,e_x), 0)
    b_y = np.concatenate((d_y,e_y), 0)
    
    t_x, t_y = load_data(path+'\\train',balance=False)
    
    _, _, loss, accuracy, recall, precision = model(t_x.shape[1:],0.01)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, path+ '\\516370910248\\model')

    result = []

    test_output = sess.run('output/BiasAdd:0', {'input:0': b_x})
    pred = np.argmax(test_output, axis=1)
    result.extend(list(pred))
        
    sess.close()

    TP,TN,FP,FN=0,0,0,0
    for i, v in enumerate(result):
        if v == 1 and b_y[i]==1:
            TP+=1
        elif v == 0 and b_y[i]==1:
            FN+=1
        elif v == 1 and b_y[i]==0:
            FN+=1
        elif v == 0 and b_y[i]==0:
            TN+=1
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    balanced_accuracy = (TP)/(TP+FN)+FP/(FP+TN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("accuracy:",round(accuracy,2))
    print("balanced_accuracy:",round(balanced_accuracy,2))
    print("precision:",round(precision,2))
    print("recall:",round(recall,2))
    print("F1:",round((2*precision*recall)/(precision+recall),2))
'''    
        
def load_model():
##load a trained model
    tf.reset_default_graph()
    path = os.path.split(os.path.realpath(__file__))[0][:-12] 
    data = np.load(path + '\\test\\names_onehots.npy', allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    data_size = onehots.shape[0]

    b_x, b_y = load_data(path+'\\train',balance=False)
    _, _, loss, accuracy, recall, precision = model(b_x.shape[1:],0.01)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, path+ '\\516370910248\\model')

    result = []

    test_output = sess.run('output/BiasAdd:0', {'input:0': onehots})
    pred = np.argmax(test_output, axis=1)
    result.extend(list(pred))
        
    sess.close()

    file = path+'\\output_516370910248.csv'
    with open(file,'w') as f:
        f.write('Chemical,Label\n')
        for i, v in enumerate(result):
            f.write(name[i] + ',%d\n' % v)
        print('Successfully predicted in ',file)
    
load_model()
