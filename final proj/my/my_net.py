import sys, os
sys.path.append("C:/repo/computer_vision/final proj")
from common.layers import *
from collections import OrderedDict
from common.trainer import Trainer
import pickle
import numpy as np

class myAlex:
    """
    AlexNet 구현
    
    conv - relu - 
    conv - relu - pooling -
    conv - relu -
    conv - relu - 
    conv - relu - pooling -
    fc - fc - softmax
    
    """
    
    def __init__(self, input_dim=(3, 32, 32),
                conv_param_1={'filter_num':96, 'filter_size':11, 'pad':1, 'stride':3},
                conv_param_2={'filter_num':256, 'filter_size':5, 'pad':2, 'stride':1},
                conv_param_3={'filter_num':384, 'filter_size':3, 'pad':1, 'stride':1}, 
                conv_param_4={'filter_num':384, 'filter_size':3, 'pad':1, 'stride':1}, 
                conv_param_5={'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1}, 
                hidden_size=50, output_size=10, weight_init_std=0.01):
        
        pre_node_nums = np.array([3*3*3, 
                                  96*3*3, 
                                  256*3*3, 
                                  384*3*3, 384*3*3, 
                                  256*3*3, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값
        
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        # 계층 생성===========
        #1층
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        
        #2층
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=3, pool_w=3, stride=2))
        self.layers.append(BatchNormalization(0.5, 0.5))
        
        #3층
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        
        #4층
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        
        #5층
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=3, pool_w=3, stride=2))
       
        #6층
        #Flatten
        self.layers.append(Affine(np.random.randn(256, 10), np.zeros(10)))
        
        #7층
        self.layers.append(Relu())
        
        #8층
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 6, 8, 10, 13)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]