# coding: utf-8
import sys, os
sys.path.append("C:/repo/computer_vision/final proj")  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from my_net import myAlex
import pickle
from common.trainer import Trainer

cifar10=datasets.cifar10
(train_images, train_labels), (test_images, test_labels)=cifar10.load_data()
train_images = np.transpose(train_images, (0, 3, 1, 2))
test_images = np.transpose(test_images, (0, 3, 1, 2))

valid_images=[]
valid_labels=[]

for i in range(5000):
    valid_images.append(test_images[i])
    valid_labels.append(test_labels[i])
    
test_images_5000=[]
test_labels_5000=[]

for i in range(5000):
    test_images_5000.append(test_images[5000+i])
    test_labels_5000.append(test_labels[5000+i])
    
test_images_5000=np.array(test_images_5000)
test_labels_5000=np.array(test_labels_5000)
valid_images=np.array(valid_images)
valid_labels=np.array(valid_labels)

print("Train samples:", train_images.shape, train_labels.shape)
print("Valid samples :", valid_images.shape, valid_labels.shape)
print("Test samples:", test_images_5000.shape, test_labels_5000.shape)

network = myAlex()
trainer = Trainer(network, train_images, train_labels, test_images, test_labels,
                  epochs=10, mini_batch_size=100,
                  optimizer='AdaGrad', optimizer_param={'lr':0.00001},
                  evaluate_sample_num_per_epoch=1000
                  )
trainer.train()

# 매개변수 보관
network.save_params("myAlex_params.pkl")
print("매개변수 값이 저장되었습니다.")