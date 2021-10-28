import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModuleregression
import numpy as np
import pandas as pd


def regressiontrain():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    # csv file should have the type:
    # label,data_npy
    # label,data_npy
    # ....
    #
    train_csv = pd.read_csv(r'D:\cjq\project\python\kagglechalleng\BoneAgeChallenge\dataprocess\train.csv')
    train_Data = train_csv.iloc[:, :].values
    np.random.shuffle(train_Data)
    # For Image
    trainimages = train_Data[:, 1]
    trainlabels = train_Data[:, 0]
    ResNet3d = ResNet2dModuleregression(256, 256, channels=1, n_class=1, mean_age=127.3207517246848,
                                        std_age=41.18202139939618)
    ResNet3d.train(trainimages, trainlabels, "resnet.pd", "log\\regression\\", 0.001, 0.5, 10, 64)


if __name__ == "__main__":
    regressiontrain()
