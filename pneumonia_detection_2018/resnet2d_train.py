import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    # csv file should have the type:
    # label,data_npy
    # label,data_npy
    # ....
    #
    train_csv = pd.read_csv(
        r'dataprocess\dataset\trainclassify.csv')
    train_Data = train_csv.iloc[:, :].values
    np.random.shuffle(train_Data)
    # For Image
    trainimages = train_Data[:, 1]
    trainlabels = train_Data[:, 0]

    val_csv = pd.read_csv(r'dataprocess\dataset\validationclassify.csv')
    val_Data = val_csv.iloc[:, :].values
    np.random.shuffle(val_Data)
    # For Image
    valimages = val_Data[:, 1]
    vallabels = val_Data[:, 0]

    ResNet3d = ResNet2dModule(256, 256, channels=1, n_class=2, costname="cross_entropy")
    ResNet3d.train(trainimages, trainlabels, valimages, vallabels, "resnet.pd", "log\\classify\\resnetcross_entropy\\",
                   0.001, 0.5, 10, 64)


if __name__ == "__main__":
    train()
