import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import Vnet2dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv(
        r'D:\cjq\project\python\kagglechalleng\pneumonia_detection_2018\dataprocess\dataset\trainseg.csv')
    trainData = csvdata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    np.random.shuffle(trainData)
    imagedata = trainData[:, 0]
    maskdata = trainData[:, 1]

    Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient")
    Vnet2d.train(imagedata, maskdata, "Vnet2d.pd", "log\\segmeation\\vnet2d\\", 0.001, 0.5, 10, 6)


if __name__ == '__main__':
    train()
    print('success')
