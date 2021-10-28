import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModule
import cv2
import pandas as pd
import numpy as np


def predict():
    test_datacsv = pd.read_csv("dataprocess\data\\test.csv")
    test_data = test_datacsv.iloc[:, :].values
    # For Image
    images = test_data[:, 1]
    # For Labels
    labels = test_data[:, 0]

    predictvalues = []
    for num in range(len(images)):
        src_image = cv2.imread(images[num], 0)
        resize_image = cv2.resize(src_image, (256, 256))
        ResNet2dbinary = ResNet2dModule(256, 256, channels=1, n_class=2, costname="cross_entropy", inference=True,
                                        model_path=r"log\classifybinary\model\resnet.pd")
        predictvaluebinary, _ = ResNet2dbinary.prediction(resize_image)
        ResNet2dbinary.closs_session()
        if predictvaluebinary == 1:
            ResNet2dfour = ResNet2dModule(256, 256, channels=1, n_class=4, costname="cross_entropy", inference=True,
                                          model_path=r"log\classifyfour\model\resnet.pd")
            predictvaluefour, _ = ResNet2dfour.prediction(resize_image)
            ResNet2dfour.closs_session()
            predictvalues.append(predictvaluefour + 1)
        else:
            predictvalues.append(predictvaluebinary)

    name = 'classify_metrics.csv'
    out = open(name, 'w')
    out.writelines("y_predict" + "," + "y_score" + "," + "y_true" + "\n")
    labels = labels.tolist()
    for index in range(np.shape(images)[0]):
        out.writelines(
            str(predictvalues[index]) + "," + str(predictvalues[index]) + "," + str(labels[index]) + "\n")


if __name__ == "__main__":
    predict()
