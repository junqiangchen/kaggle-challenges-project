import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModuleregression
import cv2
import pandas as pd
import numpy as np


def predict():
    test_datacsv = pd.read_csv(r'D:\cjq\project\python\kagglechalleng\BoneAgeChallenge\dataprocess\validation.csv')
    test_data = test_datacsv.iloc[:, :].values
    # For Image
    images = test_data[:, 1]
    # For Labels
    labels = test_data[:, 0]
    ResNet3d = ResNet2dModuleregression(256, 256, channels=1, n_class=1, mean_age=127.3207517246848,
                                        std_age=41.18202139939618, inference=True,
                                        model_path=r"log\regression\model\resnet.pd")
    for num in range(len(images)):
        src_image = cv2.imread(images[num], 0)
        resize_image = cv2.resize(src_image, (256, 256))
        resize_image = (resize_image - np.mean(resize_image)) / np.std(resize_image)
        predictvalue = ResNet3d.prediction(resize_image)
        outputimage = cv2.resize(src_image, (512, 512))
        true_age = labels[num] * 41.18202139939618 + 127.3207517246848
        predict_age = predictvalue
        cv2.putText(outputimage, 'Actual:' + str(true_age), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(outputimage, 'Predicted:' + str(predict_age), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))
        cv2.imwrite("D:\cjq\project\python\kagglechalleng\BoneAgeChallenge\dataprocess\dataset\\" + str(num) + ".bmp",
                    outputimage)


if __name__ == "__main__":
    predict()
