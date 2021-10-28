import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import Vnet2dModule
from dataprocess.utils import calcu_iou
import cv2
import pandas as pd
import numpy as np


def predict_test():
    Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient", inference=True,
                          model_path="log\segmeation\\vnet2d\model\Vnet2d.pd")
    csvdata = pd.read_csv(
        r'D:\cjq\project\python\kagglechalleng\pneumonia_detection_2018\dataprocess\dataset\validationseg.csv')
    traindata = csvdata.iloc[:, :].values
    imagedata = traindata[:, 0]
    maskdata = traindata[:, 1]
    iou_values = []
    for i in range(len(imagedata)):
        src_image = cv2.imread(imagedata[i], 0)
        mask_image = cv2.imread(maskdata[i], 0)
        resize_image = cv2.resize(src_image, (512, 512))
        resize_image = (resize_image - np.mean(resize_image)) / np.std(resize_image)
        pd_mask_image = Vnet2d.prediction(resize_image)
        pd_mask_image[pd_mask_image >= 128] = 255
        pd_mask_image[pd_mask_image < 128] = 0
        new_mask_image = cv2.resize(pd_mask_image, (mask_image.shape[1], mask_image.shape[0]))
        cv2.imwrite("D:\cjq\project\python\kagglechalleng\pneumonia_detection_2018\dataprocess\dataset\\" + str(
            i) + "image.bmp", src_image)
        cv2.imwrite(
            "D:\cjq\project\python\kagglechalleng\pneumonia_detection_2018\dataprocess\dataset\\" + str(i) + "mask.bmp",
            mask_image)
        cv2.imwrite("D:\cjq\project\python\kagglechalleng\pneumonia_detection_2018\dataprocess\dataset\\" + str(
            i) + "pdmask.bmp", new_mask_image)
        iou_value = calcu_iou(mask_image, new_mask_image, 255)
        iou_values.append(iou_value)
    print("mean iou:", np.mean(iou_values))


if __name__ == "__main__":
    predict_test()
    print('success')
