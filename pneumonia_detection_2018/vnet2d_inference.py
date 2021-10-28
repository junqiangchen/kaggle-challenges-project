import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModule
from Vnet2d.vnet_model import Vnet2dModule
from dataprocess.utils import file_name_path
import cv2
import os
import numpy as np


def predict_test():
    test_image_path = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\input\samples"
    test_mask_path = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\input\samplesmask"
    allfilenames = file_name_path(test_image_path, False, True)
    for number in range(len(allfilenames)):
        imagefile = allfilenames[number]
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, 0)
        resize_image = cv2.resize(src_image, (256, 256))
        resize_image = (resize_image - np.mean(resize_image)) / np.std(resize_image)
        ResNet3d = ResNet2dModule(256, 256, channels=1, n_class=2, costname="cross_entropy", inference=True,
                                  model_path=r"log\classify\resnetcross_entropy\model\resnet.pd-55840")
        predictvalue, _ = ResNet3d.prediction(resize_image)
        ResNet3d.closs_session()
        if predictvalue == 1:
            Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient", inference=True,
                                  model_path="log\segmeation\\vnet2d\model\Vnet2d.pd")
            resize2_image = cv2.resize(src_image, (512, 512))
            resize2_image = (resize2_image - np.mean(resize2_image)) / np.std(resize2_image)
            pd_mask_image = Vnet2d.prediction(resize2_image)
            new_mask_image = cv2.resize(pd_mask_image, (src_image.shape[1], src_image.shape[0]))
            maskfilepath = os.path.join(test_mask_path, imagefile)
            cv2.imwrite(maskfilepath, new_mask_image)
            Vnet2d.closs_session()


if __name__ == "__main__":
    predict_test()
    print('success')
