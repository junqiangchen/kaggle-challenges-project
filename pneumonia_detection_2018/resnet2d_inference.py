import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModule
from dataprocess.utils import file_name_path
import cv2
import numpy as np

image_pre = ".jpg"


def inference():
    ResNet3d = ResNet2dModule(256, 256, channels=1, n_class=2, costname="cross_entropy", inference=True,
                              model_path=r"log\classify\resnetcross_entropy\model\resnet.pd-55840")
    test_image_path = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\input\samples"
    name = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\input\test_classify.csv"
    out = open(name, 'w')
    out.writelines("patientId" + "," + "Target" + "\n")
    allfilenames = file_name_path(test_image_path, False, True)
    for number in range(len(allfilenames)):
        imagefile = allfilenames[number]
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, 0)
        resize_image = cv2.resize(src_image, (256, 256))
        resize_image = (resize_image - np.mean(resize_image)) / np.std(resize_image)
        predictvalue, predict_prob = ResNet3d.prediction(resize_image)
        out.writelines(imagefile[: -len(image_pre)] + "," + str(int(predictvalue)) + "\n")


if __name__ == "__main__":
    inference()
