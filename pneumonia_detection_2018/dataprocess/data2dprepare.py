import cv2
import pandas as pd
import os
import numpy as np
from dataprocess.utils import file_name_path

image_pre = ".jpg"


def prepareclassifyandsegmentationdata():
    classify = r'E:\MedicalData\kagglechallenge\pneumonia_detection_2018\input\stage_2_train_labels.csv'
    image_path = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\input\images"
    classifyzerodir = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\classify\0"
    classifyonedir = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\classify\1"
    segimagedir = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\segmentation\Image"
    segmaskdir = r"E:\MedicalData\kagglechallenge\pneumonia_detection_2018\segmentation\Mask"
    all_image_files = file_name_path(image_path, False, True)
    csvdata = pd.read_csv(classify)
    traindata = csvdata.iloc[:, :].values
    imagedata = traindata[:, 0]
    xdata = traindata[:, 1]
    ydata = traindata[:, 2]
    widthdata = traindata[:, 3]
    heightdata = traindata[:, 4]
    labeldata = traindata[:, 5]

    for index in range(len(all_image_files)):
        image_file_path = image_path + "/" + all_image_files[index]
        image = cv2.imread(image_file_path, 0)
        image_file_pre = all_image_files[index][:-len(image_pre)]

        for csvindex in range(len(imagedata)):
            if image_file_pre == imagedata[csvindex]:
                if labeldata[csvindex] == 0:
                    zero_imagepath = classifyzerodir + "/" + str(image_file_pre) + ".jpg"
                    cv2.imwrite(zero_imagepath, image)
                if labeldata[csvindex] == 1:
                    one_imagepath = classifyonedir + "/" + str(image_file_pre) + ".jpg"
                    cv2.imwrite(one_imagepath, image)

        for csvindex in range(len(imagedata)):
            if image_file_pre == imagedata[csvindex]:
                if labeldata[csvindex] == 1:
                    seg_imagepath = segimagedir + "/" + str(image_file_pre) + ".jpg"
                    seg_maskpath = segmaskdir + "/" + str(image_file_pre) + ".jpg"
                    if os.path.exists(seg_maskpath):
                        seg_mask = cv2.imread(seg_maskpath, 0)
                        cv2.imwrite(seg_imagepath, image)
                        cv2.rectangle(seg_mask, (int(xdata[csvindex]), int(ydata[csvindex])),
                                      (int(xdata[csvindex] + widthdata[csvindex]),
                                       int(ydata[csvindex] + heightdata[csvindex])),
                                      (255, 255, 255), thickness=-1)
                        cv2.imwrite(seg_maskpath, seg_mask)
                    else:
                        cv2.imwrite(seg_imagepath, image)
                        seg_mask = np.zeros_like(image, 'uint8')
                        cv2.rectangle(seg_mask, (int(xdata[csvindex]), int(ydata[csvindex])),
                                      (int(xdata[csvindex] + widthdata[csvindex]),
                                       int(ydata[csvindex] + heightdata[csvindex])),
                                      (255, 255, 255), thickness=-1)
                        cv2.imwrite(seg_maskpath, seg_mask)


if __name__ == '__main__':
    prepareclassifyandsegmentationdata()
    print('success')
