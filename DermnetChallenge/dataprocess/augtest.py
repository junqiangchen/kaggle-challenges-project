from dataprocess.Augmentation.ImageAugmentation import DataAug, DataAugClassify


def segdataAug():
    aug = DataAug(rotation=20, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation('segmeatationdata.csv', 2, path=r"E:\MedicalData\TNSCUI2020\segmentation\augtrain\\")


def classifydataAug():
    aug = DataAugClassify(rotation=5, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation('trainnobalance.csv', 2, path=r"E:\MedicalData\kagglechallenge\Dermnet\train_aug\\")


if __name__ == '__main__':
    classifydataAug()
    print('success')
