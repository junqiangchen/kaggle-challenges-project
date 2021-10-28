from dataprocess.utils import file_name_path
import pandas as pd
import numpy as np

flag0 = "Acne and Rosacea Photos"
flag1 = "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions"
flag2 = "Atopic Dermatitis Photos"
flag3 = "Bullous Disease Photos"
flag4 = "Cellulitis Impetigo and other Bacterial Infections"
flag5 = "Eczema Photos"
flag6 = "Exanthems and Drug Eruptions"
flag7 = "Hair Loss Photos Alopecia and other Hair Diseases"
flag8 = "Herpes HPV and other STDs Photos"
flag9 = "Light Diseases and Disorders of Pigmentation"
flag10 = "Lupus and other Connective Tissue diseases"
flag11 = "Melanoma Skin Cancer Nevi and Moles"
flag12 = "Nail Fungus and other Nail Disease"
flag13 = "Poison Ivy Photos and other Contact Dermatitis"
flag14 = "Psoriasis pictures Lichen Planus and related diseases"
flag15 = "Scabies Lyme Disease and other Infestations and Bites"
flag16 = "Seborrheic Keratoses and other Benign Tumors"
flag17 = "Systemic Disease"
flag18 = "Tinea Ringworm Candidiasis and other Fungal Infections"
flag19 = "Urticaria Hives"
flag20 = "Vascular Tumors"
flag21 = "Vasculitis Photos"
flag22 = "Warts Molluscum and other Viral Infections"


def splitclassifyintotraintest(path, file_name):
    label_0_path = path + "/" + flag0
    label_1_path = path + "/" + flag1
    label_2_path = path + "/" + flag2
    label_3_path = path + "/" + flag3
    label_4_path = path + "/" + flag4
    label_5_path = path + "/" + flag5
    label_6_path = path + "/" + flag6
    label_7_path = path + "/" + flag7
    label_8_path = path + "/" + flag8
    label_9_path = path + "/" + flag9
    label_10_path = path + "/" + flag10
    label_11_path = path + "/" + flag11
    label_12_path = path + "/" + flag12
    label_13_path = path + "/" + flag13
    label_14_path = path + "/" + flag14
    label_15_path = path + "/" + flag15
    label_16_path = path + "/" + flag16
    label_17_path = path + "/" + flag17
    label_18_path = path + "/" + flag18
    label_19_path = path + "/" + flag19
    label_20_path = path + "/" + flag20
    label_21_path = path + "/" + flag21
    label_22_path = path + "/" + flag22
    label_0_files = file_name_path(label_0_path, False, True)
    label_1_files = file_name_path(label_1_path, False, True)
    label_2_files = file_name_path(label_2_path, False, True)
    label_3_files = file_name_path(label_3_path, False, True)
    label_4_files = file_name_path(label_4_path, False, True)
    label_5_files = file_name_path(label_5_path, False, True)
    label_6_files = file_name_path(label_6_path, False, True)
    label_7_files = file_name_path(label_7_path, False, True)
    label_8_files = file_name_path(label_8_path, False, True)
    label_9_files = file_name_path(label_9_path, False, True)
    label_10_files = file_name_path(label_10_path, False, True)
    label_11_files = file_name_path(label_11_path, False, True)
    label_12_files = file_name_path(label_12_path, False, True)
    label_13_files = file_name_path(label_13_path, False, True)
    label_14_files = file_name_path(label_14_path, False, True)
    label_15_files = file_name_path(label_15_path, False, True)
    label_16_files = file_name_path(label_16_path, False, True)
    label_17_files = file_name_path(label_17_path, False, True)
    label_18_files = file_name_path(label_18_path, False, True)
    label_19_files = file_name_path(label_19_path, False, True)
    label_20_files = file_name_path(label_20_path, False, True)
    label_21_files = file_name_path(label_21_path, False, True)
    label_22_files = file_name_path(label_22_path, False, True)
    print(len(label_0_files), len(label_1_files), len(label_2_files), len(label_3_files), len(label_4_files),
          len(label_5_files), len(label_6_files), len(label_7_files), len(label_8_files), len(label_9_files),
          len(label_10_files), len(label_11_files), len(label_12_files), len(label_13_files), len(label_14_files),
          len(label_15_files), len(label_16_files), len(label_17_files), len(label_18_files), len(label_19_files),
          len(label_20_files), len(label_21_files), len(label_22_files))

    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for index in range(len(label_0_files)):
        out.writelines("0" + "," + label_0_path + "/" + str(label_0_files[index]) + "\n")
    for index in range(len(label_1_files)):
        out.writelines("1" + "," + label_1_path + "/" + str(label_1_files[index]) + "\n")
    for index in range(len(label_2_files)):
        out.writelines("2" + "," + label_2_path + "/" + str(label_2_files[index]) + "\n")
    for index in range(len(label_3_files)):
        out.writelines("3" + "," + label_3_path + "/" + str(label_3_files[index]) + "\n")
    for index in range(len(label_4_files)):
        out.writelines("4" + "," + label_4_path + "/" + str(label_4_files[index]) + "\n")
    for index in range(len(label_5_files)):
        out.writelines("5" + "," + label_5_path + "/" + str(label_5_files[index]) + "\n")
    for index in range(len(label_6_files)):
        out.writelines("6" + "," + label_6_path + "/" + str(label_6_files[index]) + "\n")
    for index in range(len(label_7_files)):
        out.writelines("7" + "," + label_7_path + "/" + str(label_7_files[index]) + "\n")
    for index in range(len(label_8_files)):
        out.writelines("8" + "," + label_8_path + "/" + str(label_8_files[index]) + "\n")
    for index in range(len(label_9_files)):
        out.writelines("9" + "," + label_9_path + "/" + str(label_9_files[index]) + "\n")
    for index in range(len(label_10_files)):
        out.writelines("10" + "," + label_10_path + "/" + str(label_10_files[index]) + "\n")
    for index in range(len(label_11_files)):
        out.writelines("11" + "," + label_11_path + "/" + str(label_11_files[index]) + "\n")
    for index in range(len(label_12_files)):
        out.writelines("12" + "," + label_12_path + "/" + str(label_12_files[index]) + "\n")
    for index in range(len(label_13_files)):
        out.writelines("13" + "," + label_13_path + "/" + str(label_13_files[index]) + "\n")
    for index in range(len(label_14_files)):
        out.writelines("14" + "," + label_14_path + "/" + str(label_14_files[index]) + "\n")
    for index in range(len(label_15_files)):
        out.writelines("15" + "," + label_15_path + "/" + str(label_15_files[index]) + "\n")
    for index in range(len(label_16_files)):
        out.writelines("16" + "," + label_16_path + "/" + str(label_16_files[index]) + "\n")
    for index in range(len(label_17_files)):
        out.writelines("17" + "," + label_17_path + "/" + str(label_17_files[index]) + "\n")
    for index in range(len(label_18_files)):
        out.writelines("18" + "," + label_18_path + "/" + str(label_18_files[index]) + "\n")
    for index in range(len(label_19_files)):
        out.writelines("19" + "," + label_19_path + "/" + str(label_19_files[index]) + "\n")
    for index in range(len(label_20_files)):
        out.writelines("20" + "," + label_20_path + "/" + str(label_20_files[index]) + "\n")
    for index in range(len(label_21_files)):
        out.writelines("21" + "," + label_21_path + "/" + str(label_21_files[index]) + "\n")
    for index in range(len(label_22_files)):
        out.writelines("22" + "," + label_22_path + "/" + str(label_22_files[index]) + "\n")


def choose_nobalancesample(file_name):
    train_csv = pd.read_csv(r'train.csv')
    train_Data = train_csv.iloc[:, :].values
    trainimages = train_Data[:, 1]
    trainlabels = train_Data[:, 0]

    numlabel = []
    maxlabel = np.max(trainlabels)
    for label in range(0, maxlabel + 1, 1):
        nums = np.sum(trainlabels == label)
        numlabel.append(nums)
    maxlabelnum = max(numlabel)

    nobalancelabel = []
    for label in range(0, maxlabel + 1, 1):
        nums = np.sum(trainlabels == label)
        if nums < maxlabelnum / 2:
            nobalancelabel.append(label)

    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for index in range(len(trainlabels)):
        if trainlabels[index] in nobalancelabel:
            out.writelines(str(trainlabels[index]) + "," + str(trainimages[index]) + "\n")


if __name__ == '__main__':
    # splitclassifyintotraintest(r"E:\MedicalData\kagglechallenge\Dermnet\train",
    #                            "trainclassifydata.csv")
    # splitclassifyintotraintest(r"E:\MedicalData\kagglechallenge\Dermnet\test",
    #                            "testclassifydata.csv")
    choose_nobalancesample("trainnobalance.csv")
    print('success')
