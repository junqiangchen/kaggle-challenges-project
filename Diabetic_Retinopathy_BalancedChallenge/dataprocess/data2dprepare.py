from dataprocess.utils import file_name_path
import pandas as pd

flag0 = "No_DR"
flag1 = "Mild"
flag2 = "Moderate"
flag3 = "Severe"
flag4 = "Proliferate_DR"


def splitclassifyintotraintest(path, file_name):
    label_0_path = path + "/" + flag0
    label_1_path = path + "/" + flag1
    label_2_path = path + "/" + flag2
    label_3_path = path + "/" + flag3
    label_4_path = path + "/" + flag4

    label_0_files = file_name_path(label_0_path, False, True)
    label_1_files = file_name_path(label_1_path, False, True)
    label_2_files = file_name_path(label_2_path, False, True)
    label_3_files = file_name_path(label_3_path, False, True)
    label_4_files = file_name_path(label_4_path, False, True)

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


def splitclassifyintotraintestbinarytask(path, file_name):
    train_csv = pd.read_csv(path)
    train_Data = train_csv.iloc[:, :].values
    # For Image
    trainimages = train_Data[:, 1]
    trainlabels = train_Data[:, 0]
    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for index in range(len(trainlabels)):
        if trainlabels[index] is 0:
            out.writelines("0" + "," + str(trainimages[index]) + "\n")
        else:
            out.writelines("1" + "," + str(trainimages[index]) + "\n")


def splitclassifyintotraintestfourtask(path, file_name):
    train_csv = pd.read_csv(path)
    train_Data = train_csv.iloc[:, :].values
    # For Image
    trainimages = train_Data[:, 1]
    trainlabels = train_Data[:, 0]
    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for index in range(len(trainlabels)):
        if trainlabels[index] is 1:
            out.writelines("0" + "," + str(trainimages[index]) + "\n")
        elif trainlabels[index] is 2:
            out.writelines("1" + "," + str(trainimages[index]) + "\n")
        elif trainlabels[index] is 3:
            out.writelines("2" + "," + str(trainimages[index]) + "\n")
        elif trainlabels[index] is 4:
            out.writelines("3" + "," + str(trainimages[index]) + "\n")


if __name__ == '__main__':
    # splitclassifyintotraintest(r"E:\MedicalData\kagglechallenge\Diabetic_Retinopathy\colored_images", "alldata.csv")
    splitclassifyintotraintestbinarytask(
        r"D:\cjq\project\python\kagglechalleng\Diabetic_Retinopathy_BalancedChallenge\dataprocess\data\trainaug.csv",
        "traindatabinarytask.csv")
    splitclassifyintotraintestbinarytask(
        r"D:\cjq\project\python\kagglechalleng\Diabetic_Retinopathy_BalancedChallenge\dataprocess\data\validation.csv",
        "validationdatabinarytask.csv")
    splitclassifyintotraintestfourtask(
        r"D:\cjq\project\python\kagglechalleng\Diabetic_Retinopathy_BalancedChallenge\dataprocess\data\trainaug.csv",
        "traindatafourtask.csv")
    splitclassifyintotraintestfourtask(
        r"D:\cjq\project\python\kagglechalleng\Diabetic_Retinopathy_BalancedChallenge\dataprocess\data\validation.csv",
        "validationdatafourtask.csv")
    print('success')
