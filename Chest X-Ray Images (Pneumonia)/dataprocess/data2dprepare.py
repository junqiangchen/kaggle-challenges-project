from dataprocess.utils import file_name_path

NORMAL = "NORMAL"
BAC_PNEUMONIA = "PNEUMONIA"


def splitclassifyintotraintest(path, file_name):
    label_zero_path = path + "/" + NORMAL
    label_one_path = path + "/" + BAC_PNEUMONIA

    label_zero_files = file_name_path(label_zero_path, False, True)
    label_one_files = file_name_path(label_one_path, False, True)
    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for index in range(len(label_zero_files)):
        out.writelines("0" + "," + label_zero_path + "/" + str(label_zero_files[index]) + "\n")
    for index in range(len(label_one_files)):
        out.writelines("1" + "," + label_one_path + "/" + str(label_one_files[index]) + "\n")


if __name__ == '__main__':
    splitclassifyintotraintest(r"E:\MedicalData\kagglechallenge\Chest X-Ray Images (Pneumonia)\train",
                               "trainclassifydata.csv")
    splitclassifyintotraintest(r"E:\MedicalData\kagglechallenge\Chest X-Ray Images (Pneumonia)\test",
                               "testclassifydata.csv")
    print('success')
