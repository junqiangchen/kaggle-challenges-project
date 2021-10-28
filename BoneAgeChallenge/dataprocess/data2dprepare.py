import pandas as pd


def splitclassifyintotraintest(file_name):
    classify = r'E:\MedicalData\kagglechallenge\RSNABoneAge\boneage-training-dataset\boneage-training-dataset.csv'
    image_path = r"E:\MedicalData\kagglechallenge\RSNABoneAge\boneage-training-dataset\boneage-training-dataset"
    csvdata = pd.read_csv(classify)
    traindata = csvdata.iloc[:, :].values
    mean_bone_age = csvdata['boneage'].mean()
    std_bone_age = csvdata['boneage'].std()
    print('mean: ' + str(mean_bone_age))
    print('std: ' + str(std_bone_age))
    imagedata = traindata[:, 0]
    labeldata = traindata[:, 1]
    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for index in range(len(imagedata)):
        # models perform better when features are normalised to have zero mean and unity standard deviation
        # using z score for the training
        z_scorelabel = (labeldata[index] - mean_bone_age) / (std_bone_age)
        out.writelines(str(z_scorelabel) + "," + image_path + "/" + str(imagedata[index]) + ".png" + "\n")


if __name__ == '__main__':
    splitclassifyintotraintest("regressionzscoredata.csv")
    print('success')
