import os

path = os.chdir("D:/Neil/college 2/Semester 4/Research/Heena/implementation/Model creation/dataset/train/without_mask/")

i=1
for file in os.listdir(path):

    new_file_name = "train_without_mask_{}.jpg".format(i)
    os.rename(file,new_file_name)

    i=i+1