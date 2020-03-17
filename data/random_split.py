import os
import numpy as np

f_img = open("./list_image.txt","r")
f_object = open("./object_annotation.txt","r")
f_hand = open("./hand_annotation.txt","r")

f_img_train = open("./train/list_image_train.txt", "w")
f_object_train = open("./train/object_annotation_train.txt", "w")
f_hand_train = open("./train/hand_annotation_train.txt", "w")

f_img_test = open("./train/list_image_test.txt", "w")
f_object_test = open("./train/object_annotation_test.txt", "w")
f_hand_test = open("./train/hand_annotation_test.txt", "w")

lines_img = f_img.readlines()
lines_hand = f_hand.readlines()
lines_object = f_object.readlines()

randoms = np.random.random(len(lines_img))
for i, line in enumerate(lines_img):
    if randoms[i] > 0.9:
        f_img_test.write(lines_img[i])
        f_object_test.write(lines_object[i])
        f_hand_test.write(lines_hand[i])
    else:
        f_img_train.write(lines_img[i])
        f_object_train.write(lines_object[i])
        f_hand_train.write(lines_hand[i])

f_img.close()
f_object.close()
f_hand.close()

f_img_train.close()
f_object_train.close()
f_hand_train.close()

f_img_test.close()
f_object_test.close()
f_hand_test.close()