import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2
train, test = train_test_split(os.listdir('C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images'), test_size=0.1, shuffle=True, random_state=42)


for i in test:
    image = cv2.imread(f'C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images\\{i}')
    print(f'C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images_test\\{i}')
    cv2.imwrite(f'C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images_test\\{i}', image)

for i in train:
    image = cv2.imread(f'C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images\\{i}')
    print(f'C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images_train\\{i}')
    cv2.imwrite(f'C:\\Users\\User\\PycharmProjects\\ARISE\\data\\croped_images_train\\{i}', image)