import os
import pickle
import cv2
from PIL import Image

path = "/hard"

file_list = {}
for file in os.listdir(path):
    f = open(os.path.join(path, file), 'rb')
    f = pickle.load(f)
    # print(f)
    file_list[file.split('-')[-1].split('.')[0]] = f

img = file_list['feature']['3'][6]['feature']
im = Image.fromarray(img)
im.show()

I = file_list['feature']['3'][6]['feature']
I = cv2.resize(I, (8, 8))
im = Image.fromarray((I*256).astype('uint8'))