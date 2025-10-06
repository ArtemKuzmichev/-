import numpy as np
from PIL import Image


def zeros():
    
    img_mat=np.zeros((600,1000,3),dtype=np.uint8) # кодировавние 3 цветами одного пикселя
    for i in range(600):
        for j in range (1000):
            img_mat[i,j]=(i+j)%256

    return img_mat

img_mat=zeros()

img=Image.fromarray(img_mat,mode='RGB')
img.save('img0.png')
