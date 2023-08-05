import os
import numpy as np
import cv2
import argparse
import multiprocessing

A = r'/home/liu/code/EPDN/datasets/Recover/train/trainA'
AB = r'/home/liu/code/EPDN/datasets/Recover/train/train2A'

def image_write(file):
    path_A = A +  r'/' + file
    path_AB = AB + r'/' + file
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=8)

for root,dirs,files in os.walk(A):
    pool.map(image_write, files)

