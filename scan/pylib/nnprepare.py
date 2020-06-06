import cv2
import numpy as np
import os




def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def nn1folders(inpfolder, outfolder, count):
    for i in range(count):
        os.renames(inpfolder+'render'+str(i)+'/blendertexture.png', outfolder+'gray/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/-1nom.png', outfolder+'nom/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/-1nom.npy', outfolder+'nom/'+str(i)+'.npy')
        os.renames(inpfolder+'render'+str(i)+'/-1denom.png', outfolder+'denom/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/-1denom.npy', outfolder+'denom/'+str(i)+'.npy')
        os.renames(inpfolder+'render'+str(i)+'/im_wrap1.png', outfolder+'wrap/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/scan_wrap1.npy', outfolder+'wrap/'+str(i)+'.npy')


def nn4folders(inpfolder, outfolder, count):
    for i in range(count):
        # os.renames(inpfolder+'render'+str(i)+'/blendertexture.png', outfolder+'gray/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/5nom.png', outfolder+'nom/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/5nom.npy', outfolder+'nom/'+str(i)+'.npy')
        os.renames(inpfolder+'render'+str(i)+'/5denom.png', outfolder+'denom/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/5denom.npy', outfolder+'denom/'+str(i)+'.npy')
        os.renames(inpfolder+'render'+str(i)+'/im_wrap2.png', outfolder+'wrap/'+str(i)+'.png')
        os.renames(inpfolder+'render'+str(i)+'/scan_wrap2.npy', outfolder+'wrap/'+str(i)+'.npy')


inpfolder = '/home/samir/Desktop/blender/pycode/'+'scans'+'/'
outfolder1 = '/home/samir/Desktop/blender/pycode/'+'new1'+'/1/'
outfolder4 = '/home/samir/Desktop/blender/pycode/'+'new1'+'/4/'
# nn1folders(inpfolder,outfolder1,90)
nn4folders(inpfolder,outfolder4,90)