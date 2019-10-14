import cv2
import numpy as np
from PIL import Image
# from scan.pylib.gamma import *

width = 4096
height = 2732
periods = 2
hf_periods = 16
stampwidth = 400
stampheight = 400
stampborder = 200
widthcount = 4
heightcount = 3
squares = 12

def make_gamma(w, h):
    g = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]
    ima = np.full((w + 10, h + 10), 255)
    l = np.zeros(w)
    for i in range(23):
        for j in range(round(w/24)):
            l[i*round(w/24) + j] = g[i]
    for k in range(h):
        ima[5:-5, k+5] = l
    # ima = np.transpose(ima)
    # cv2.imwrite(folder + 'gamma1.png', ima)
    return ima


def makeimage(w, h, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    raw_inp = np.ones(w)
    for i in range(w):
        raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi)/3.0 + wvcount*float(i)/float(w))))
        # imaline[i] = np.polyval(gamma_correct, raw_inp[i])
        imaline[i] = raw_inp[i]
    for j in range(h):
        ima[:, j] = imaline
    # ima = np.transpose(ima)
    # cv2.imwrite(str(phi + 1) + '_cos.jpg', ima)
    return ima

def maskimage(w, h,val):
        ima = np.full((w,h), val)
        return(ima)


def getstart(i):
    startindex = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    startx = startindex[i][0] *(stampwidth+2*stampborder) + stampborder
    starty = startindex[i][1] *(stampheight+2*stampborder) + stampborder
    return startx, starty

def copystamp(x,y, stamp, wholeima):
    for i in range (stampwidth):
        for j in range (stampheight):
            wholeima[x+i, y+j] = stamp[i, j]


def addborders(ima,val):
    for i in range(heightcount+1):
        for j in range(width):
            ima[j, i*(stampheight+2*stampborder)-2] = val
            ima[j, i*(stampheight+2*stampborder)-1] = val
            ima[j, i*(stampheight+2*stampborder)] = val
            ima[j, i*(stampheight+2*stampborder)+1] = val
            ima[j, i*(stampheight+2*stampborder)+2] = val
    for i in range(widthcount+1):
        for j in range(height):
            ima[ i*(stampwidth+2*stampborder)-2, j] = val
            ima[ i*(stampwidth+2*stampborder)-1, j] = val
            ima[ i*(stampwidth+2*stampborder), j] = val
            ima[i*(stampwidth+2*stampborder)+1, j] = val
            ima[ i*(stampwidth+2*stampborder)+2, j] = val
    
    return ima

def makestamps(stampcount, wvcount, phi, folder):
    wholeima =  np.zeros((width,height))
    stampimage = makeimage(stampwidth, stampheight, wvcount, phi)
    for i in range(stampcount-1):
        startx, starty = getstart(i)
        print(i, startx, starty)
        copystamp(startx, starty, stampimage, wholeima)
    stampimage = make_gamma(stampwidth, stampheight)
    startx, starty = getstart(stampcount-1)
    copystamp(startx, starty, stampimage, wholeima)
    wholeima = addborders(wholeima, 250)
    wholeima = np.transpose(wholeima)
    cv2.imwrite(folder + str(phi + 1) + '_cos.jpg', wholeima)
 
    
def makemaskstamps(stampcount, folder):
    wholeima =  np.zeros((width,height))
    stampimage = maskimage(stampwidth, stampheight, 200)
    for i in range(stampcount):
        startx, starty = getstart(i)
        copystamp(startx, starty, stampimage, wholeima)
    wholeima = np.transpose(wholeima)
    cv2.imwrite(folder +  'mask.png', wholeima)


def maketexture(w, h, value, folder):
    ima = np.full((w,h), value)
    ima = np.transpose(ima)
    cv2.imwrite(folder + 'texture.png', ima)


def makeblack(w, h, value, folder):
    ima = np.full((w,h), value)
    ima = np.transpose(ima)
    cv2.imwrite(folder + 'black.png', ima)
# file = '/home/samir/db2/scan/static/scan_folder/gamma_im_folder/image1.png'
# gamma_correct = compensate_gamma(file)

folder = "/home/samir/db3/prototype/pylib/minicosines2/"
makestamps(squares, hf_periods, -1,folder)
makestamps(squares, hf_periods, 0,folder)
makestamps(squares, hf_periods, 1,folder)
makestamps(squares, periods, 5,folder)
makestamps(squares, periods, 6,folder)
makestamps(squares, periods, 7,folder)
makemaskstamps(squares, folder)
maketexture(width, height, 100,folder)
makeblack(width, height,0, folder)
# make_gamma( stampwidth, stampheight)


# makeimage(width, height, hf_periods, -1)
# makeimage(width, height, hf_periods, 0)
# makeimage(width, height, hf_periods, 1)
# makeimage(width, height, periods, 5)
# makeimage(width, height, periods, 6)
# makeimage(width, height, periods, 7)
