import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
 

width = 4096
height = 2732
periods = 2
hf_periods = 17
stampwidth = 408
stampheight = 408
stampborder = 184
widthcount = 4
heightcount = 3
squares = 12

def addtext(filename, text):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 28)
    draw.text((200, 200), text, (255), font=font)
    img.save(filename)

def addindexedtext(filename, text):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 36)   
    for i in range(squares):
        startx, starty = getstart(i)
        draw.text((startx, starty +310), text, (255),font= font)
        draw.text((startx, starty +340), text, (255),font= font)
        draw.text((startx, starty +370), text, (255),font= font)
    img.save(filename)




def make_gamma(w, h):
    g = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]
    # w = round(w/2)
    ima = np.full((w + 10, h + 10), 0)
    l = np.zeros(w)
    for i in range(23):
        for j in range(round(w/48)):
            l[100 +i*round(w/48) + j] = g[i]
    for k in range(round(h/2)):
        ima[5:-5, k+5] = l
    l = np.zeros(w)
    for i in range(23):
        for j in range(round(w/48)):
            l[100+i*round(w/48) + j] = g[23-i]
    for k in range(round(h/2), h-100):
        ima[5:-5, k+5] = l
    marker = centerline(w)
    for j in range(h-100, h):
        ima[:, j] = marker
    return ima

def markerline(w,modulo):
    line = np.zeros(w)
    for i in range(20, w):
        if divmod(i, modulo)[1] == 0:
            line[i] = 200
    for i in range(round(w/2)-2, round(w/2)+2):
            line[i] = 200
    return(line)

def centerline(w):
    line = np.zeros(w+10)
    for i in range(round(w/2)-2, round(w/2)+2):
        line[i] = 200
    return(line)


def makeimage(w, h, wvcount, phi, modulo):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    raw_inp = np.ones(w)
    for i in range(w):
        raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi)/3.0 + wvcount*float(i)/float(w))))
        # imaline[i] = np.polyval(gamma_correct, raw_inp[i])
        imaline[i] = raw_inp[i]
    for j in range(h-100):
        ima[:, j] = imaline
    marker = markerline(w, modulo)
    for j in range(h-100, h):
        ima[:, j] = marker
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

def makestamps(stampcount, wvcount, phi, modulo, folder):
    wholeima =  np.zeros((width,height))
    stampimage = makeimage(stampwidth, stampheight, wvcount, phi,modulo)
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

folder = "/home/samir/db3/prototype/pylib/minicosines3/"
makestamps(squares, hf_periods, -1, 8,folder)
makestamps(squares, hf_periods, 0, 8,folder)
makestamps(squares, hf_periods, 1, 8,folder)
makestamps(squares, periods, 5, 68, folder)
makestamps(squares, periods, 6, 68, folder)
makestamps(squares, periods, 7, 68, folder)
makemaskstamps(squares, folder)
maketexture(width, height, 100,folder)
makeblack(width, height,0, folder)
# make_gamma( stampwidth, stampheight)
addindexedtext(folder + '0_cos.jpg', '1')
addindexedtext(folder + '1_cos.jpg', '2')
addindexedtext(folder + '2_cos.jpg', '3')
addindexedtext(folder + '6_cos.jpg', '4')
addindexedtext(folder + '7_cos.jpg', '5')
addindexedtext(folder + '8_cos.jpg', '6')


# makeimage(width, height, hf_periods, -1)
# makeimage(width, height, hf_periods, 0)
# makeimage(width, height, hf_periods, 1)
# makeimage(width, height, periods, 5)
# makeimage(width, height, periods, 6)
# makeimage(width, height, periods, 7)
