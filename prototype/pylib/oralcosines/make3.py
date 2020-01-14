import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
import matplotlib
 

width = 700
height = 480
periods = 1.5
hf_periods = 70
stampwidth = 600
stampheight = 450
stampborder = 7
widthcount = 1
heightcount =1
squares = widthcount*heightcount
xoffset = 0
yoffset = 0

def addtext(filename, text):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 28)
    draw.text((200, 200), text, (255), font=font)
    img.save(filename)

def addindexedtext(filename, text):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 86)   
    for i in range(squares):
        startx, starty = getstart(i)
        draw.text((startx, starty +stampheight/2), str(i), (255),font= font)
    img.save('index.jpg')

def gamma_image(w, h, value):
    ima = np.zeros((w, h))
    ima.fill(value*1)
    ima = np.transpose(ima)
    cv2.imwrite(folder+'gammas/'+'gamma' + str(value)+'.png', ima)
    # print(ima[50, :])
    return ima



def make_gamma(w, h):
    g = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]
    # w = round(w/2)
    ima = np.full((w + 10, h + 10), 0)
    l = np.zeros(w)
    for i in range(23):
        for j in range(round(w/24)):
            l[i*round(w/24) + j] = g[i]
    for k in range(round(h/2)):
        ima[5:-5, k+5] = l
    l = np.zeros(w)
    for i in range(23):
        for j in range(round(w/24)):
            l[i*round(w/24) + j] = g[23-i]
    for k in range(round(h/2), h):
        ima[5:-5, k+5] = l
    # marker = centerline(w)
    # for j in range(h-100, h):
    #     ima[:, j] = marker
    return ima

def markerline(w,modulo):
    line = np.zeros(w)
    for i in range(20, w):
        if divmod(i, modulo)[1] == 0:
            line[i] = 255
    for i in range(round(w/2)-2, round(w/2)+2):
            line[i] = 255
    return(line)

def centerline(w):
    line = np.zeros(w+10)
    for i in range(round(w/2)-2, round(w/2)+2):
        line[i] = 255
    return(line)


def makeimage(w, h, wvcount, phi, modulo):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    raw_inp = np.ones(w)
    for i in range(w):
        raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi)/3.0 + wvcount*float(i)/float(w))))
        # imaline[i] = np.polyval(gamma_correct, raw_inp[i])
        imaline[i] = raw_inp[i]*255*(imaline[i]/255)**(1/.9)*1.8 # Add gamma compensation!!
    for j in range(h):
        ima[:, j] = imaline

    # marker = markerline(w, modulo)
    # for j in range(h-100, h):
    #     ima[:, j] = marker
    # ima = np.transpose(ima)

    # cv2.imwrite(str(phi + 1) + '_cos.jpg', ima)
    print(ima.shape)
    print(ima[200,200])
    return ima

def maskimage(w, h,val):
        ima = np.full((w,h), val)
        return(ima)


def getstart(i):
    startx = xoffset+i%widthcount *(stampwidth+2*stampborder) + stampborder
    starty = yoffset+i//widthcount *(stampheight+2*stampborder) + stampborder

    return startx, starty

def copystamp(x,y, stamp, wholeima):
    for i in range (stampwidth):
        for j in range (stampheight):
            wholeima[x+i, y+j] = stamp[i, j]


def addborders(ima,val):
    for i in range(heightcount+1):
        for j in range(width-xoffset):
            ima[j+xoffset, i*(stampheight+2*stampborder)-2] = val
            ima[j+xoffset, i*(stampheight+2*stampborder)-1] = val
            ima[j+xoffset, i*(stampheight+2*stampborder)] = val
            ima[j+xoffset, i*(stampheight+2*stampborder)+1] = val
            ima[j+xoffset, i*(stampheight+2*stampborder)+2] = val
    for i in range(widthcount+1):
        for j in range(height):
            ima[ i*(stampwidth+2*stampborder)-2+xoffset, j] = val
            ima[ i*(stampwidth+2*stampborder)-1+xoffset, j] = val
            ima[ i*(stampwidth+2*stampborder)+xoffset, j] = val
            ima[i*(stampwidth+2*stampborder)+1+xoffset, j] = val
            ima[ i*(stampwidth+2*stampborder)+2+xoffset, j] = val
    
    return ima

def makestamps(stampcount, wvcount, phi, modulo, folder):
    wholeima =  np.zeros((width,height))
    stampimage = makeimage(stampwidth, stampheight, wvcount, phi,modulo)
    for i in range(stampcount):
        startx, starty = getstart(i)
        print(i, startx, starty)
        copystamp(startx, starty, stampimage, wholeima)
    # stampimage = make_gamma(stampwidth, stampheight)
    # startx, starty = getstart(stampcount-1)
    # copystamp(startx, starty, stampimage, wholeima)
    wholeima = addborders(wholeima, 250)
    wholeima = np.transpose(wholeima)
    file = folder + str(phi + 1) + '_cos.jpg'
    # gray = cv2.cvtColor(wholeima, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros([height,width,3])
    # img2[:,:,0] = wholeima
    img2[:,:,1] = wholeima
    # img2[:,:,2] = wholeima
    cv2.imwrite(folder + str(phi + 1) + '_cos.jpg', img2)
 
    
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

folder = "/home/samir/db3/prototype/pylib/oralcosines/"
makestamps(squares, hf_periods, -1, 8,folder)
makestamps(squares,hf_periods, 0, 8,folder)
makestamps(squares, hf_periods, 1, 8,folder)
makestamps(squares, periods, 5, 68, folder)
makestamps(squares, periods, 6, 68, folder)
makestamps(squares, periods, 7, 68, folder)
# makemaskstamps(squares, folder)
# maketexture(width, height, 100,folder)
# makeblack(width, height,0, folder)
# make_gamma( stampwidth, stampheight)
# addindexedtext(folder + '0_cos.jpg', '1')
# addindexedtext(folder + '1_cos.jpg', '2')
# addindexedtext(folder + '2_cos.jpg', '3')
# addindexedtext(folder + '6_cos.jpg', '4')
# addindexedtext(folder + '7_cos.jpg', '5')
# addindexedtext(folder + '8_cos.jpg', '6')

# ima = make_gamma( height-10 ,width-10)
# cv2.imwrite(folder + 'gamma.png', ima)
# j=0
# i=1
# while j < 8:
#     gamma_image(width, height, i+128)
#     j+=1
#     i=i<<1 
#     print(i)
# makeimage(width, height, hf_periods, -1)
# makeimage(width, height, hf_periods, 0)
# makeimage(width, height, hf_periods, 1)
# makeimage(width, height, periods, 5)
# makeimage(width, height, periods, 6)
# makeimage(width, height, periods, 7)
