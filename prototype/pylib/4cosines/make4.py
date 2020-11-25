import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
import matplotlib
from PIL import Image
import cv2
from matplotlib.pyplot import *
from numpy.linalg import norm
import pylab
import pygame



#!!!! Images are rotated in make stamps

width = 170
height = 170
periods = 2
hf_periods = 39
stampwidth = 145
stampheight = 145
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
    folder = "/home/samir/db3/prototype/pylib/oralcosines/"



def make_gamma(w, h):
    g = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]
    # w = round(w/2)
    w= w-10
    h=h-10
    ima = np.full((w + 10, h + 10), 0)
    l = np.zeros(h)
    for i in range(23):
        for j in range(2, round(h/24)):
            l[ i*round(h/24) + j] = g[i]
    for k in range(round(w/2)):
        ima[ k+5, 5:-5] = l
    l = np.zeros(h)
    for i in range(23):
        for j in range(2, round(h/24)):
            l[ i*round(h/24) + j] = g[23-i]
    for k in range(round(w/2), w):
        ima[k+5, 5:-5] = l
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


def makeimage(w, h, wvcount, phi):#,g):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    raw_inp = np.ones(w)
    for i in range(w):
        raw_inp[i] = 255.0*(.5 + .5*np.cos(np.pi*(wvcount*i/w-phi)))
        # raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(np.pi*(1.0*float(phi )+ 2*wvcount*float(i)/float(w))))
        # raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi )/3.0 + wvcount*float(i)/float(w))))
        # imaline[i] = raw_inp[i]*255*(imaline[i]/255)**1#(1/.8)*1.8 # Add gamma compensation!!
        imaline[i] = raw_inp[i]
    for j in range(h):
        ima[:, j] = imaline
    print(ima.shape)
    return ima

# def makeimage(w, h, wvcount, phi):#,g):
#     ima = np.zeros((w, h))
#     imaline = np.ones(w)
#     raw_inp = np.ones(w)
#     for i in range(w):
#         # raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(np.pi*(1.0*float(phi )+ 2*wvcount*float(i)/float(w))))
#         raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi )/3.0 + wvcount*float(i)/float(w))))
#         # raw_inp[i] = g(255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi)/3.0 + wvcount*float(i)/float(w)))))
#         # imaline[i] = np.polyval(gamma_correct, raw_inp[i])
#         imaline[i] = raw_inp[i]*255*(imaline[i]/255)**1#(1/.8)*1.8 # Add gamma compensation!!
#     for j in range(h):
#         ima[:, j] = imaline

#     # marker = markerline(w, modulo)
#     # for j in range(h-100, h):
#     #     ima[:, j] = marker
#     # ima = np.transpose(ima)

#     # cv2.imwrite(str(phi + 1) + '_cos.jpg', ima)
#     print(ima.shape)
#     return ima


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

def makestamps(stampcount, wvcount, seq, phi,folder):#,g):
    wholeima =  np.zeros((width,height))
    stampimage = makeimage(stampwidth, stampheight, wvcount, phi)#,g)
    for i in range(stampcount):
        startx, starty = getstart(i)
        print(i, startx, starty)
        copystamp(startx, starty, stampimage, wholeima)
    # stampimage = make_gamma(stampwidth, stampheight)
    # startx, starty = getstart(stampcount-1)
    # copystamp(startx, starty, stampimage, wholeima)
    wholeima = addborders(wholeima, 250)
    wholeima = np.transpose(wholeima)  
    file = folder + str(seq + 1) + '_cos.jpg'
    # gray = cv2.cvtColor(wholeima, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros([height,width,3])
    # img2[:,:,0] = wholeima
    img2[:,:,1] = wholeima
    # img2[:,:,2] = wholeima
    Rotated = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(folder + str(seq + 1) + '_cos.png', Rotated)# img2)
 
    
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



def ideal(x,y):
    coefs = np.polyfit(x,y,1)
    polynom = np.poly1d(coefs)
    print('ideal polynom', polynom)
    return polynom

def gamma_comp(garray, line_array):
    g_comp = np.zeros(25)
    for i in range(0, 25):
        g_comp[i] = line_array[i] - (garray[i] - line_array[i])
    print('g_comp=', g_comp)
    return g_comp


def comp_num_gamma():
    gamma_curve = [0,6,9,13,16,17,22,26,34,38,47,54,60,72,74,84,93,101,114,158,174,181,185,200,255]
    yarray = gamma_curve
    xarray = np.arange(0, 250, 10)
    # yarray[23:] = 255
    yarray[0] = yarray[1]
    print(yarray)
    poly = np.polyfit(xarray, yarray, 8)
    print('poly:', poly)
    x = np.linspace(0, 240, 25)
    lx = np.zeros(2)
    ly = np.zeros(2)
    lx[0] = 0
    lx[1] = 250
    ly[0] = yarray[0]
    ly[1] = yarray[24]
    polynom = ideal(lx, ly)
    ideal_line = polynom(xarray)
    gideal = gamma_comp(yarray, ideal_line)
    g_poly_comp = np.polyfit(xarray, gideal, 8)
    plot(xarray, np.polyval(polynom, xarray), 'b-')
    plot(xarray, np.polyval(poly, xarray), 'g-')
    plot(xarray, np.polyval(g_poly_comp, xarray), 'r-')
    plot(xarray, gideal, 'b o')
    plot(xarray, yarray, 'r o')
    g = np.poly1d(g_poly_comp)
    for i in range(1, 25):
        print('g', i, g(i*10))

    show()
    # gamma_cos(width, height, periods, -1, g)
    # gamma_cos(width, height, periods, 0, g)
    # gamma_cos(width, height, periods, 1, g)
    # gamma_cos(width, height, 1, 5, g)
    # gamma_cos(width, height, 1, 6, g)
    # gamma_cos(width, height, 1, 7, g)
    folder = "/home/samir/db3/prototype/pylib/oralcosines/gammacos/"
    makestamps(squares, hf_periods, -1, folder,g)
    makestamps(squares,hf_periods, 0, folder,g)
    makestamps(squares, hf_periods, 1, folder,g)
    makestamps(squares, periods, 5,  folder,g)
    makestamps(squares, periods, 6,  folder,g)
    makestamps(squares, periods, 7,  folder,g)

    return g_poly_comp
# file = '/home/samir/db2/scan/static/scan_folder/gamma_im_folder/image1.png'
# gamma_correct = compensate_gamma(file)
# comp_num_gamma()
def makemanystamps():
    folder = "/home/samir/db3/prototype/pylib/oralcosines/"
    makestamps(squares, hf_periods, -1, 0, folder)
    makestamps(squares,hf_periods, 0, 2/3, folder)
    makestamps(squares, hf_periods, 1, 4/3, folder)
    makestamps(squares, hf_periods, 2, 1/3, folder)
    makestamps(squares,hf_periods, 3, 1, folder)
    makestamps(squares, hf_periods, 4, 5/3, folder)
    makestamps(squares, periods, 5, 0,  folder)
    makestamps(squares, periods, 6, 2/3,  folder)
    makestamps(squares, periods, 7, 4/3,  folder)
    makestamps(squares, periods, 8, 1/3,  folder)
    makestamps(squares, periods, 9, 1,  folder)
    makestamps(squares, periods, 10, 5/3,  folder)

def makemygama(h,w):
    file = "/home/samir/db3/prototype/pylib/4cosines/" + 'gamma.png'
    # gray = cv2.cvtColor(wholeima, cv2.COLOR_BGR2GRAY)
    wholeima = make_gamma(w,h)
    img2 = np.zeros([height,width,3])
    # img2[:,:,0] = wholeima
    img2[:,:,1] = wholeima
    # img2[:,:,2] = wholeima
    Rotated = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(file, Rotated)# img2)
# makemaskstamps(squares, folder)
# maketexture(width, height, 100,folder)
# makeblack(width, height,0, folder)
# make_gamma( stampwidth, stampheight)
# addindexedtext(folder + '0_cos.jpg', '1')
# addindexedtext(folder + '1_cos.jpg', '2')
# addindexedtext(folder + '2_cos.jpg', '3')
# addindexedtext(folder + '6_cos.jpg', '4')
# addindexedtext(folder + '7_cos.jpg', '5')
# addindexedtext(folder + '8_cos.jpg', '6')    folder = "/home/samir/db3/prototype/pylib/oralcosines/"

# img2[:,:,1] = ima
# cv2.imwrite(folder + 'gamma11.png', img2)
# for j in range(24):
#     gamma_image(width, height, j*10)
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

folder = "/home/samir/db3/prototype/pylib/4cosines/"
makestamps(squares, hf_periods, 0, 0, folder)
makestamps(squares,hf_periods, 1, 1/2, folder)
makestamps(squares, hf_periods, 2, 1, folder)
makestamps(squares, hf_periods, 3, 3/2, folder)
makestamps(squares, periods, 5, 0, folder)
makestamps(squares,periods, 6, 1/2, folder)
makestamps(squares, periods, 7, 1, folder)
makestamps(squares, periods, 8, 3/2, folder)

maketexture(width,height,230,folder)
makeblack(width,height,0,folder)
# makemygama(width,height,folder)