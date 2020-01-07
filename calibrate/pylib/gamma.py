import cv2
import numpy as np
from PIL import Image
import cv2
from matplotlib.pyplot import *
from numpy.linalg import norm
import pylab
import pygame


width = 854
height = 480
periods = 49


def gamma_curve(folder):
    gam_curve = np.zeros(250, dtype=np.float)
    for i in range(1, 250):
        img = Image.open(folder + 'image' + str(i) + '.png').convert('L')
        img = np.array(img)
        print('shape', img.shape)
        print('i=', i, img[100, 100], img[100, 400])
        print(i*10,  np.average(img[50:100, 150:250]))
        gam_curve[i] = np.average(img[50:100, 150:250])
    return gam_curve


def make_grey_image():
    pygame.init()
    screen = pygame.display.set_mode([width, height])
    for i in range(255):
        screen.fill([i, i, i])
        pygame.display.update()


def make_image(w, h):
    g = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]
    ima = np.full((w + 10, h + 10), 255)
    l = np.zeros(w)
    for i in range(24):
        for j in range(round(w/24)):
            l[i*round(w/24) + j] = g[i]
    for k in range(h):
        ima[5:-5, k+5] = l
    ima = np.transpose(ima)
    cv2.imwrite('gamma1.png', ima)


def gamma_adjust(input):
    output = 255 *(input/255)**1/2.2
    return output


def take_im_slice(file):
    gamma_ima = cv2.imread(file)
    im_slice = gamma_ima[100:150, 110:590]
    return im_slice


def single_gamma_curve(file):
    gam_curve = np.zeros(25, dtype=np.float)
    img = cv2.imread(file)
    img = np.array(img)
    print('shape', img.shape)
    for i in range(0, 23):
        gam_curve[i] = np.average(img[0:10, (i*20 +5): (i*20 + 9)])
    return gam_curve


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


def gamma_cos(w, h, wvcount, phi, g):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    for i in range(w):
        imaline[i] = g(255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi)/3.0 + wvcount*float(i)/float(w)))))
    print(imaline)
    for j in range(h):
        ima[:, j] = imaline
    ima = np.transpose(ima)
    cv2.imwrite(str(phi + 1) + '_cos.jpg', ima)


def compensate_gamma(file):
    myslice = take_im_slice(file)
    cv2.imwrite("slice_file.png", myslice)
    print(myslice.shape)
    yarray = single_gamma_curve('slice_file.png')
    xarray = np.arange(0, 250, 10)
    yarray[23:] = 255
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
    gamma_cos(width, height, periods, -1, g)
    gamma_cos(width, height, periods, 0, g)
    gamma_cos(width, height, periods, 1, g)
    gamma_cos(width, height, 1, 5, g)
    gamma_cos(width, height, 1, 6, g)
    gamma_cos(width, height, 1, 7, g)

    return g_poly_comp


# file = '/home/samir/db2/scan/static/scan_folder/gamma_im_folder/image1.png'
# compensate_gamma(file)

# make_image(850, 480)
