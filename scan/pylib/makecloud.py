import numpy as np
import cv2
import argparse
import sys
import os
from PIL import Image
from jsoncloud import generate_json_pointcloud, generate_pointcloud


Pro = np.array([0,60,85]) # projector origin
ProI=np.array([0, 76.2, 24.35]) # Porjector origin projection
Cam = np.array([0,45, 20]) # camera origin
COr = np.array([0, 60, -35])
CamPro = np.array([15, 65])
nR = np.array([0,0,1])
Cmo = np.array([0, 45.082, 19.578])
nCam = np.array([0, 0.2588, 0.9659])
ProCam = np.array([0, 15, 65])
Cos15 = np.cos(15/360*2*np.pi)
Sin15 = np.sin(15/360*2*np.pi)
Pix = .001 # Pixel size
PhiMin = 30
PhiMax = 255
RefLength = 21.782
YrefStart = 48.846
deltaref = .2138

rwidth = 170
rheight = 170




def getcmi(x,y):
    cmi = np.array([0.,0.,0.])    
    cmi[0] = Cmo[0]+(x-85)*Pix
    cmi[1] = Cmo[1]+ (y-85)* Cos15*Pix
    cmi[2] = Cmo[2]+ (y-85)* Sin15*Pix
    return(cmi)

def makecmitable():
    cmi = [0.,0.,0.]
    cmitab = np.arange(170.0*170*3).reshape(170,170,3)
    print('shape', cmitab.shape)
    for x in range(170):
        for y in range(170):
            cmi[0] = Cmo[0]+(x-85)*Pix
            cmi[1] = Cmo[1]+ (y-85)* Cos15*Pix
            cmi[2] = Cmo[2]+ (y-85)* Sin15*Pix 
            cmitab[x,y] = cmi
        print('cmi:', cmitab[x,50])

    return cmitab         


def getmref(cmi):
    mref = np.array([0,0,0])
    mref = Cam + np.dot(nR, (Cam-COr))/np.dot(nR, (cmi-Cam))*(cmi-Cam)
    return(mref)


# def getnref(cni):
#     return(nref)


def getrefline(phi):
    r = (phi - PhiMin)/(PhiMax-PhiMin)
    rdash = r/Cos15
    y = rdash*RefLength + YrefStart
    refLinePoint = [0, y, -35]
    return(refLinePoint)


# def getq(cmi):
#     return(q)


def calcm3dpoint( mref, nref,cmi):
    distance = np.zeros((170,1))

    for i in range(0,100):
        normal = np.cross((Pro-[(i-50)*deltaref,nref[1],-35]),(mref-Cam))
        mag = np.sqrt(normal.dot(normal))
        normalized = normal/mag
        # print('norm', normalized)
        distance[i]= abs(np.dot(ProCam, normalized))
        # print(i, distance[j])

        # distance = abs(np.dot(ProCam, normalized))
        if distance[i] < 1 :
            print('found:', i, distance[i], distance[i-20])
            break
    print('min, max:', np.min(distance), np.max(distance))
    # print(distance)

    return()

def getm3dpoint(x,y,phi):
    cmi = getcmi(x,y)
    mref = getmref(cmi)
    nreflinepoint = getrefline(phi)
    print(cmi, mref, nreflinepoint)
    m3dpoint = calcm3dpoint(mref, nreflinepoint, cmi)
    return(m3dpoint)


def threedpoints(unwrapfile):
    # unwrap = np.zeros((rheight, rwidth), dtype=np.float)
    # reference = np.zeros((rheight, rwidth), dtype=np.float)
    thdpoints =  np.zeros((rheight, rwidth), dtype=np.float) 
    unwrap = np.load(unwrapfile)
    # reference = np.load(referencefile)
    for j in range(rwidth):
        for i in range(rheight):
            getm3dpoint(i,j,unwrap[i,j])
            # if wrap[i, j] < 0:
            #     if greynom[i, j] < 0:
            #         wrap[i, j] += 2*np.pi
            #     else:
            #         wrap[i, j] += 1 * np.pi
            # im_wrap[i, j] = 128/np.pi * wrap[i, j]
            print(i,j)

# makecmitable()
unwfile = '/home/samir/Desktop/blender/pycode/scans/render'+ str(0)+'/unwrap.npy'
ref_unwfile ='/home/samir/Desktop/blender/pycode/reference/scan_ref_folder/unwrap.npy'
threedpoints(unwfile) 
