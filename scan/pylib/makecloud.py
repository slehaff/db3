import numpy as np
import cv2
import argparse
import sys
import os
from PIL import Image
from jsoncloud import generate_json_pointcloud, generate_pointcloud


Pro = [0,60,85] # projector origin
ProI=[0, 76.2, 24.35] # Porjector origin projection
Cam = [0,45, 20] # camera origin
COr = [15, -55]
CamPro = [15, 65]
nR = [0,0,1]
Cmo = [0, 45.082, 19.578]
nCam = [0, 0.2588, 0.9659]
Cos15 = np.cos(15/360*2*np.pi)
Sin15 = np.sin(15/360*2*np.pi)
Pix = .001 # Pixel size

def getcmicni(x,y):
    cmi = []
    cni = []
    
    cmi[0] = Cmo[0]+(x-85)*Pix
    cmi[1] = Cmo[1]+ (y-85)* Cos15*Pix
    cmi[2] = Cmo[2]+ (y-85)* Sin15*Pix


    return(cmi, cni)

def makecmitable():
    cmi = []
    cmitab = []
    for x in range(170):
        for y inrange(170):
            cmi[0] = Cmo[0]+(x-85)*Pix
            cmi[1] = Cmo[1]+ (y-85)* Cos15*Pix
            cmi[2] = Cmo[2]+ (y-85)* Sin15*Pix 
            cmitab[x,y] = cmi  

    return cmitab         


def getmref(cmi):
    return(mref)


def getnref(cni):
    return(nref)


def getq(cmi):
    return(q)


def calcm3dpoint(q, mref, nref,cmi):
    return(m3dpoint)

def getm3dpoint(x,y,phi):
    cmi, cni = getcmicni(x,y)
    mref = getmref(cmi)
    nref = getnref(cni)
    q = getq(cmi)
    m3dpoint = calcm3dpoint(q, mref, nref, cmi)
    return(m3dpoint)


makecmitable()