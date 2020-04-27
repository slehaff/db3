import numpy as np
import cv2
import argparse
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
PhiMin = 30 # Ref Min
PhiMax = 255
RefLength = 21.782 # Ref Max
YrefStart = 48.846
deltaref = .2138

rwidth = 170
rheight = 170




def getcmi(x,y):
    cmi = np.array([0.,0.,0.])    
    cmi[0] = (Cmo[0]+(x-85)*Pix)
    cmi[1] = (Cmo[1]-45+ (y-85)* Cos15*Pix)
    cmi[2] = (Cmo[2]-20+ (y-85)* Sin15*Pix)
    return(cmi)

# def getcni(x,y):
#     cni = np.array([0.,0.,0.])    
#     cni[0] = Cmo[0]+(x-85)*Pix
#     cni[1] = Cmo[1]+ (y-85)* Cos15*Pix
#     cni[2] = Cmo[2]+ (y-85)* Sin15*Pix
#     return(cni)

def makecmitable():
    cmi = [0.,0.,0.]
    cmitab = np.arange(170.0*170*3).reshape(170,170,3)
    print('shape', cmitab.shape)
    for x in range(170):
        for y in range(170):cmi = np.array([0.,0.,0.])
    return cmitab         


def getmref(cmi):
    mref = np.array([0,0,0])
    # mref = Cam + np.dot(nR, (Cam-COr))/np.dot(nR, (Cam-cmi))*(Cam-cmi)
    mref = Cam + 55*cmi/np.linalg.norm(cmi)
    print(np.linalg.norm(cmi))
    return(mref)


def getcni(cmi,phi):
    cni = np.array([0.,0.,0.])
    l = np.array([0.,0.,0.])
    l= (ProI- cmi)/np.linalg.norm(ProI-cmi)
    deltaphi = (PhiMax-PhiMin)/170
    cni[0] = phi/deltaphi #pixels
    cni[1] = cni[0]/l[1]*l[0]


    cni = 0
    return(cni)



def getnref(cni):
    nref = np.array([0,0,0])
    nref = Cam + np.dot(nR, (Cam-COr))/np.dot(nR, (cni-Cam))*(cni-Cam)
    return(nref)

# cmi = np.array([0.,0.,0.])
#     y = rdash*RefLength + YrefStart
#     refLinePoint = [0, y, -35]
#     return(refLinePoint)


def getq(cmi):
    q= Cam + np.dot(ProCam, nR)/np.dot(cmi,nR)*cmi
    return(q)


def calcm3dpoint( mref, nref,cmi):
    distance = np.zeros((170,1))

    for i in range(0,100):
        normal = np.cross((Pro-[(i-50)*deltaref,nref[1],-35]),(mref-Cam))
        mag = np.sqrt(normal.dot(normal))
        normalized = normal/mag
        # print('norm', normalized)
        distance[i]= abs(np.dot(ProCam, normalized))
        if distance[i] ==0 :
            # print('found:', i, distance[i], distance[icmi = np.array([0.,0.,0.])-1])
            break
    # # print('min, max:', np.min(distance), np.max(distance))
    # print(distance)

    return()

def getm3dpoint(x,y,phi):
    cmi = getcmi(x,y)
    mref = getmref(cmi)
    nreflinepoint = getrefline(phi)
    # print(cmi, mref, nreflinepoint)
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
            print(getm3dpoint(i,j,unwrap[i,j]))

        print('j:', j)


def test3dpoints(unwrapfile):
    # Read 4 sample points:
    unwrap = np.load(unwrapfile)
    print('50,50, unwrap(50,50):', unwrap[50,50])
    print('70,50, unwrap(70,50):', unwrap[70,50])
    print('50,70, unwrap(50,70):', unwrap[50,70])
    print('100,100, unwrap(100,100):', unwrap[100,100])
    print('150,150, unwrap(150,150):', unwrap[150,150])
    print('150,100, unwrap(150,100):', unwrap[150,100])
    plist = [[50,50],[70,50],[50,70],[100,100],[150,150],[150,100]]
    cmilist =[]
    x= []
    y= []
    z= []
    Cma = [0,45,20]
    print(len(plist))
    for i in range(len(plist)):
        cmi = getcmi(plist[i][0], plist[i][1])
        print(plist[i][0], plist[i][1], cmi)
        cmilist.append(cmi)
        x.append( cmi[0])
        y.append(cmi[1])
        z.append(cmi[2])
        mref = getmref(cmi)
        print('mref:', mref)
        # x.append( mref[0])
        # y.append(mref[1])
        # z.append(mref[2])
        # plt.plot(Cma, mref)

    print(len(x))
    figure = plt.figure()
    ax = figure.add_subplot(111, projection = '3d')
    ax.scatter(x,y,z, c = 'r', marker = 'o')
    x2 = [5,-5,0,0,0,0,0]
    y2 = [0,0,0,45,60,60,45.082]
    z2 = [0,0,0,20,85,85,19.578]
    # ax.scatter(x2,y2,z2, c = 'g', marker = '+')
    ax.set_xlabel('Xaxis')
    ax.set_ylabel('Yaxis')
    ax.set_zlabel('Zaxis')
    plt.show()

# makecmitable()
unwfile = '/home/samir/Desktop/blender/pycode/scanplanes/render'+ str(1)+'/unwrap.npy'
ref_unwfile ='/home/samir/Desktop/blender/pycode/reference/scan_ref_folder/unwrap.npy'
test3dpoints(unwfile)