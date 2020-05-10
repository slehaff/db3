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
ProI=np.array([0, 75.35837215, 27.68177481]) # Porjector origin projection
Cam = np.array([0,0,0]) # camera origin
COr = np.array([0, 15, -55])
CamPro = np.array([15, 65])
nR = np.array([0,0,1])
Cmo = np.array([0, 45.11318, 19.5776])
cmo = np.array([0,  .11318, -.42239]) # camera image plane with cam = origo
nCam = np.array([0, 0.2588, 0.9659])
ProCam = np.array([0, 15, 65])
Cos11 = np.cos(11/360*2*np.pi)
Sin11 = np.sin(11/360*2*np.pi)
Cos15 = np.cos(15/360*2*np.pi)
Sin15 = np.sin(15/360*2*np.pi)
Cos75 = np.cos(75/360*2*np.pi)
Sin75 = np.sin(75/360*2*np.pi)
Pix = .001 # Pixel size
# PhiMin = 30 # Ref Min
# PhiMax = 255
RefLength = 21.782 # Ref Max
YrefStart = 48.846
deltaref = .2138

rwidth = 170
rheight = 170



def pointransform(point):
    point[0]= point[1]
    point[1]= point[2]
    point[2] = 1
    R = np.array([[Cos75,-Sin75,0],[Sin75,Cos75,0],[0,0,1]])
    T = np.array([[1,0,-45],[0,1,-20],[0,0,1]])
    RT = np.matmul(R,T)
    transpoint = np.matmul(RT,point)
    return(transpoint)

def vectransform(vector):
    vector[0]= vector[1]
    vector[1]= vector[2]
    vector[2] = 0
    R = np.array([[Cos75,-Sin75,0],[Sin75,Cos75,0],[0,0,1]])
    T = np.array([[1,0,-45],[0,1,-20],[0,0,1]])
    RT = np.matmul(R,T)
    transvect = np.matmul(RT,vector)
    return(transvect)



def getcmi(x,y):
    cmi = np.array([0.,0.,0.])    
    cmi[0] = ((x-85)*Pix)+ cmo[0]
    cmi[1] = (((y-85)* Cos15*Pix)+ cmo[1])
    cmi[2] = (((y-85)* Sin15*Pix)+ cmo[2])
    # cmi = np.add(Cam,cmi)
    print('cmi:', cmi-cmo)
    return(cmi)


def getmref(cmi):
    mref = np.array([0,0,0])
    mref =  np.multiply(-55/np.dot(cmi,nR), cmi)
    mref = np.add(mref, Cam)
    # print('mref:' ,mref)
    return(mref)


def getProI():
    ProT = pointransform(Pro)
    CmoT = pointransform(Cmo)
    vecA = Pro-Cmo
    vecB = np.array([0, Cos15, Sin15])
    ProI = Cmo + np.dot(vecA, vecB)/(np.linalg.norm(vecB))*vecB
    # print('ProI:', ProI)
    
    return(ProI)


def getcni(cmi,phi, phimax, phimin):
    cni = np.array([0.,0.,0.])
    l = np.array([0.,0.,0.])
    Cmi = cmi+Cam
    l= (ProI- Cmi)/np.linalg.norm(ProI-Cmi)
    # print('lll:', l)
    deltaphi = (phimin-phimax)/170
    # print('deltaphi:', deltaphi)
    # cni[1] = Pix*phi/deltaphi*Cos15  #+cmo[1]#-.085*Cos15 #pixels
    # cni[1] = Pix*170*phi/phimax*Cos15
    cni[1] = Pix*170*phi/1.1*Cos15
    # print('cni[1]:', cni[1])
    t = (cni[1]-cmi[1])/l[1]
    # print('Cmi:',Cmi)
    cni = cmi +t*l
    # print('cni:', cni, 'cmi:', cmi, 't:', t)
    # print('l:', l)
    # cni = cnvcni(cni[1], cni[0])
    # print( 'cmi:', cmi, 'cni:', cni)
    return(cni)

def cnvcni(x,y):
    cni = np.array([0.,0.,0.])    
    cni[0] = ((x-85)*Pix)+ cmo[0]
    cni[1] = (((y-85)* Cos15*Pix)+ cmo[1])
    cni[2] = (((y-85)* Sin15*Pix)+ cmo[2])
    # cmi = np.add(Cam,cmi)
    return(cni)



def getnref(cni):
    nref = np.array([0,0,0])
    nref =  np.multiply(-55/np.dot(cni,nR), cni)
    nref = np.add(nref, Cam)
    return(nref)



def getq(cmi):
    q= Cam + np.dot(ProCam, nR)/np.dot(cmi,nR)*cmi
    return(q)


def calcm3dpoint(que ,mref, nref,Cmi):
    PQ = que-Pro
    QMref = mref-que
    MNref = mref-nref
    nom = np.linalg.norm(PQ)*np.linalg.norm(QMref)
    denom = (np.linalg.norm(PQ+MNref))
    m3d=que + np.linalg.norm(PQ)*np.linalg.norm(QMref)/(np.linalg.norm(PQ+MNref))*(Cmi-Cam)/np.linalg.norm(Cmi-Cam)
    # print('PQ:',PQ, 'QMref:', QMref, 'MNref:', MNref)
    # print('3dinfo:',nom,denom,np.linalg.norm(Cmi-Cam))
    return(m3d)

# def getm3dpoint(x,y,phi):
#     cmi = getcmi(x,y)
#     mref = getmref(cmi)
#     nreflinepoint = getrefline(phi)
#     # print(cmi, mref, nreflinepoint)
#     m3dpoint = calcm3dpoint(mref, nreflinepoint, cmi)
#     return(m3dpoint)


# def threedpoints(unwrapfile):
#     # unwrap = np.zeros((rheight, rwidth), dtype=np.float)
#     # reference = np.zeros((rheight, rwidth), dtype=np.float)
#     thdpoints =  np.zeros((rheight, rwidth), dtype=np.float) 
#     unwrap = np.load(unwrapfile)
#     # reference = np.load(referencefile)
#     for j in range(rwidth):
#         for i in range(rheight):
#             print(getm3dpoint(i,j,unwrap[i,j]))

#         print('j:', j)

def getxref(phi,reference):
    for i in range(170):
        while (phi -reference[i, 85])< .05:
            i+=1

    return(i)

def test3dpoints(unwrapfile, ref_unwrapfile):
    # Read 4 sample points:
    unwrap = np.load(unwrapfile)
    refunwrap = np.load(ref_unwrapfile)
    ProI = getProI()

    # print('150,100, unwrap(150,100):', unwrap[150,100])
    plist = [[85,0], [85,85],[85,169]]#,[85,169],[169,169],[0,169],[0,0],[0,169],[169,0],[169,169],[20,20],[50,50],[70,50],[50,70],[100,100],[100,150],[150,150],[150,100]]
    cmilist =[]
    x= []
    y= []
    z= []
    x1= []
    y1= []
    z1= []
    mx3= []
    my3= []
    mz3= []
    nx4= []
    ny4= []
    nz4= []
    x5= [cmo[0]]
    y5= [cmo[1]]
    z5= [cmo[2]]
    Cma = [0,45,20]
    # print(len(plist))
    for i in range(len(plist)):
        cmi = getcmi(plist[i][0], plist[i][1])
        # cmi = cmi + Cam
        # print('cmi:', cmi)
        PhiMin = refunwrap[0,85]
        PhiMax = refunwrap[169,85]
        # xref= getxref(unwrap[plist[i][1], plist[i][0]], refunwrap)
        # print('phi:', unwrap[plist[i][1], plist[i][0]],  'xref:', xref)
        # print('minmax:', PhiMax, PhiMin)
        cni = getcni(cmi, unwrap[plist[i][1], plist[i][0]], PhiMax, PhiMin)
        # print(plist[i][0], plist[i][1], cmi, np.dot(cmi,nR))
        cmilist.append(cmi)
        mx3.append( cmi[0])
        my3.append(cmi[1])
        mz3.append(cmi[2])
        nx4.append( cni[0])
        ny4.append(cni[1])
        nz4.append(cni[2])
        mref = getmref(cmi)
        print('mref:', mref)
        x.append( mref[0])
        y.append(mref[1])
        z.append(mref[2])
        nref = getnref(cni)
        # print('nref:', nref)
        x1.append( nref[0])
        y1.append(nref[1])
        z1.append(nref[2])
        que= getq(cmi)
        x1.append( que[0])
        y1.append(que[1])
        z1.append(que[2])
        print(plist[i][1], plist[i][0])
        print('cni:', cni)
        print('cmi:', cmi)
        Cmi = cmi + Cam
        M3d= calcm3dpoint(que, mref, nref, Cmi)
        # print('M3d:', M3d)
        x5.append( M3d[0])
        y5.append(M3d[1])
        z5.append(M3d[2])
    # l2= (cmilist[2]- cmilist[1])/np.linalg.norm(cmilist[2]- cmilist[1])
    # print('l2:', l2)
    # print(len(x))
    figure = plt.figure()
    ax = figure.add_subplot(111, projection = '3d')
    ax.scatter(x,y,z, c = 'r', marker = '.')
    # ax.scatter(x1,y1,z1, c = 'b', marker = '.')
    # ax.scatter(mx3,my3,mz3, c = 'r', marker = '.')
    # ax.scatter(nx4,ny4,nz4, c = 'b', marker = '.')
    # ax.scatter(x5,y5,z5, c = 'r', marker = '+')
    # x2 = [35,-35,0,0,0, ProI[0]]
    # y2 = [0,0,0,45,60,ProI[1]]
    # z2 = [0,0,0,20,85, ProI[2]]
    # ax.scatter(x2,y2,z2, c = 'g', marker = 'x')
    ax.set_xlabel('Xaxis')
    ax.set_ylabel('Yaxis')
    ax.set_zlabel('Zaxis')
    # plt.show()
# def getcni(x,y):
#     cni = np.array([0.,0.,0.])    
#     cni[0] = Cmo[0]+(x-85)*Pix
#     cni[1] = Cmo[1]+ (y-85)* Cos15*Pix
#     cni[2] = Cmo[2]+ (y-85)* Sin15*Pix
#     return(cni)

def makereference(ref_file):
    reference = np.zeros((rwidth, rheight), dtype=np.float)
    for i in range(170):
        reference[i,:]= .9*i/170
    np.save(ref_unwfile, reference, allow_pickle=False)



# makecmitable()
unwfile = '/home/samir/Desktop/blender/pycode/scanplanes/render'+ str(1)+'/unwrap.npy'
ref_unwfile ='/home/samir/Desktop/blender/pycode/reference/scan_ref_folder/unwrap.npy'
test3dpoints(unwfile, ref_unwfile)
# makereference(ref_unwfile)





####################################################################################################
# def makecmitable():
#     cmi = [0.,0.,0.]
#     cmitab = np.arange(170.0*170*3).reshape(170,170,3)
#     print('shape', cmitab.shape)
#     for x in range(170):
#         for y in range(170):cmi = np.array([0.,0.,0.])
#     return cmitab         
