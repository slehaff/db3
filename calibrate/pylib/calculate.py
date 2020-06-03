import numpy as np
import cv2
import glob
from scipy.spatial.transform import Rotation as R
import sys
import os
from PIL import Image
from pyntcloud import PyntCloud
# from jsoncloud import  generate_pointcloud

WIDTH = 170
HEIGHT = 170

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def calculate(folder):
    print('calculate call')
    f = open("output.txt", "a")
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(folder+'*/blendertexture.png', recursive= True)
    print('image count:',len(images))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # print('corners:', corners, file = f)
###########################################################################################
####################  Her we return the projector corners ################################
############## Read the corresponding phase file (unwrapped) convert phase(corner) to vp


        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            # print(objpoints, file= f)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # print('phi..:', getcornerabsphase(fname, corners))
            print(fname, file = f)
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)
            # print('imgpoints:', imgpoints)
            # print('objpoints:', objpoints)
    #Calibrate!
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('matrix:', mtx)
    
    for fname in images:
        getcornerswcoords(fname, rvecs, tvecs)
        # calctargetpoints(mtx,dist,imgpoints)
    print("dist:", dist)
    print('rvecs count:', len(rvecs))
    print('tvecs count:', len(tvecs))    
    print("rvecs:", rvecs)
    print("tvecs:", tvecs)
    # undistort(folder,mtx, dist, 170, 170)
    np.savez( folder + '/cal' + '.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    cv2.destroyAllWindows()
    return mtx, dist, rvecs, tvecs


def undistort(folder, mtx, dist, w, h ):
    print('undistort call')
    images = glob.glob(folder + '*/blendertexture.png', recursive= True)
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        print('dst:', dst.shape)
        cv2.imshow('dst', dst)
        # cv2.imwrite(folder + '/dst.png', dst)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return

def worldtargets(folder):
    print('worldtargets call')
    f = open(folder+"/worldtargets.txt", "a")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    mtx, dist, rvecs, tvecs = loadnpz(folder)
    mtx[0][0]=mtx[0][0] # Divide by image width
    mtx[1][1]=mtx[1][1] #Divide by image width
    print('length rvecs:', len(rvecs))
    print('mtx mm:', mtx)
    i = 0
    # objpoints = []  # 3d point in real world space
    # imgpoints = []  # 2d points in image plane.
    images = glob.glob(folder+'*/blendertexture.png', recursive= True)
    print('image count:',len(images))
    mresults=[[0,0,0,0,0,0,0]]
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            print(fname,i)
            # objpoints.append(objp)
            # print(objpoints, file= f)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # print('imagecorners:', corners)
            results =calctargetpoints(fname,mtx, dist, rvecs[i], tvecs[i], corners, f)
            mresults= np.vstack([mresults, results])
            i +=1
    print('length:', len(mresults))
    print(mresults, file=f)
    I=np.ones(len(mresults))
    I= np.transpose(I)
    m=np.linalg.lstsq(mresults,I, rcond= None)
    print('mmmmm:',m , file=f)
    m_file = folder + 'm_file.npy'
    np.save(m_file, m, allow_pickle=True)
    mm = np.load(m_file, allow_pickle=True)
    print('mm:', mm[0] )
    # threedcoords = np.zeros((WIDTH, HEIGHT,3))
    # folder ='/home/samir/Desktop/blender/pycode/calibrate/render0' 
    # threedcoords= getworldcoords(folder, mtx, m)
    # file_save = folder + '/worldcoords.npy'
    # np.save(file_save, threedcoords, allow_pickle=False)
    # generate_pointcloud(threedcoords, folder +'/worldcoords.ply' )

def inv(m):
    # print('inv call')
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]


def getworldcoords(folder, mtx, m):
    print('getworldcoords call')
    A = np.zeros((3,3))
    fname = folder + '/unwrap.png'
    # wrldcoords = np.zeros((WIDTH, HEIGHT,3))
    wrldcoords= np.zeros((WIDTH, HEIGHT), dtype=np.float)
    # xyinput = cv2.imread(fname)
    for i in range(WIDTH):
        # print('i:', i)
        for j in range(HEIGHT):
            A[0][0] = mtx[0][0]
            A[0][1]= 0
            A[0][2]= mtx[0][2]- j
            A[1][1] = mtx[1][1]
            A[1][2] = mtx[1][2]- j
            A[2][0] = m[0][4] - getphase(fname, i, j)*m[0][0]
            A[2][1] = m[0][5] - getphase(fname, i, j)*m[0][1]
            A[2][2] = m[0][6] - getphase(fname, i, j)*m[0][2]

            # A[0][0] = mtx[0][0]
            # A[0][1]= 0
            # A[0][2]= mtx[0][2]- j
            # A[1][1] = mtx[1][1]
            # A[1][2] = mtx[1][2]- j
            # A[2][0] = m[0][4] - getphase(fname, i, j)*m[0][0]
            # A[2][1] = m[0][5] - getphase(fname, i, j)*m[0][1]
            # A[2][2] = m[0][4] - getphase(fname, i, j)*m[0][2]

            invA = inv(A)

            b = np.transpose([0,0,(getphase(fname, i, j)*m[0][3]-1)])
            if (i == 80) :
                print('invA:', invA)
                print('b:',b)
            wrldcoords[i,j] = abs(np.transpose(np.matmul(invA, b))[2])


    return wrldcoords

def getphase(unwrapfile, x, y):
    # print('file:', file, x,y)
    # print('getphase call')
    # file= file[:-18]+'unwrap.png'
    # print('phase file:', file, x, y)
    img = cv2.imread(unwrapfile)
    phase = img[x,y]
    return(phase[0]/1)  #256)



def calctargetpoints(fname, mtx,dist, rvecsi, tvecsi,imgpoints, printfile):
    print('calculatetargetpoints call')
    # print('mtx call:', mtx)
    results = [[0,0,0,0,0,0,0]]
    #extend matrix col:
    # print(rvecsi, tvecsi)
    r = R.from_rotvec(np.transpose (rvecsi)[0])
    rotmat = r.as_matrix()
    # print('euler:', r.as_euler('zyx', degrees=True))
    # print('rotmat:', rotmat)
    extmat = np.append(rotmat, tvecsi, axis=1) # Specify checker pattern 1.8mm
    # print('extrmat:', extmat)
    totmat = np.matmul(mtx, extmat)
    # print('totmat:', totmat)
    row=[0,0,0,1]
    totmat = np.vstack([totmat, row])
    invtotmat= np.linalg.inv(totmat)
    # print('invtotmat:', invtotmat)
    print(len(imgpoints))
    for i in range(len(imgpoints)):
        c = np.array([[0],[0],[0],[1]])
        c[0]= imgpoints[i][0][0]
        c[1]= imgpoints[i][0][1]
        result=np.transpose(np.dot(invtotmat,c))
        phase = getphase(fname, int(imgpoints[i][0][0]), int(imgpoints[i][0][1]))
        # print('phase', phase)
        resultmult = phase* np.array(result)
        myequa=np.array([])
        myequa= np.append(resultmult[0], result[0], axis=0)
        myequa= myequa[:-1]
        # print('myequa:', myequa)
        results=np.vstack([results,myequa])
    # print(results)
    return(results)

def loadnpz(folder):
    print('loadnpz call')
    with np.load(folder + '/cal.npz') as X:
        mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        return mtx, dist, rvecs, tvecs

def getcornerabsphase(file, corners):
    print('getcornerabsphase call')
    file= file[:-10]+'unwrap.png'
    img = cv2.imread(file)
    # img = np.load(file)
    phicorners = []
    for i in range (len(corners)):
        phi =img[int(corners[i][0][0]), int(corners[i][0][1])][0]
        phicorners.append(phi)
    return phicorners


def getcornerswcoords(file, rvecs, tvecs):
    print('getcornerswcoords call')
    print(rvecs[1][0], rvecs[1][1], rvecs[1][2])
    return 

def pnp(objectPoints,imgPoints,w,h,f):
    print('pnp call')
    cameraMatrix = np.array([[f,0,w/2.0],
                     [0,f,h/2.0],
                    [0,0,1]])
    distCoeffs = np.zeros((5,1))
    _,rvecs, tvecs  =cv2.solvePnP(objectPoints[:,np.newaxis,:], imgPoints[:,np.newaxis,:], cameraMatrix, distCoeffs)#,False,flags=cv2.SOLVEPNP_EPNP)
    return rvecs,tvecs


def rot_params_rv(rvecs):
    print('rot_params_rv call')
    from math import pi,atan2,asin
    R = cv2.Rodrigues(rvecs)[0]
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= [roll,pitch,yaw]
    return rot_params


def generate_pointcloud(rgb_file, mask_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    # rgb = Image.open(rgb_file)
    # depth = Image.open(depth_file)
    depth = Image.open(depth_file).convert('I')
    mask = Image.open(mask_file).convert('I')

    # if rgb.size != depth.size:
    #     raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")


    points = []    
    for v in range(HEIGHT):
        for u in range(WIDTH):
            # color =   rgb.getpixel((u,v))
            # Z = depth.getpixel((u,v)) / scalingFactor
            # if Z==0: continue
            # X = (u - centerX) * Z / focalLength
            # Y = (v - centerY) * Z / focalLength
            if (mask.getpixel((u,v))<55):
                Z = depth.getpixel((u, v))*.22 
                if Z == 0: continue
                Y = .22 * v
                X = .22 * u
                points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,127,127,127))#color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()





# folder ='/home/samir/db3/calibrate/static/calibrate_folder/old_ones/' 
# file_load = folder + '/worldcoords.npy'
# threedcoords= np.load(file_load)
# generate_pointcloud(threedcoords, folder +'cal_im_folder20'+ '/worldcoords.ply' )
print('calculate.py')
folder = '/home/samir/Desktop/blender/pycode/scans/'
calfolder = '/home/samir/Desktop/blender/pycode/calibrate/'
# mtx, dist, _, _ =calculate(calfolder)
# undistort(calfolder, mtx, dist, 170, 170)
# worldtargets(calfolder)
mtx,_,_,_ = loadnpz(calfolder)
m = np.load(calfolder+'m_file.npy', allow_pickle=True)
print(mtx)
print(m[0])
coords = getworldcoords(folder+'render1',mtx,m)
print(coords[10,70])
cv2.imwrite(folder+'render1' + '/unwrap3.png', coords)
generate_pointcloud(folder+'render1/' + 'blendertexture.png', folder+'render1/' + '5mask.png', folder+'render1/' + 'unwrap3.png', folder+'render1/' +'pointcl-3.ply')



# def generate_pointcloud(worldcoords,ply_file):
#     print('generate_pointcloud call')
#     """
#     Generate a colored point cloud in PLY format from a color and a depth image.
    
#     Input:
#     rgb_file -- filename of color image
#     depth_file -- filename of depth image
#     ply_file -- filename of ply file
    
#     """
#     # rgb = Image.open(rgb_file)
#     # depth = Image.open(depth_file)
#     # depth = Image.open(depth_file).convert('I')

#     # if rgb.size != depth.size:
#     #     raise Exception("Color and depth image do not have the same resolution.")
#     # if rgb.mode != "RGB":
#     #     raise Exception("Color image is not in RGB format")
#     # if depth.mode != "I":
#     #     raise Exception("Depth image is not in intensity format")


#     points = []    
#     for v in range(WIDTH):
#         for u in range(HEIGHT):
#             # color = rgb.getpixel((u,v))
#             # Z = depth.getpixel((u,v)) / scalingFactorythonFiles/lib/python/debugpy/no_wheels/debugpy/launcher
#             X = u # worldcoords[u,v][0]
#             points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,120,120,120))
#     file = open(ply_file,"w")
#     file.write('''ply
# format ascii 1.0
# element vertex %d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# property uchar alpha
# end_header
# %s
# '''%(len(points),"".join(points)))
#     file.close()

