import numpy as np
import cv2
import glob


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def calculate(folder):
    f = open("output.txt", "a")
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(folder+'*/image1.png', recursive= True)
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
            print(objpoints, file= f)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # print('phi..:', getcornerabsphase(fname, corners))
            print(fname, file = f)
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)
            print('imgpoints:', imgpoints)
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
    # undistort(folder,mtx, dist, 400, 400)
    np.savez( folder + '/cal' + '.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    cv2.destroyAllWindows()
    return mtx, dist, rvecs, tvecs


def undistort(folder, mtx, dist, w, h ):
    images = glob.glob(folder + '*/image1.png', recursive= True)
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
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return

def calctargetpoints(mtx,dist,imgpoints):
    results = [[0,0,0]]
    #extend matrix col:
    b=np.zeros((3,1))
    mtx= np.reshape(mtx,(3,3))
    print('mtx:', mtx)
    a = np.append(mtx,b, axis=1)
    print(a)
    for i in range(len(imgpoints)):
        c = [[0],[0],[0],[1]]
        c[0]= imgpoints[i][0]
        c[1]= imgpoints[i][1]
        print('c:', imgpoints[i][1][0][0])
        results[i]=np.matmul(a,c)
    print('3ds:', results)
    return(results)



def getcornerabsphase(file, corners):
    file= file[:-10]+'unwrap.png'
    img = cv2.imread(file)
    # img = np.load(file)
    phicorners = []
    for i in range (len(corners)):
        phi =img[int(corners[i][0][0]), int(corners[i][0][1])][0]
        phicorners.append(phi)
    return phicorners


def getcornerswcoords(file, rvecs, tvecs):
    print(rvecs[1][0], rvecs[1][1], rvecs[1][2])
    return 

def pnp(objectPoints,imgPoints,w,h,f):
    cameraMatrix = np.array([[f,0,w/2.0],
                     [0,f,h/2.0],
                    [0,0,1]])
    distCoeffs = np.zeros((5,1))
    revtval,rvecs, tvecs  =cv2.solvePnP(objectPoints[:,np.newaxis,:], imgPoints[:,np.newaxis,:], cameraMatrix, distCoeffs)#,False,flags=cv2.SOLVEPNP_EPNP)
    return rvecs,tvecs


def rot_params_rv(rvecs):
    from math import pi,atan2,asin
    R = cv2.Rodrigues(rvecs)[0]
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= [roll,pitch,yaw]
    return rot_params