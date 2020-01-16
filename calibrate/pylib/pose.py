import glob
import numpy as np
import cv2
from calibrate.pylib.decompose import *


def draw(img, crnrs, imagpts):
    corner = tuple(crnrs[0].ravel())
    img = cv2.line(img, corner, tuple(imagpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imagpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imagpts[2].ravel()), (0, 0, 255), 5)
    return img


def pose(folder):
    # Load previously saved data
    with np.load(folder + '/cal.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        print('mtx from pose:', mtx)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    for fname in glob.glob(folder+'/*.png'):
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # f1.write(fname + '=' + '\n' + str(corners) + '\n\n')
            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
            # f1.write('rvecs:' + '=' + '\n' + str(rvecs) + '\n\n')
            # f1.write('tvecs:' + '=' + '\n' + str(tvecs) + '\n\n')
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            # build view matrix
            rotM = cv2.Rodrigues(rvecs)[0]
            myInv = cv2.invert(rotM) # For rotation matrix, transpose and inverse are identical
            myTrans = cv2.transpose(rotM)

            # Get camera rotation angles
            angles = rotationMatrixToEulerAngles(myTrans)

            camposition = -myTrans.dot(tvecs)
            camposition = camposition.tolist()

            newpose = {
                'campositionx': camposition[0],
                'campositiony': camposition[1],
                'campositionz': camposition[2],
                'anglex': angles[0],
                'angley': angles[1],
                'anglez': angles[2]
            }

            view_matrix = np.array([[rotM[0][0], rotM[0][1], rotM[0][2], tvecs[0]],
                                     [rotM[1][0], rotM[1][1], rotM[1][2], tvecs[1]],
                                     [rotM[2][0], rotM[2][1], rotM[2][2], tvecs[2]],
                                     [0.0, 0.0, 0.0, 1.0]])

            inverse_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
                                        [-1.0, -1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0, -1.0],
                                        [1.0, 1.0, 1.0, 1.0]])
            view_matrix *= inverse_matrix
            view_matrix = np.transpose(view_matrix)
            rotM = view_matrix

            image = draw(image, corners2, imgpts)
            cv2.imshow('image', image)
            k = cv2.waitKey(0) & 0xff
            if k == 's':
                cv2.imwrite(fname[:6]+'.png', image)
    # f1.close()
    cv2.destroyAllWindows()
    return newpose