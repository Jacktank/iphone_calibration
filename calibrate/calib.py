# coding:utf-8

import numpy as np
import cv2
import glob
import os
'''
相机标定脚本
'''
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)

row_no = 9
col_no = 6

#这里用的图7行*9列
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((row_no*col_no,3), np.float32)
objp[:,:2] = np.mgrid[0:row_no,0:col_no].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

image_root = './parse/board_image/image_0/'
images = os.listdir(image_root)

for fname in images:
    print(os.path.join(image_root, fname))
    img = cv2.imread(os.path.join(image_root, fname))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (row_no,col_no),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (row_no,col_no), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(30)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx)
print(dist)
np.savetxt('./matrix.txt',mtx)
np.savetxt('./dist.txt',dist)


h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
for fname in images:
    img = cv2.imread(os.path.join(image_root, fname))
    # 通过调用函数，传递ROI参数就可以复制结果。
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    # 首先找到原图片与校正图片之间映射函数。然后使用重映射函数。
    # undistort
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    #
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('calibresult.png', dst)