import os
import sys
import argparse

import numpy as np
import cv2 as cv
import glob

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--images_path", default="data/usbcam/", help="the path where images are stored")
parser.add_argument('--debug', help='Enable VSCode debugging', default=False, action='store_true')
parser.add_argument('--innerW', default=6, help="number of inner corners (width)")
parser.add_argument('--innerH', default=9, help="number of inner corners (height)")
parser.add_argument('--viz', default=False, action='store_true', help="whether to generate image distortions")
parser.add_argument('--outfile', default='calibration_params.txt')

args = parser.parse_args()

if __name__ == "__main__":

    if args.debug:
        # Ref: https://vinta.ws/code/remotely-debug-a-python-app-inside-a-docker-container-in-visual-studio-code.html
        import ptvsd
        print("Enabling attach starts.")
        ptvsd.enable_attach(address=('0.0.0.0', 8091))
        ptvsd.wait_for_attach()
        print("Enabling attach ends.")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((args.innerW*args.innerH,3), np.float32)
    objp[:,:2] = np.mgrid[0:args.innerH,0:args.innerW].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    all_paths = glob.glob(f"{args.images_path}/*.png")

    imgs = []
    imgs_corners = []
    imgs_undistorted = []
    
    for path in all_paths:
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (args.innerW,args.innerH), None)
        # make sure that the corners were found
        if not ret:
            print(path)
            continue
        imgs.append(img)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (args.innerW,args.innerH), corners2, ret)
        imgs_corners.append(img)

    # get the camera calibrations
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    params = extract_params(mtx, dist)
    write_to_file(params, args.outfile)

    imgs_undistorted = [undistort(img, mtx, dist) for img in imgs]

    corners = np.concatenate(imgs_corners[0:15], axis=0)
    undistorted = np.concatenate(imgs_undistorted[0:15], axis=0)

    cmb = np.concatenate((corners, undistorted), axis=1)
    cv.imwrite("viz.png", cmb)
