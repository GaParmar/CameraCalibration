import cv2 as cv

def extract_params(cameraMatrix, distCoeffs):
    return {
        "fx" : cameraMatrix[0,0],
        "fy" : cameraMatrix[1,1],
        "cx" : cameraMatrix[0,2],
        "cy" : cameraMatrix[1,2],
        "k1" : distCoeffs[0,0],
        "k2" : distCoeffs[0,1],
        "p1" : distCoeffs[0,2],
        "p2" : distCoeffs[0,3],
        "p3" : distCoeffs[0,4] 
    }

def write_to_file(params, fname):
    with open(fname, "w") as f:
        for key in params:
            f.write(f"{key}: {params[key]}\n")

def undistort(img, mtx, dist):
    h,w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    mapx,mapy = cv.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv.remap(img,mapx,mapy,cv.INTER_LINEAR)

    x,y,w,h = roi
    # dst = dst[y:y+h, x:x+h]
    # resized = cv.resize(dst, (img.shape[:2][::-1]))
    return dst
    