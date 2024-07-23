import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


CHECKERBOARD = (6,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []

imgpoints = [] 

reprojection_errors = [] 

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


images = glob.glob('./Images_Real/*.jpg')
counter = 0
for fname in images:
    h,w = img.shape[:2]
    # print(h,w)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # print(ret)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    imgpoints_reproj, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)


    error = cv2.norm(imgpoints[0], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
    reprojection_errors.append(error)

# cv2.destroyAllWindows()

h,w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

focal_length_x = mtx[0, 0]
focal_length_y = mtx[1, 1]
print("Focal length x :\n", focal_length_x)
print("Focal length y :\n", focal_length_y)
print()

skew_parameter = mtx[0, 1]
print("Skew :\n", skew_parameter)

principal_point_x = mtx[0, 2]
principal_point_y = mtx[1, 2]
print("Principal point x :\n",principal_point_x)
print("Principal point y :\n",principal_point_y)
print()

print("dist :\n")
print(dist)
print()


print("rvecs :\n")
print(rvecs)
print()

print("tvecs :\n")
print(tvecs)
print()


import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Define the number of inner corners of the chessboard
CHECKERBOARD = (6,8)  # Adjust according to your chessboard

# Criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
reprojection_errors = [] # To store re-projection errors

# Extracting path of individual image stored in a given directory
images = glob.glob('./Images_Real/*.jpg')
count = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

   
        h, w = img.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        imgpoints_reproj, _ = cv2.projectPoints(objp, rvecs[count], tvecs[count], mtx, dist)
        count += 1

        error = cv2.norm(imgpoints[0], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
        reprojection_errors.append(error)

   
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Detected Corners')
        axes[0].plot(corners2[:, 0, 0], corners2[:, 0, 1], 'ro')  

        axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Re-projected Corners')
        axes[1].plot(imgpoints_reproj[:, 0, 0], imgpoints_reproj[:, 0, 1], 'go') 

        plt.show()


plt.bar(range(len(reprojection_errors)), reprojection_errors)
plt.xlabel('Image Index')
plt.ylabel('Re-projection Error')
plt.title('Re-projection Error for Each Image')
plt.show()


mean_error = np.mean(reprojection_errors)
std_dev_error = np.std(reprojection_errors)
print("Mean Re-projection Error:", mean_error)
print("Standard Deviation of Re-projection Error:", std_dev_error)

print(reprojection_errors)



count = 1
for fname in images[:5]:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)
    
    plt.title(f"Undistorted Img {count}")
    plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    count += 1

plane_normals = []
for rvec in rvecs:
    rmat, _ = cv2.Rodrigues(rvec)
    normal = np.dot(rmat, np.array([0, 0, 1]))  
    plane_normals.append(normal)

for i, normal in enumerate(plane_normals):
    print(f"Image {i+1} - Checkerboard Plane Normal: {normal}")