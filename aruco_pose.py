import numpy as np
import cv2
import cv2.aruco as aruco
import glob

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
dictionary = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_board = aruco.CharucoBoard_create(5,7,.025,.0125, aruco_dict)
# termination critera
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = glob.glob('*.jpg')

for fname in images:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  size = gray.shape
  res = aruco.detectMarkers(gray, aruco_dict)

  allCharucoCorners = []
  allCharucoIds = [];

  if len(res[0]) > 0:
    res2 = aruco.interpolateCornersCharuco(res[0], res[1], gray, aruco_board)
    if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
      allCharucoCorners.append(res2[1])
      allCharucoIds.append(res2[2])

    aruco.drawDetectedMarkers(gray, res[0], res[1])
  
  cv2.imshow('frame', gray)
  cv2.waitKey(0)


cal = aruco.calibrateCameraCharuco(allCharucoCorners, allCharucoIds, aruco_board, size, None, None)
mtx = cal[1]
dist = cal[2]
rvecs = cal[3][0]
tvecs = cal[4][0]
print cal

cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
while(True):
  # capture frames
  ret, frame = cap.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  parameters = aruco.DetectorParameters_create()

  corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)


  if corners:
    # print "id" + str(ids)
    pose = aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist, rvecs, tvecs)
    # print pose
    pose_rvecs = pose[0]
    pose_tvecs = pose[1]
    gray = aruco.drawDetectedMarkers(gray, corners)
    for i in range(len(ids)):
      aruco.drawAxis(gray, mtx, dist, pose_rvecs[i], pose_tvecs[i], 0.1)

  # fix: creates a ton of windows
  cv2.imshow('frame', gray)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break



cap.release()
cv2.destroyAllWindows()



