import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import time

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

cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
lastIDPosition = {}

# goal: calculate position of viewer
position = np.array([0, 0, 0])

while(True):
    # capture frames
    start_time = time.time() # start time of the loop

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if corners:
        velocities = np.zeros((len(ids), 3))

        pose = aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist, rvecs, tvecs)

        pose_rvecs = pose[0]
        pose_tvecs = pose[1]
        gray = aruco.drawDetectedMarkers(gray, corners)

        for i in range(len(ids)):
            # convert to rotation matrix
            currentRotation, _ = cv2.Rodrigues(pose_rvecs[i])
            # transpose and rotate coordinates to match the wall
            currentTranslation = np.transpose(currentRotation.dot(np.transpose(pose_tvecs[i])))

            aruco.drawAxis(gray, mtx, dist, pose_rvecs[i], currentTranslation, 0.1)

            currentID = ids[i][0]
            currentTime = time.time()

            try:
                (oldPosition, oldTime) = lastIDPosition[str(currentID)]

                # filter out old entries, likely from last time the target crossed the camera
                # number should be tuned more finely
                if (currentTime - oldTime) > 5:
                    lastIDPosition[str(currentID)] = (currentTranslation, currentTime)
                    continue

                # calculate velocity based on old position
                velocity = (currentTranslation - oldPosition)/(currentTime - oldTime)
                velocities[i] = velocity

                lastIDPosition[str(currentID)] = (currentTranslation, currentTime)
            except KeyError:
                # first time seeing this target. remember for next time
                lastIDPosition[str(currentID)] = (currentTranslation, currentTime)
                continue

        # integration...kinda
        avgVelocity = np.average(velocities, axis=0) * 10
        position = position + avgVelocity
        print(position)

    # fix: creates a ton of windows
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop


cap.release()
cv2.destroyAllWindows()
