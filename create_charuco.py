import cv2
import cv2.aruco as aruco
import glob

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_board = aruco.CharucoBoard_create(5,7,.025,.0125, aruco_dict)
img = aruco_board.draw((600, 800))

cv2.imwrite('charuco.png', img)