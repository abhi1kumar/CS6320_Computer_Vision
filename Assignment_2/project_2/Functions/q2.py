import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt




#[c,r,but] = ginput(10)
c = np.array([902.75, 1226.75, 1445.75, 1439.75, 1432.25, 1102.25, 886.25, 893.75, 1217.75, 1220.75])
r = np.array([170.75, 181.25, 185.75, 514.25, 947.75, 943.25, 938.75, 610.25, 617.75, 506.75])

#points = np.zeros((r.shape[0],3))
points = np.array([[7,5,0], [7,2,0], [7,0,0], [4,0,0], [0,0,0], [0,3,0], [0,5,0], [3,5,0], [3,2,0],[4,2,0]])

for i in range(r.shape[0]):
	img = cv2.imread('left2.jpg');
	c_x = int(c[i])
	c_y = int(r[i])
	print(points[i])
	cv2.circle(img, (c_x, c_y), radius=10, color=(0,0,255), thickness=10, lineType=8, shift=0)
	img = cv2.resize(img, (600, 300)) 
	cv2.imshow("image", img)
	
cv2.waitKey(0)
cv2.destroyAllWindows()
