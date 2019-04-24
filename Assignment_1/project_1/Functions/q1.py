"""
	Version 1 2018/01/21 Abhinav Kumar u1209853
"""

import argparse
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randrange


################################################################################
# Get distances between two points
################################################################################
def get_dist(a,b):
	return np.linalg.norm((a - b), ord=2)



################################################################################
# Fit line using two points
################################################################################
def fit_line(p,q):
	x1 = p[0]
	y1 = p[1]

	x2 = q[0]
	y2 = q[1]

	# (y1 – y2)x + (x2 – x1)y + (x1y2 – x2y1) = 0
	# From https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
	a = (y1 - y2)
	b = (x2 - x1)
	c = (x1*y2 - x2*y1)

	return np.array([[a],[b],[c]])



################################################################################
# Distances of all points from a line
################################################################################
def get_dist_from_line(pts, coeff):
	temp  = np.array([coeff[0][0], coeff[1][0]])
	denom = np.linalg.norm(temp, ord=2)
	dist  = np.absolute(np.dot (pts, coeff))
	dist  = dist/denom

	return dist



################################################################################
# Delete contents of edge_set which are present in inlier
################################################################################
def del_numpy_from_another(edge_set, inlier):
	ind_all = []

	for i in range(inlier.shape[0]):
		b = inlier[i]
		ind      = np.where(np.all(edge_set == b,axis=1))
		ind_all.append(ind)
	
	edge_set = np.delete(edge_set, (ind_all), axis=0)			

	return edge_set



################################################################################
# Main function
################################################################################
def main():
	# Argument Parsing
	parser = argparse.ArgumentParser(description='Detect Lines')

	parser.add_argument('-i', '--input_file', default= "Inputs/img1.png", help='input file/image path relative to the current directory')

	# Parsed values
	args = parser.parse_args()
	file_path_rel = args.input_file
	file_path     = os.path.join(os.getcwd(), file_path_rel)

	# Parameters
	output_folder      = os.path.join(os.getcwd(), 'Outputs')
	total_no_iters     = 10000
	max_pair_dist      = 100
	min_pt_line_dist   = 2
	min_line_pixel_num = 200

	print("\nInput Image = " + file_path)
	imgc = cv2.imread(file_path)
	imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)
	img  = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)

	h,w = img.shape

	# if image is too small, adjust the parameters
	if (w < 300 and h < 300):
		min_pt_line_dist   = 1
		min_line_pixel_num = 50

	edges = cv2.Canny (img, 50, 200)

	row, col      = np.nonzero(edges)
	edge_set      = np.transpose(np.vstack((row,col)))
	edge_set_ones = np.ones((edge_set.shape[0],3))
	edge_set_ones[:,:-1] = edge_set
	print("#Edge Points= " + str(edge_set.shape))

	line_set = []

	for i in range(total_no_iters):
		pt1_index = randrange(edge_set.shape[0])
	
		# Search for the second point
		while (True):	
			pt2_index = randrange(edge_set.shape[0])
			dist = get_dist(edge_set[pt1_index], edge_set[pt2_index])
			if (pt2_index != pt1_index and dist < max_pair_dist):
				break
	
		# Fit line
		coeff = fit_line(edge_set[pt1_index], edge_set[pt2_index])	
	
		# Find distances of all points from this line
		dist_all = get_dist_from_line(edge_set_ones, coeff)

		# Find all points within a specified distance from the line	
		ind, =  np.where(np.ravel(dist_all) < min_pt_line_dist)

		# if there are sufficient number of inlier pixels
		if (len(ind) > min_line_pixel_num):
			# Valid line
			# print("Valid line found")

			# Insert it in the line_set
			inlier = edge_set[ind]
			line_set.append(inlier)

			# Remove it from the edge_set
			edge_set = del_numpy_from_another(edge_set, inlier)
			edge_set_ones = np.ones((edge_set.shape[0],3))
			edge_set_ones[:,:-1] = edge_set


	print("#Valid lines= " + str(len(line_set)))
	color_edges = np.zeros((img.shape[0], img.shape[1], 3))

	# Now color the lines
	for i in range(len(line_set)):
		# Get a random color		
		new_color = np.random.choice(range(256), size=3)

		line = line_set[i]
		for j in range(line.shape[0]):
			color_edges[line[j][0], line[j][1]] = new_color

	# Plot and Save
	base             = os.path.basename(file_path)
	file_name_no_ext = os.path.splitext(base)[0]
	output_file = os.path.join(output_folder, file_name_no_ext + '_line.png')
	print("Output Image = " + output_file)
	
	plt.subplot(121), plt.imshow(imgc)
	plt.title('Original Image')  , plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(color_edges)
	plt.title('Line Color Image'), plt.xticks([]), plt.yticks([])
	plt.savefig(output_file, bbox_inches="tight")
	#plt.show()

if __name__ == '__main__':
    main()
