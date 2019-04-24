"""
	Version 1 2018/01/22 Abhinav Kumar u1209853
"""

import argparse
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randrange



################################################################################
# Get Normalized Cross Correlation of two matrices
################################################################################
def NCC(A,B):
	eps = 1e-8 # Avoid division by zero (if pixels are all black)

	# Type conversion is essential since A and B are uint8. Taking dot product
	# causes to overflow. Convert first to float and then take the dot product 
	a = np.ndarray.flatten(A).astype(np.float64)
	b = np.ndarray.flatten(B).astype(np.float64)

	a_norm = np.linalg.norm(a, ord=2)
	b_norm = np.linalg.norm(b, ord=2)

	return np.dot(a,b)/((a_norm * b_norm) + eps)



################################################################################
# Main function
################################################################################
def main():
	# Argument Parsing
	parser = argparse.ArgumentParser(description='Calculates Disparity')

	parser.add_argument('-l', '--left_image_file' , default= "Inputs/left1.png",  help='left  image path relative to the current directory')
	parser.add_argument('-r', '--right_image_file', default= "Inputs/right1.png", help='right image path relative to the current directory')

	# Parsed values
	args = parser.parse_args()
	left_file_path_rel  = args.left_image_file
	right_file_path_rel = args.right_image_file
	left_file_path  = os.path.join(os.getcwd(), left_file_path_rel)
	right_file_path = os.path.join(os.getcwd(), right_file_path_rel)
	

	# Parameters
	output_folder   = os.path.join(os.getcwd(), 'Outputs')
	DISPARITY_RANGE = 50
	WIN_SIZE        = 5
	
	img1c = cv2.imread(left_file_path)
	img1c = cv2.cvtColor(img1c, cv2.COLOR_BGR2RGB)
	img1  = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
	
	img2c = cv2.imread(right_file_path)
	img2c = cv2.cvtColor(img2c, cv2.COLOR_BGR2RGB)
	img2  = cv2.cvtColor(img2c, cv2.COLOR_BGR2GRAY)


	print("Left  Input Image = " + left_file_path)
	print("Right Input Image = " + right_file_path)	
	print(img1.shape)
	
	"""
	# Inbuilt function
	stereo = cv2.StereoBM_create(blockSize=15)
	disparity = stereo.compute(img1, img2)	
	"""

	h,w = img1.shape

	# if image is too big, probably adjust the window size
	if (w > 1500 and h > 1000):
		WIN_SIZE        = 51
		
	EXTEND    = int((WIN_SIZE-1)/2) 
	disparity = np.zeros(img1.shape)

	
	for y in range(EXTEND, h-EXTEND):
		for x in range(EXTEND, w-EXTEND):
			best_disparity = 0
			best_NCC	   = 0
			patch1 = img1[y-EXTEND: y+EXTEND+1, x-EXTEND: x+EXTEND+1]

			for disp in range(1, DISPARITY_RANGE):
				if(x-disp-EXTEND >= 0 and x-disp+EXTEND < w):
					patch2 = img2[y-EXTEND: y+EXTEND+1, x-disp-EXTEND: x-disp+EXTEND+1]				
										
					curr_NCC = NCC(patch1, patch2)

					if (curr_NCC > best_NCC):
						best_NCC = curr_NCC
						best_disparity = disp
					
			disparity[y,x] = best_disparity
			
	base             = os.path.basename(left_file_path)
	file_name_no_ext = os.path.splitext(base)[0]
	output_file      = os.path.join(output_folder, file_name_no_ext[4:] + '_disparity.png')
	print("Output Disparity Map Path = " + output_file)
		
	# Plot and Save
	plt.subplot(131), plt.imshow(img1c)
	plt.title('Left Image') , plt.xticks([]), plt.yticks([])
	plt.subplot(132), plt.imshow(img2c)
	plt.title('Right Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(133), plt.imshow(disparity, cmap = 'gray')
	plt.title('Disparity')  , plt.xticks([]), plt.yticks([])
	plt.savefig(output_file, bbox_inches="tight")
	#plt.show()


if __name__ == '__main__':
    main()
