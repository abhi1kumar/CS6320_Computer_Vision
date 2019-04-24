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
# Main function
################################################################################
def main():
	# Argument Parsing
	parser = argparse.ArgumentParser(description='Detect Sky')

	parser.add_argument('-i', '--input_file', default= "Inputs/detectSky1.bmp", help='input file/image path relative to the current directory')
	
	# Parsed values
	args = parser.parse_args()
	file_path_rel = args.input_file
	file_path     = os.path.join(os.getcwd(), file_path_rel)
	
	# Parameters
	output_folder = os.path.join(os.getcwd(), 'Outputs')
	R_MIN         = 0
	R_MAX         = 100
	G_MIN         = 1
	G_MAX         = 150
	B_MIN         = 100
	B_MAX         = 255

	print("\nInput Image = " + file_path)
	img  = cv2.imread(file_path)
	img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	h,w,c = img.shape
	sky = np.zeros(img.shape, dtype=np.uint8)

	for i in range(h):
		for j in range(w):
			r = img[i,j,0]
			g = img[i,j,1]
			b = img[i,j,2]

			if ( r > R_MIN and r < R_MAX and g > G_MIN and g < G_MAX and b > B_MIN and b < B_MAX):
				#print("pixel found")			
				sky[i,j,0] = 255
				sky[i,j,1] = 255
				sky[i,j,2] = 255


	# Plot and Save		
	base             = os.path.basename(file_path)
	file_name_no_ext = os.path.splitext(base)[0]
	output_file      = os.path.join(output_folder, file_name_no_ext + '_sky.png')
	print("Output Image = " + output_file)
	
	plt.subplot(121), plt.imshow(img)
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(sky)
	plt.title('Segmented Sky') , plt.xticks([]), plt.yticks([])
	plt.savefig(output_file, bbox_inches="tight")

if __name__ == '__main__':
    main()
