import os
import numpy
import glob

###############################################################################
# Generates SIFT features of the file in the directory
###############################################################################
def generate_sift_features(folder_path):
    num_images = len(glob.glob1(folder_path,"*.png"))
    
    for filename in os.listdir(directory):
        if filename.endswith(".png"): 
            print(os.path.join(directory, filename))


################################################################################
# Main function
################################################################################
def main():
	# Argument Parsing
	parser = argparse.ArgumentParser(description='Bag of Words and Vocabulary Tree')

	parser.add_argument('-i', '--input_file', default= "Inputs/img1.png", help='input file/image path relative to the current directory')

    # Parsed values
	args = parser.parse_args()
	file_path_rel = args.input_file
	file_path     = os.path.join(os.getcwd(), file_path_rel)

