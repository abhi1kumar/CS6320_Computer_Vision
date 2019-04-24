"""
    Calculates Disparity
	Version 1 2018/03/12 Abhinav Kumar u1209853
"""

import argparse
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


################################################################################
# Cost function of the unary nodes
# d is a scalar number
################################################################################
def get_cost_unary(left, right, x, y, d):
    return np.abs( left[y, x] - right[y, x-d])


################################################################################
# Cost function of the pairwise nodes: Vectorised Implementation
# Using Potts Model
################################################################################
def get_cost_pairs(d, d1):
    diff = np.abs(d-d1)
    out  = np.zeros(diff.shape)
    out[diff > 0] = 1
    return out


################################################################################
# Cost function of the pairwise nodes
# Using Potts Model
################################################################################
def get_cost_pairs_new(d, d1):
    if (np.abs(d - d1) > 0):
        return 1
    else:
        return 0


################################################################################
# Returns index of the pixel in the image size from the key
# [x, y] suggests the coordinates in the original image
# The convention is as follows -
#        0 
#        A
#        |
#        |
# 3<---(x,y) ---> 2  
#        |
#        |
#        V
#        1
################################################################################
def get_index_from_key(x, y, h, w, key):
    out_x = -1
    out_y = -1

    if   (key == 0):
        if (y-1 >= 0):
            out_y = y-1
            out_x = x
    elif (key == 1):
        if (y+1 < h):
            out_y = y-1
            out_x = x
    elif (key == 2):
        if (x+1 < w):
            out_x = x+1
            out_y = y
    else:
        if (x-1 >= 0):
            out_x = x-1
            out_y = y

    return out_x, out_y    


################################################################################
# Calculates cost of the disparity image
################################################################################
def get_cost_image(img1, img2, disparity, lambda_weight):
    h,w = img1.shape

    cost = 0.0

    for y in range(h):
        for x in range(w):
            d = int(disparity[y, x])

            # Unary costs
            if (x-d >= 0):
                cost += get_cost_unary(img1, img2, x, y, d)
        
            # Pairwise costs
            for key in range(4):
                x2, y2 = get_index_from_key(x, y, h, w, key)
                if (x2 < 0 or y2 < 0):
                    pass
                else:
                    d2 = int(disparity[y2, x2])
                    cost += lambda_weight * get_cost_pairs_new(d, d2)

    return cost    


################################################################################
# Main function
################################################################################
def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Calculates Disparity')

    parser.add_argument('-l', '--left_image_file' , default= "Inputs/left1.png" , help='left  image path relative to the current directory')
    parser.add_argument('-r', '--right_image_file', default= "Inputs/right1.png", help='right image path relative to the current directory')
    parser.add_argument('-w', '--weight'          , default= "100"              , help='value of the lambda')

    # Parsed values
    args = parser.parse_args()
    left_file_path_rel  = args.left_image_file
    right_file_path_rel = args.right_image_file
    lambda_weight       = int(args.weight)

    left_file_path  = os.path.join(os.getcwd(), left_file_path_rel)
    right_file_path = os.path.join(os.getcwd(), right_file_path_rel)

    # Parameters
    output_folder   = os.path.join(os.getcwd(), 'Outputs')
    DISPARITY_RANGE = 50
    max_iter        = 20

    img1c = cv2.imread(left_file_path)
    img1c = cv2.cvtColor(img1c, cv2.COLOR_BGR2RGB)
    img1  = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)

    img2c = cv2.imread(right_file_path)
    img2c = cv2.cvtColor(img2c, cv2.COLOR_BGR2RGB)
    img2  = cv2.cvtColor(img2c, cv2.COLOR_BGR2GRAY)


    print("Left  Input Image = " + left_file_path)
    print("Right Input Image = " + right_file_path)	
    print("Lambda = %d" %(lambda_weight))

    mini = 160
    maxi = 240
    #img2 = img2[mini:maxi, mini:maxi]
    #img1 = img1[mini:maxi, mini:maxi]
    print(img1.shape)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    h,w = img1.shape 

    # Intialise the graph with unary and binary factor nodes    
    unary_fac_nodes = np.zeros((h, w))

    # labels
    labels = np.zeros((h, w))

    # beliefs
    belief = np.zeros((h, w, DISPARITY_RANGE), dtype=np.float64)

    # messages
    messages_var_to_unary_fac = np.zeros((h, w, DISPARITY_RANGE), dtype=np.float64)
    messages_var_to_pair1_fac = np.zeros((h, w, 4, DISPARITY_RANGE), dtype=np.float64) # for top-down connections,   key = 0,1
    messages_var_to_pair2_fac = np.zeros((h, w, 4, DISPARITY_RANGE), dtype=np.float64) # for left-right connections, key = 2,3

    messages_unary_fac_to_var = np.zeros((h,   w, DISPARITY_RANGE), dtype=np.float64)
    messages_pair1_fac_to_var = np.zeros((h-1, w, 4, DISPARITY_RANGE), dtype=np.float64) # for top-down connections,   key = 0,1
    messages_pair2_fac_to_var = np.zeros((h, w-1, 4, DISPARITY_RANGE), dtype=np.float64) # for left-right connections, key = 2,3

    ############################################################################ 
    # First terminating condition
    ############################################################################ 
    for iter in range(max_iter):
        print("Iteration %d, Cost %f" %(iter, get_cost_image(img1, img2, labels, lambda_weight) ))
        
        ########################################################################        
        # Compute outgoing messages of the factor nodes
        # The factor nodes compute a local minima for each of the nodes
        ########################################################################
        # First for unary factor nodes
        for i in range(h):
            for j in range(w):
                # for all ranges of disparity
                for d in range(DISPARITY_RANGE):
                       messages_unary_fac_to_var[i, j, d] = get_cost_unary(img1, img2, j, i, d)             
        
        # Then for pairs1 nodes
        for i in range(h-1):
            for j in range(w):
                xi = j
                yi = i

                # for top-down connections, key = 0,1
                for key in range(2):

                    for d in range(DISPARITY_RANGE):
                        d1 = np.arange(DISPARITY_RANGE)
                        temp_cost = lambda_weight * get_cost_pairs(d, d1)
                        if (key == 0):
                            temp_cost += messages_var_to_pair1_fac[i+1, j, 0, :] # message from other node which is at the bottom
                        else:
                            temp_cost += messages_var_to_pair1_fac[i  , j, 1, :] # message from other node which is at the top

                        messages_pair1_fac_to_var[i, j, key, d] = np.min(temp_cost)

        # Then for pairs2 nodes
        for i in range(h):
            for j in range(w-1):
                xi = j
                yi = i

                # for top-down connections, key = 2,3
                for key in range(2,4):
                    
                    for d in range(DISPARITY_RANGE):
                        d1 = np.arange(DISPARITY_RANGE)
                        temp_cost = lambda_weight * get_cost_pairs(d, d1)
                        if (key == 2):
                            temp_cost += messages_var_to_pair2_fac[i, j  , 2, d1] # message from other node which is at the left
                        else:
                            temp_cost += messages_var_to_pair2_fac[i, j+1, 3, d1] # message from other node which is at the right

                        messages_pair2_fac_to_var[i, j, key, d] = np.min(temp_cost)

        ########################################################################        
        # Compute belief at a variable node and the outgoing messages from belief to
        # factor nodes
        ########################################################################
        for i in range(h):
            for j in range(w):                

                #print("(%d, %d)" %(i, j))

                # Get belief from unary nodes
                belief[i, j, :] = messages_unary_fac_to_var[i, j, :]

                # fac node at bottom of this node
                if (i < h-1):
                    belief[i, j, :] += messages_pair1_fac_to_var[i, j, 0, :]

                # fac node at top of this node                                        
                if (i > 0):
                    belief[i, j, :] += messages_pair1_fac_to_var[i-1, j, 1, :]
                
                # fac node at right of this node
                if (j < w-1):
                    belief[i, j, :] += messages_pair2_fac_to_var[i, j, 3, :]

                # fac node at left of this node
                if (j >  0):
                    belief[i, j, :] += messages_pair2_fac_to_var[i, j-1, 2, :]
                
                # Now compute the outgoing messages                         
                messages_var_to_unary_fac[i, j, :]          = belief[i, j, :] - messages_unary_fac_to_var[i, j, :]

                # fac node at bottom of this node
                if (i < h-1):
                    messages_var_to_pair1_fac[i, j, 0, :]   = belief[i, j, :] - messages_pair1_fac_to_var[i, j, 0, :]

                # fac node at top of this node                                        
                if (i > 0):
                    messages_var_to_pair1_fac[i-1, j, 1, :] = belief[i, j, :] - messages_pair1_fac_to_var[i-1, j, 1, :]
                
                # fac node at right of this node
                if (j < w-1):
                    messages_var_to_pair2_fac[i, j, 3, :]   = belief[i, j, :] - messages_pair2_fac_to_var[i, j, 3, :]

                # fac node at left of this node
                if (j >  0):
                    messages_var_to_pair2_fac[i, j-1, 2, :] = belief[i, j, :] - messages_pair2_fac_to_var[i, j-1, 2, :]

        ########################################################################        
        # Labels and Second terminating condition
        ########################################################################
        temp_labels = np.argmin(belief, axis=2)

        if (np.linalg.norm((labels.ravel() - temp_labels.ravel()), ord=1) < 1):
            # new label is same as previous label
            break;
        else:
            labels = temp_labels
     
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    base             = os.path.basename(left_file_path)
    file_name_no_ext = os.path.splitext(base)[0]
    output_file_1    = os.path.join(output_folder, file_name_no_ext[4:] + '_lambda_' + str(lambda_weight) +'_orig_disp.png')
    output_file_2    = os.path.join(output_folder, file_name_no_ext[4:] + '_lambda_' + str(lambda_weight) +'_disparity.png')
    print("\nOutput Orig + Dispar Path = " + output_file_1)    
    print("Output Disparity Map Path = " + output_file_2 + "\n")

    # Plot and Save
    plt.subplot(131), plt.imshow(img1c)
    plt.title('Left Image') , plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img2c)
    plt.title('Right Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(labels, cmap = 'gray')
    plt.title('Disparity')  , plt.xticks([]), plt.yticks([])
    plt.savefig(output_file_1, bbox_inches="tight")
    #plt.show()
    plt.close()
    
    plt.imshow(labels, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig(output_file_2, bbox_inches="tight")   
    plt.close()

if __name__ == '__main__':
    main()
