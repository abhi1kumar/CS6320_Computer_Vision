"""
	Version 1 2018/03/04 Abhinav Kumar u1209853
"""

import argparse
import os
import sys
import numpy as np

################################################################################
# Returns the actual index in the matrix
################################################################################
def get_act_index(ind, shift=1):
    return ind-shift


################################################################################
# Cost function of the parity constraints
################################################################################
def get_cost_parity(x, C = 1000):
    x_sum = np.sum(x)
    
    if x_sum % 2 == 0:
        return 0
    else:
        return C


################################################################################
# Cost function of the codeword bits
################################################################################
def get_cost_codeword_bits(num_var_nodes):
    cost = np.zeros((num_var_nodes, 2))
    cost[0,1] = 3
    cost[1,1] = 2
    cost[2,1] = 2.5
    cost[3,1] = 5.4
    cost[4,1] = 4
    cost[5,0] = 0.2
    cost[6,0] = 0.7

    return cost


################################################################################
# Binary numbers in numpy array format
################################################################################
def get_binary(x, n):
    bin_string = format(x, 'b').zfill(n)
    bin_string_with_comma = ','.join(bin_string)
    bin_array = np.fromstring(bin_string_with_comma, dtype=int, sep=',')
    
    return bin_array


################################################################################
# Main function
################################################################################
def main():

    # Parameters
    num_var_nodes = 7
    num_fac_nodes = 10
    max_iter = 10

    ############################################################################        
    # Initialisations
    ############################################################################
    # labels
    labels = np.zeros((num_var_nodes,))

    # beliefs
    belief = np.zeros((num_var_nodes, 2))
    
    # Adjacency Matrix
    adj_matrix = np.zeros((num_var_nodes, num_fac_nodes)).astype(int)
    
    for i in range(num_var_nodes):
        adj_matrix[i,i] = 1

    adj_matrix[get_act_index(1), get_act_index(8)] = 1
    adj_matrix[get_act_index(2), get_act_index(8)] = 1
    adj_matrix[get_act_index(3), get_act_index(8)] = 1
    adj_matrix[get_act_index(5), get_act_index(8)] = 1

    adj_matrix[get_act_index(1), get_act_index(9)] = 1
    adj_matrix[get_act_index(2), get_act_index(9)] = 1
    adj_matrix[get_act_index(4), get_act_index(9)] = 1
    adj_matrix[get_act_index(6), get_act_index(9)] = 1

    adj_matrix[get_act_index(1), get_act_index(10)] = 1
    adj_matrix[get_act_index(3), get_act_index(10)] = 1
    adj_matrix[get_act_index(4), get_act_index(10)] = 1
    adj_matrix[get_act_index(7), get_act_index(10)] = 1

    print("Displaying adjancency matrix of variable nodes (as rows) vs factor nodes (as columns)")
    print(adj_matrix)

    # Cost function of first seven factor nodes
    cost1 = get_cost_codeword_bits(num_var_nodes) 

    # Initialize all messages from var to factor nodes to zeros
    messages_var_to_fac = np.zeros((num_var_nodes, num_fac_nodes, 2))

    # Initialize all messages to ones
    messages_fac_to_var = np.zeros((num_var_nodes, num_fac_nodes, 2))

    ############################################################################ 
    # First terminating condition
    ############################################################################ 
    for i in range(max_iter):

        print(i)

        ########################################################################        
        # Compute outgoing messages of the factor nodes
        # The factor nodes compute a local minima for each of the nodes
        ########################################################################
        for j in range(num_fac_nodes):
            var_indices = np.nonzero(adj_matrix[:,j])[0]
            
            # compute message from this factor node to all other variable nodes
            # to which it is connected to
            for k in range(var_indices.shape[0]):                
                var_node_index  = var_indices[k] 

                #other_var is used to see which messages are summed up for this
                # variable node                
                other_var_index = np.delete(var_indices, k)
                other_var_len   = other_var_index.shape[0]

                #print(other_var_index)
                #print(var_node_index)

                if (other_var_len > 0):
                    # Compute the message from this fac node to other one
                    # for all configurations
                    # 
                    # this also suggests we are in the node 8-10
                    # so use get_cost_parity to get the cost
                    for n in range(2):                                            
                        min_cost = 1000 # a large number

                        #print("n = %d" %(n))

                        for l in range(2**other_var_len):
                            bin_array = get_binary(l, other_var_len+1)
                            bin_array[0] = n
                                                        
                            curr_cost = get_cost_parity(bin_array)
                            temp = curr_cost

                            # Add the value of messages from all other nodes
                            for m in range(other_var_len):
                                curr_cost += messages_var_to_fac[other_var_index[m], j, bin_array[m+1] ]
                            #print(str(bin_array[0]) + " " + str(bin_array[1]) + " " + str(bin_array[2]) + " " + str(bin_array[3]) + " " + str(temp) + " " + str(curr_cost)) 

                            if (curr_cost < min_cost):
                                min_cost = curr_cost

                        messages_fac_to_var[var_node_index, j, n] = min_cost
                            
                else:
                    # only one node connected to this node and therefore use
                    # values of the matrix cost                
                    messages_fac_to_var[var_node_index, j, 0] = cost1[j, 0]
                    messages_fac_to_var[var_node_index, j, 1] = cost1[j, 1]


        # print(messages_fac_to_var[:, :, 0])
        # print(messages_fac_to_var[:, :, 1])

        ########################################################################        
        # Compute belief at a node and the outgoing messages from belief to
        # factor nodes
        ########################################################################
        for j in range(num_var_nodes):
            fac_indices = np.nonzero(adj_matrix[j,:])[0]

            # Get belief
            for k in range(fac_indices.shape[0]):
                fac_node_index  = fac_indices[k]
                belief[j, 0] += messages_fac_to_var[j, fac_node_index, 0]
                belief[j, 1] += messages_fac_to_var[j, fac_node_index, 1]

            # Get the messages from this variable node to all the neighbours
            for k in range(fac_indices.shape[0]):
                fac_node_index  = fac_indices[k]
                messages_var_to_fac[j, fac_node_index, 0] = belief[j, 0] - messages_fac_to_var[j, fac_node_index, 0]
                messages_var_to_fac[j, fac_node_index, 1] = belief[j, 1] - messages_fac_to_var[j, fac_node_index, 1]
          
        # print(belief)

        ########################################################################        
        # Labels and Second terminating condition
        ########################################################################
        temp_labels = np.argmin(belief, axis=1)

        if (np.linalg.norm((labels - temp_labels), ord=1) < 1):
            # new label is same as previous label
            break;
        else:
            labels = temp_labels

    print("**** Final labels **** ")
    print(labels)

if __name__ == '__main__':
    main()
