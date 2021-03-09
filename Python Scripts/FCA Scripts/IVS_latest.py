# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:42:29 2019

@author: hmusugu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:12:27 2019

@author: hmusugu
"""

import numpy as np
import sys
import random
import pandas as pd

#Input data CSV file - configs that will be removed
removed_configs_file = "C:\\Users\\hmusugu\\Desktop\\FCA\\Aman's Data\\Removed_Configs.xlsx"
#Input data CSV file - SIMS configuration data file
infile = "C:\\Users\\hmusugu\\Desktop\\FCA\\Aman's Data\\IVS.xlsx"


df = pd.read_excel(infile, header = None)
input_matrix = np.array(df.values)

df1 = pd.read_excel(removed_configs_file,header=None)
removed_indices = df1.iloc[:,0:1].values.tolist()


fixed_option_columns = []
fixed_option_choices_values = ['TRAILHAWK']

prioritized_columns = []
ranked_option_values = {}
                       #{'GVWS_AAZ_': ['6010 LB. GVW (AAZF7)','6050 LB. GVW (AAZFD)','6100 LB. GVW (AAZFG)','6150 LB. GVW (AAZFH)','6200 LB. GVW (AAZFL)','6250 LB. GVW (AAZFN)','6300 LB. GVW (AAZFR)','6350 LB. GVW (AAZFT)','6400 LB. GVW (AAZFW)','6500 LB. GVW (AAZGC)','6650 LB. GVW (AAZA6)','6750 LB. GVW (AAZGQ)','6800 LB. GVW (AAZGT)','6900 LB. GVW (AAZGW)','6950 LB. GVW (AAZGZ)','7000 LB. GVW (AAZHL)','7050 LB. GVW (AAZHM)','7600 LB. GVW (AAZL4)','7850 LB. GVW (AAZGM)']} 
                       #{'CAB_STYLE_LT_TRK': ['SINGLE CAB (REGULAR CAB)','SUPER SINGLE CAB (SUPER CAB)','DOUBLE CAB (CREW CAB)'],
                       #'LIGHT_TRUCK_WHEELBASES': ['122" Wheelbase','141" Wheelbase','145" Wheelbase','157" Wheelbase','163.7" Wheelbase']}

config_ids_column = 'Configuration ID'
#Provide the column name corresponding to Contribution Profit
#contrib_prof_column = 'Adj_Incentives_Avg'
#Provide the column name corresponding to Net Revenue
net_rev_column = 'MSRP'
#Provide the column name corresponding to Volume for each configuration
volume_column = 'Volume'
#Provide the column name corresponding to the number of total parts for each configuration
total_parts_column = 'PART_COUNT'
#Provide the column name corresponding to the number of unique parts for each configuration
unique_parts_column = 'UNIQUE_PART'
#Provide the column name corresponding to labor time for each configuration


#Determine column indices for each of the inputs in the input file

fixed_option_columns_indices = [list(input_matrix[0]).index(x) for x in fixed_option_columns] #What is happening here???
prioritized_columns_indices = [list(input_matrix[0]).index(x) for x in prioritized_columns]   #What is happening here???

#config_ids_column_index = list(input_matrix[0]).index(config_ids_column)
#contrib_prof_column_index = list(input_matrix[0]).index(contrib_prof_column)
net_rev_column_index = list(input_matrix[0]).index(net_rev_column)
volume_column_index = list(input_matrix[0]).index(volume_column)
#total_parts_column_index = list(input_matrix[0]).index(total_parts_column)
#unique_parts_column_index = list(input_matrix[0]).index(unique_parts_column) 


#Identify unique choices for each option in option_columns
options_choices_unique = list(input_matrix[0][6:])
fixed_options_choices_all = [[input_matrix[i][x] for i in range(1,len(input_matrix))] for x in fixed_option_columns_indices]
fixed_options_choices_unique = list(set([item for sublist in fixed_options_choices_all for item in sublist]))
fixed_options_choices_unique = list(set(fixed_options_choices_unique + fixed_option_choices_values))



#Create the new transformed matrices (one with all features (OUTPUT_MATRIX), the other only with prioritized features(OUTPUT_MATRIX_PRIORITISED))

#############################################################################
#Output Matrix
#############################################################################
output_matrix = df.iloc[1:,6:].values.tolist()
output_matrix = np.array(output_matrix).astype(int)


#Create Config_IDs
Config_IDs = [input_matrix[i][1] for i in range(0,len(output_matrix)+1)]

#Update the removed_indices to be the corresponding row indice value in Config_IDs
removed_indices = [Config_IDs.index(x[0]) for x in removed_indices[1:len(removed_indices)]]

#Start of Distance Calculation
###############################################################################

#######Hardcode
remaining_configurations_global = []
for i2 in range(2601,2717):
    remaining_configurations_global.append(i2)

#The remaining_configurations_global:
#   -Are configurations that have not been removed by removed_configs_file specifications
volume_transfer_output = []
contrib_prof_transfer_output = []
total_incentives_rearranged = 0

#For each one of the removed_indices, find the remaining configurations that are closest to it
for i in removed_indices:

    #Start with the "total" set of remaining cofigurations (remaining_configurations_global)
    #in each removed_indices iteration
    remaining_configurations = remaining_configurations_global

    #Eliminating remaining_configurations that do not have the same fixed_option_columns values
    for j in fixed_options_choices_unique:
        if output_matrix[i][options_choices_unique.index(j)] == 1:
            remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index(j)] == 1]
            
    #Eliminating remaining_configurations as follows:
    #   -If removed_indices is a 4WD, only 4WD and AWD are kept in remaining_configurations
    #   -If removed_indices is a FWD, only FWD are kept in remaining_configurations
    #   -If removed_indices is a AWD, only AWD are kept in remaining_configurations
    if not ((output_matrix[i][options_choices_unique.index('AWD')] == 1 or output_matrix[i][options_choices_unique.index('4X4')] == 1) and (output_matrix[i][options_choices_unique.index('EDE')] == 1 or output_matrix[i][options_choices_unique.index('EC1')] == 1)):
        if output_matrix[i][options_choices_unique.index('4X4')] == 1:
            remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('4X4')] == 1 or output_matrix[x][options_choices_unique.index('AWD')] == 1]
        elif output_matrix[i][options_choices_unique.index('FWD')] == 1:
            remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('FWD')] == 1]
        elif output_matrix[i][options_choices_unique.index('AWD')] == 1:
            remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('AWD')] == 1]
        else:
            print ("No drive type")
            sys.exit()

        if output_matrix[i][options_choices_unique.index('TRAILHAWK')] == 0:
            if output_matrix[i][options_choices_unique.index('EDE')] == 1 or output_matrix[i][options_choices_unique.index('EC1')] == 1:
                flip = random.random()
                if flip <= 2: #0.80:
                    remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('EDE')] == 1 or output_matrix[x][options_choices_unique.index('EC1')] == 1]
            elif output_matrix[i][options_choices_unique.index('EHK')] == 1:
                remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('EHK')] == 1]
            else:
                print ("No engine type")
                sys.exit()
    else:
        #Map to either FWD 2.4L or AWD 3.2L
        #remaining_configurations = [x for x in remaining_configurations if not (output_matrix[x][options_choices_unique.index('FWD')] == 1 and output_matrix[x][options_choices_unique.index('3.2L V6 24V VVT ENGINE W/ESS')] == 1)]        

        #Map to either FWD 2.4L or AWD 3.2L at 50-50 coin flip
        flip = random.random()
        if flip <= 0.40:
            remaining_configurations_test = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('FWD')] == 1 and (output_matrix[x][options_choices_unique.index('EDE')] == 1 or output_matrix[x][options_choices_unique.index('EC1')] == 1)]        
            if len(remaining_configurations_test) == 0:
                remaining_configurations = [x for x in remaining_configurations if (output_matrix[x][options_choices_unique.index('AWD')] == 1 or output_matrix[x][options_choices_unique.index('4X4')] == 1) and output_matrix[x][options_choices_unique.index('EHK')] == 1]                 
            else:
                remaining_configurations = remaining_configurations_test
        else:
            remaining_configurations_test = [x for x in remaining_configurations if (output_matrix[x][options_choices_unique.index('AWD')] == 1 or output_matrix[x][options_choices_unique.index('4X4')] == 1) and output_matrix[x][options_choices_unique.index('EHK')] == 1]                 
            if len(remaining_configurations_test) == 0:
                remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('FWD')] == 1 and (output_matrix[x][options_choices_unique.index('EDE')] == 1 or output_matrix[x][options_choices_unique.index('EC1')] == 1)]        
            else:
                remaining_configurations = remaining_configurations_test

        # #Map to either FWD 2.4L or AWD 3.2L at MSRP Bounds
        # current_MSRP = input_matrix[i][net_rev_column_index]
        # if output_matrix[i][options_choices_unique.index('SPORT')] == 1:
        #     if current_MSRP <= 26750:
        #         remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('FWD')] == 1 and (output_matrix[x][options_choices_unique.index('2.4L I4 PZEV M-AIR ENGINE')] == 1 or output_matrix[x][options_choices_unique.index('2.4L I4 MULTIAIR ENGINE')] == 1)]
        #     else:
        #         remaining_configurations = [x for x in remaining_configurations if (output_matrix[x][options_choices_unique.index('AWD')] == 1 or output_matrix[x][options_choices_unique.index('4WD')] == 1) and output_matrix[x][options_choices_unique.index('3.2L V6 24V VVT ENGINE W/ESS')] == 1]                 
        # elif output_matrix[i][options_choices_unique.index('LATITUDE')] == 1:
        #     if current_MSRP <= 29250:
        #         remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('FWD')] == 1 and (output_matrix[x][options_choices_unique.index('2.4L I4 PZEV M-AIR ENGINE')] == 1 or output_matrix[x][options_choices_unique.index('2.4L I4 MULTIAIR ENGINE')] == 1)]
        #     else:
        #         remaining_configurations = [x for x in remaining_configurations if (output_matrix[x][options_choices_unique.index('AWD')] == 1 or output_matrix[x][options_choices_unique.index('4WD')] == 1) and output_matrix[x][options_choices_unique.index('3.2L V6 24V VVT ENGINE W/ESS')] == 1]                 
        # elif output_matrix[i][options_choices_unique.index('LIMITED')] == 1:
        #     if current_MSRP <= 33250:
        #         remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index('FWD')] == 1 and (output_matrix[x][options_choices_unique.index('2.4L I4 PZEV M-AIR ENGINE')] == 1 or output_matrix[x][options_choices_unique.index('2.4L I4 MULTIAIR ENGINE')] == 1)]
        #     else:
        #         remaining_configurations = [x for x in remaining_configurations if (output_matrix[x][options_choices_unique.index('AWD')] == 1 or output_matrix[x][options_choices_unique.index('4WD')] == 1) and output_matrix[x][options_choices_unique.index('3.2L V6 24V VVT ENGINE W/ESS')] == 1]                 

    #The remaining_configurations:
    #   -Are configurations that have not been removed by removed_configs_file specifications    
    #   -Are configurations that have the same fixed_option_columns values
    
    #If there exists configurations in remaining_configurations, continue 
    #removing configurations based on ranked_option_values restrictions
    if remaining_configurations:
        #Eliminating remaining_configurations that step down in ranked_option_values
        for j in ranked_option_values.keys():
            for k in range(0,len(ranked_option_values[j])):
                #Find the "feature ranks" of the current removed configuration
                if output_matrix[i][options_choices_unique.index(ranked_option_values[j][k])] == 1:
                    #print j,k
                    #Eliminate the "feature ranks" of the remaining_configurations with lower ranks than the current removed configuration
                    for l in range(0,k):
                        remaining_configurations = [x for x in remaining_configurations if output_matrix[x][options_choices_unique.index(ranked_option_values[j][l])] == 0]
                        #print len(remaining_configurations)
        
        #The remaining_configurations:
        #   -Are configurations that have not been removed by removed_configs_file specifications
        #   -Are configurations that have the same fixed_option_columns values        
        #   -Are configurations that fulfil the requirements in ranked_option_values (i.e. the configurations do not step down)
        
        #If there exists configurations in remaining_configurations, continue removing configurations
        #based on net_rev_range restrictions
        if remaining_configurations:
            #Eliminiate configurations that are not inside of bounds of the net_rev_range restriction
            #net_rev_upper_bound = float(input_matrix[i][net_rev_column_index]) + net_rev_range
            #net_rev_lower_bound = float(input_matrix[i][net_rev_column_index]) - net_rev_range
            
            #net_rev_upper_bound = float(input_matrix[i][net_rev_column_index])*(1+net_rev_percent)
            #net_rev_lower_bound = float(input_matrix[i][net_rev_column_index])*(1-net_rev_percent)            
            # remaining_configurations_not_relaxed = [x for x in remaining_configurations if float(input_matrix[x][net_rev_column_index]) >= net_rev_lower_bound and float(input_matrix[x][net_rev_column_index]) <= net_rev_upper_bound]
            # remaining_configurations = [x for x in remaining_configurations if float(input_matrix[x][net_rev_column_index]) >= net_rev_lower_bound and float(input_matrix[x][net_rev_column_index]) <= net_rev_upper_bound]

            # if len(remaining_configurations_not_relaxed) == 0:
            #     remaining_configurations = 
            #     total_incentives_rearranged += 
            # else:
            #     remaining_configurations = remaining_configurations_not_relaxed

            # #Buyback the volume
            # if volume_buyback_dict[Config_IDs[i]] in volume_buyback:
            #     if volume_buyback[volume_buyback_dict[Config_IDs[i]]] > 0:
            #         net_rev_upper_bound = net_rev_upper_bound + volume_buyback[volume_buyback_dict[Config_IDs[i]]]
            #         net_rev_lower_bound = 0 #net_rev_lower_bound - volume_buyback[volume_buyback_dict[Config_IDs[i]]]
            #     else:
            #         net_rev_upper_bound = net_rev_upper_bound - volume_buyback[volume_buyback_dict[Config_IDs[i]]]
            #         net_rev_lower_bound = 0 #net_rev_lower_bound + volume_buyback[volume_buyback_dict[Config_IDs[i]]]
            # remaining_configurations = [x for x in remaining_configurations if float(input_matrix[x][net_rev_column_index]) >= net_rev_lower_bound and float(input_matrix[x][net_rev_column_index]) <= net_rev_upper_bound]

            #If the net_rev_range restrictions eliminates all remaining_configurations, modify
            #the net_rev_upper_bound by adding the upper_bound_relaxation constant and recalculate
            ### if len(remaining_configurations_not_relaxed) == 0 and upper_bound_relaxation > 0:
            ###     net_rev_upper_bound = net_rev_upper_bound + upper_bound_relaxation
            ###     remaining_configurations = [x for x in remaining_configurations if float(input_matrix[x][net_rev_column_index]) >= net_rev_lower_bound and float(input_matrix[x][net_rev_column_index]) <= net_rev_upper_bound]
            ### else:
            ###     remaining_configurations = remaining_configurations_not_relaxed

            #The remaining_configurations:
            #   -Are configurations that have not been removed by removed_configs_file specifications
            #   -Are configurations that have the same fixed_option_columns values            
            #   -Are configurations that fulfil the requirements in ranked_option_values (i.e. the configurations do not step down)
            #   -Are configurations that have net revenue values in the bounds of net_rev_range
        
            #If there exists configurations in remaining_configurations, continue removing configurations
            #based on distance restrictions, calculated based on only the options in prioritized_columns,
            #and then updated based on all the options in option_columns
            if remaining_configurations:
                ##Calculate all prioritized distances from the current removed_indices value
                #all_distance_values = np.sum(abs(output_matrix_prioritized[i] - output_matrix_prioritized), axis = 1)
                ##Select the subset of prioritized distances that are in the set of configurations still remaining 
                #remaining_distance_values = all_distance_values[np.array(remaining_configurations)]
                ##Finding closest prioritized distance from the current removed_indices value to the remaining_configurations     
                #closest_distance = min(remaining_distance_values)
                ##Finding all configurations that are closest_distance away from the removed_indices value
                #remaining_configurations = [remaining_configurations[x] for x in np.where(remaining_distance_values == closest_distance)[0].tolist()]
                
                #The remaining_configurations:
                #   -Are configurations that have not been removed by removed_configs_file specifications
                #   -Are configurations that have the same fixed_option_columns values                
                #   -Are configurations that fulfil the requirements in ranked_option_values (i.e. the configurations do not step down)
                #   -Are configurations that have net revenue values in the bounds of net_rev_range
                #   -Are, from a prioritized perspective, the closest_distance to the removed configuration (removed_indices)
    
                #For the configurations in remaining_configurations, find which configurations are the closest, taking into account all option_columns
                all_options_distance_values = np.sum(abs(output_matrix[i] - output_matrix), axis = 1)
                #Select the subset of distances that are in the set of configurations from remaining_configurations
                final_distance_values = [all_options_distance_values[x] for x in remaining_configurations]
                #Finding closest distance from the current removed_indices value to remaining_configurations
                final_closest_distance = min(final_distance_values)
                #Finding all configurations that are closest_distance away from the removed_indices value
                remaining_configurations = [remaining_configurations[x] for x in np.where(final_distance_values == final_closest_distance)[0].tolist()]
                
                #The remaining_configurations:
                #   -Are configurations that have not been removed by removed_configs_file specifications
                #   -Are configurations that have the same fixed_option_columns values                
                #   -Are configurations that fulfil the requirements in ranked_option_values (i.e. the configurations do not step down)
                #   -Are configurations that have net revenue values in the bounds of net_rev_range
                #   -Are, from a prioritized perspective, the closest_distance to the removed configuration (removed_indices)
                #   -Are, from a prioritized perspective and from an all options_columns perspective, the closest_distance to the removed configuration (removed_indices)
                
                #Volme that is removed (VOLUME CALCULATION)
                removed_volume = int(input_matrix[i][volume_column_index])
                #Volume values that will receive the removed volume (including the net_rev_range restriction)
                volume_to_be_updated = [input_matrix[x][volume_column_index] for x in remaining_configurations]
                volume_percentages = [float(x)/sum([float(y) for y in volume_to_be_updated]) for x in volume_to_be_updated]
                
                #The following code allocates the removed volume *in whole values* to the volume_to_be_updated
                #The removed volume is allocated based on the distribution of volume in the remaining_configurations
                #The final unit of volume to be distributed is assigned to the closest_net_rev_index configuration
                running_sum_volume = 0
                amortized_volume = 0
                distributed_volume = [0 for x in range(0,len(volume_percentages))]
                closest_net_rev_val = min([float(input_matrix[x][net_rev_column_index]) for x in remaining_configurations], key=lambda x:abs(x - float(input_matrix[i][net_rev_column_index])))
                closest_net_rev_index = [float(input_matrix[x][net_rev_column_index]) for x in remaining_configurations].index(closest_net_rev_val)
                
                def distribute_volume_whole(percentage, volume):
                    global amortized_volume
                    real_volume = percentage * volume + amortized_volume
                    natural_volume = np.floor(real_volume)
                    amortized_volume = real_volume - natural_volume
                    return natural_volume
                
                for j in range(0,len(distributed_volume)):
                    if j == closest_net_rev_index:
                        pass
                    else:
                        distributed_volume[j] = distribute_volume_whole(volume_percentages[j], removed_volume)
                        running_sum_volume += distributed_volume[j]
                    if j == len(distributed_volume)-1:
                        distributed_volume[closest_net_rev_index] = removed_volume - running_sum_volume;
                
                #volume_transfer_output is a list of [removed configuration, configuration transferred to, amount of volume transferred]
                #Note that the configurations that we have been tracking are the rows in input_matrix, and at this stage
                #we map the rows back to their respective Config_IDs (which will be identical if the Config_IDs are sorted from 1 to len(input_matrix)-1)
                current_volume_transfer_output = [[Config_IDs[i],Config_IDs[remaining_configurations[x]],str(distributed_volume[x])] for x in range(0,len(distributed_volume)) if distributed_volume[x] > 0.0]
                for x in current_volume_transfer_output:
                    volume_transfer_output.append(x)
                
                #contrib_prof_transfer_output is a list of [removed configuration, contribution profit from configuration before transfer, contribution profit from configurations after transfer]
                #contrib_prof_before = float(input_matrix[i][contrib_prof_column_index])*float(input_matrix[i][volume_column_index])
                #contrib_prof_after = sum([float(input_matrix[x][contrib_prof_column_index])*(distributed_volume[ind]) for ind,x in enumerate(remaining_configurations)])
                #contrib_prof_transfer_output.append([Config_IDs[i], str(contrib_prof_before), str(contrib_prof_after)])
            else:
                #Removed configuration loses all volume because there are no configurations to transfer to
                volume_transfer_output.append([Config_IDs[i],'LOST VOLUME',str(input_matrix[i][volume_column_index])])
                #Removed configuration loses all contrib_prof because there are no configurations to transfer to
                #contrib_prof_before = float(input_matrix[i][contrib_prof_column_index])*float(input_matrix[i][volume_column_index])
                #contrib_prof_after = 0
                #contrib_prof_transfer_output.append([Config_IDs[i], str(contrib_prof_before), str(contrib_prof_after)])
        else:
            #Removed configuration loses all volume because there are no configurations to transfer to
            volume_transfer_output.append([Config_IDs[i],'LOST VOLUME',str(input_matrix[i][volume_column_index])])
            #Removed configuration loses all contrib_prof because there are no configurations to transfer to
            #contrib_prof_before = float(input_matrix[i][contrib_prof_column_index])*float(input_matrix[i][volume_column_index])
            #contrib_prof_after = 0
            #contrib_prof_transfer_output.append([Config_IDs[i], str(contrib_prof_before), str(contrib_prof_after)])
    else:
        #Removed configuration loses all volume because there are no configurations to transfer to
        volume_transfer_output.append([Config_IDs[i],'LOST VOLUME',str(input_matrix[i][volume_column_index])])
        #Removed configuration loses all contrib_prof because there are no configurations to transfer to
        #contrib_prof_before = float(input_matrix[i][contrib_prof_column_index])*float(input_matrix[i][volume_column_index])
        #contrib_prof_after = 0
        #contrib_prof_transfer_output.append([Config_IDs[i], str(contrib_prof_before), str(contrib_prof_after)])
            
    #print i

#My Distance Calculation method:
output_list = output_matrix.tolist()
df_config_dist = pd.DataFrame(output_list, Config_IDs)
df_config_dist_ID = df_config_dist[0:15251]
df_config_dist_Proposed = df_config_dist[15251:15387]
config_dist_list = df_config_dist.values.tolist()
volume_transfer_ids = pd.DataFrame(volume_transfer_output, columns = ['ID','Proposed','score'])
volume_transfer_ids = volume_transfer_ids.drop(['score'], axis = 1)
id_list = volume_transfer_ids['ID'].values.tolist()
id_list_proposed = volume_transfer_ids['Proposed'].values.tolist()

#Storing zeroes and ones for the ID column
config_store = []
for p1 in range(len(id_list)):
    for p2 in range(len(df_config_dist_ID)):
        if (id_list[p1] == df_config_dist_ID.index[p2]):
            dummy = df_config_dist_ID[p2:p2+1]
            dummy = dummy.values.tolist()
            config_store.append(dummy)
                       
#Storing zeroes and ones for the ID column
config_store_Proposed = []
for p1 in range(len(id_list_proposed)):
    for p2 in range(len(df_config_dist_Proposed)):
        if (id_list_proposed[p1] == df_config_dist_Proposed.index[p2]):
            dummy2 = df_config_dist_Proposed[p2:p2+1]
            dummy2 = dummy2.values.tolist()
            config_store_Proposed.append(dummy2)          
        
        
config_dist  = []
for r1 in range(len(config_store)):
    a = np.array(config_store[r1])
    b = np.array(config_store_Proposed[r1])
    c = (a-b).tolist()
    count = c[0].count(1)
    config_dist.append(count)

df_distance = pd.DataFrame(np.column_stack([id_list, id_list_proposed,config_dist]), columns = ['Removed Configuration ID','Transfer to Configuration ID', 'Configuration Distance'])
    
########Ends here

###############################################################################
#Output all of the Data
###############################################################################

#Volume Tranfer Output
current_output3 = []
for i in range(0,len(volume_transfer_output)):
    current_output3.append(volume_transfer_output[i])  
df_vol_trans = pd.DataFrame(current_output3, columns = ['Removed Config. ID', 'Proposed to Config.ID', 'Amount of Volume Transfered'])    
df_vol_trans.to_csv("Volume_Transfer1.csv")
        
    
    










