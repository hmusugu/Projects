#Python 2.7

###############################################################################

import csv
import sys
import time
from collections import Counter
import os

###############################################################################
#Input Values
###############################################################################
#Pick a folder for outputs
FOLDER = "Configs_to_Part_Fallout_Analysis_04052017_Iteration2.7" #Configs_to_Part_Fallout_Analysis_03302017_Iteration2.5"
#Pick the correct input case (CASE in [1,2,3,4,5,6,7])
CASE = 2
#Provide the file paths
PATH = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\"
NEW_PATH = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\" + FOLDER + "\\"
#Provide the parts file
infile_parts = PATH + "Parts_Detail_FINAL_Exterior_Color_to_TRM_2016.csv"
#Provide the input subset configuration file
#PATH
subset_of_configurations_all_ALL_file = "Subset_of_Configurations_2016_ALL.csv"
subset_of_configurations_all_INT_file = "Subset_of_Configurations_2016_INT.csv"
subset_of_configurations_all_US_CAN_MEX_file = "Subset_of_Configurations_2016_US_CAN_MEX.csv"
#NEW_PATH
subset_of_configurations_file = "Iteration 2.7_Configs to Remain RemoveFWD3.2.csv" #"Subset_of_Configurations_2016_All_Configs_List 03302017_v2.csv"
subset_of_configurations_some_INT_file = "Subset_of_Configurations_2016_Only_INT_Configs_List 03302017_v2.csv"
subset_of_configurations_some_US_CAN_MEX_file = "Subset_of_Configurations_2016_Only_US_CAN_MEX_Configs_List 03302017_v2.csv"

if CASE == 1:
	infile_configuration_world_file = PATH + subset_of_configurations_all_ALL_file
	infile_configuration_ID_subset = PATH + "NULL.csv"
	subset_flag = 1 #If 0, output parts that are unique to the configurations in infile_configuration_ID_subset
					#If 1, output parts that are unique to the configurations NOT in infile_configuration_ID_subset
	outfile = NEW_PATH + "OUTPUT_All_Parts_2016.csv"
elif CASE == 2:
	infile_configuration_world_file = PATH + subset_of_configurations_all_ALL_file
	infile_configuration_ID_subset = NEW_PATH + subset_of_configurations_file
	subset_flag = 1 #If 0, output parts that are unique to the configurations in infile_configuration_ID_subset
					#If 1, output parts that are unique to the configurations NOT in infile_configuration_ID_subset
	outfile = NEW_PATH + "OUTPUT_Part_Fallout_From_All_if_only_keep_" + subset_of_configurations_file
elif CASE == 3:
	infile_configuration_world_file = PATH + subset_of_configurations_all_ALL_file
	infile_configuration_ID_subset = PATH + subset_of_configurations_all_INT_file
	subset_flag = 0 #If 0, output parts that are unique to the configurations in infile_configuration_ID_subset
					#If 1, output parts that are unique to the configurations NOT in infile_configuration_ID_subset
	outfile = NEW_PATH + "OUTPUT_Unique_Parts_of_" + subset_of_configurations_all_INT_file
elif CASE == 4:
	infile_configuration_world_file = PATH + subset_of_configurations_all_INT_file
	infile_configuration_ID_subset = NEW_PATH + subset_of_configurations_some_INT_file
	subset_flag = 1 #If 0, output parts that are unique to the configurations in infile_configuration_ID_subset
					#If 1, output parts that are unique to the configurations NOT in infile_configuration_ID_subset
	outfile = NEW_PATH + "OUTPUT_Part_Fallout_From_INT_if_only_keep_" + subset_of_configurations_some_INT_file
elif CASE == 5:
	infile_configuration_world_file = PATH + subset_of_configurations_all_ALL_file
	infile_configuration_ID_subset = PATH + subset_of_configurations_all_US_CAN_MEX_file
	subset_flag = 0 #If 0, output parts that are unique to the configurations in infile_configuration_ID_subset
					#If 1, output parts that are unique to the configurations NOT in infile_configuration_ID_subset
	outfile = NEW_PATH + "OUTPUT_Unique_Parts_of_" + subset_of_configurations_all_US_CAN_MEX_file
elif CASE == 6:
	infile_configuration_world_file = PATH + subset_of_configurations_all_US_CAN_MEX_file
	infile_configuration_ID_subset = NEW_PATH + subset_of_configurations_some_US_CAN_MEX_file
	subset_flag = 1 #If 0, output parts that are unique to the configurations in infile_configuration_ID_subset
					#If 1, output parts that are unique to the configurations NOT in infile_configuration_ID_subset
	outfile = NEW_PATH + "OUTPUT_Part_Fallout_From_US_CAN_MEX_if_only_keep_" + subset_of_configurations_some_US_CAN_MEX_file
elif CASE == 7:
	infile_configuration_world_file = NEW_PATH + subset_of_configurations_file
	infile_configuration_ID_subset = NEW_PATH + subset_of_configurations_some_INT_file
	subset_flag = 0 #If 0, output parts that are unique to the configurations in infile_configuration_ID_subset
					#If 1, output parts that are unique to the configurations NOT in infile_configuration_ID_subset
	outfile = NEW_PATH + "OUTPUT_Unique_Parts_of_" + subset_of_configurations_some_INT_file

# #Output data CSV file
# outfile_counts = NEW_PATH + "OUTPUT_Commonality_of_Parts_in_" + subset_of_configurations_file

######################################
#Columns in infile_parts
configuration_ID_column = 'Configuration_ID'
base_part_column = 'BASE_PART'

###############################################################################
#Start of Data Transformation of infiles
###############################################################################
#Subset of configurations corresponding to the "world"
myfile = open(infile_configuration_world_file,"rb")
reader = csv.reader(myfile,delimiter=',')
configurations_in_world = list(reader)
configurations_in_world = set([i[0] for i in configurations_in_world])
myfile.close()

#Store the parts corresponding to each configuration_ID
configuration_ID_to_parts_dict = {}
with open(infile_parts, "rb") as file:
	reader = csv.reader(file)
	for i,x in enumerate(reader):
		#Determine column indices
		if i == 0:
			configuration_ID_column_index = x.index(configuration_ID_column)
			base_part_column_index = x.index(base_part_column)
		#Store data in configuration_ID_to_parts_dict
		else:
			current_configuration_ID = x[configuration_ID_column_index]
			current_base_part = x[base_part_column_index]

			#Add the current_base_part to configuration_ID_to_parts_dict
			if current_configuration_ID in configuration_ID_to_parts_dict:
				configuration_ID_to_parts_dict[current_configuration_ID].add(current_base_part)
			else:
				configuration_ID_to_parts_dict[current_configuration_ID] = {current_base_part}

#Modify configuration_ID_to_parts_dict to include only the configurations in the "world"
configuration_ID_to_parts_dict = dict((k, configuration_ID_to_parts_dict[k]) for k in configurations_in_world)

configs15250 = []
with open("C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Configs_to_Part_Fallout_Analysis_04052017_Iteration2.7\\Configs_Only_US_Retail_2016.csv",'rb') as csvfile:
	reader = csv.reader(csvfile)
	for x in reader:
		configs15250.append(x[0])
configuration_ID_to_parts_dict = dict((k, configuration_ID_to_parts_dict[k]) for k in configuration_ID_to_parts_dict.keys() if k in configs15250)

# parts = []
# outfile = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\All_US_CAN_MEX_Parts_2016.csv"
# for i in configuration_ID_to_parts_dict.values():
# 	parts += i
# parts = list(set(parts))
# with open(outfile, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for i in parts:
#         writer.writerow([i])
# configuration_ID_to_parts_dict/configuration_ID_to_parts_dict

#Find the unique parts associated to the subset of configurations in infile_configuration_ID_subset
subset_all_unique_parts = []
subset_configurations = []
with open(infile_configuration_ID_subset, "rb") as file:
	reader = csv.reader(file)
	for x in reader:
		try:
			current_base_parts = configuration_ID_to_parts_dict[x[0]]
		except:
			if x == []:
				pass
			else:
				print "A configuration ID in infile_configuration_ID_subset is not in infile_parts"
				print x
				subset_all_unique_parts/subset_all_unique_parts #sys.exit()
		else:
			subset_all_unique_parts += current_base_parts
			subset_configurations.append(x[0])
subset_all_unique_parts = set(subset_all_unique_parts)


#Find the unique parts associated to the configurations not in infile_configuration_ID_subset
other_subset_all_unique_parts = []
for x in configuration_ID_to_parts_dict.keys():
	if x not in subset_configurations:
		current_base_parts = configuration_ID_to_parts_dict[x]
		other_subset_all_unique_parts += current_base_parts
other_subset_all_unique_parts = set(other_subset_all_unique_parts)


if subset_flag == 0:
	#Find the parts in subset_all_unique_parts that are not in other_subset_all_unique_parts
	output = list(subset_all_unique_parts^(subset_all_unique_parts&other_subset_all_unique_parts))
elif subset_flag == 1:
	#Find the parts in other_subset_all_unique_parts that are not in subset_all_unique_parts
	output = list(other_subset_all_unique_parts^(other_subset_all_unique_parts&subset_all_unique_parts))


# # Flatten the values of configuration_ID_to_parts_dict to one list and count
# # the number of occurences of each part
# flattened_parts = [item for sublist in configuration_ID_to_parts_dict.values() for item in sublist]
# count_of_parts_dict = Counter(flattened_parts)

# #Loop through the current_base_parts and find which parts are on all configurations in configurations_in_world
# commonality_output = []
# for part in current_base_parts:
# 	if count_of_parts_dict[part] == len(configurations_in_world):
# 		commonality_output.append(part)

###############################################################################
#Output all of the Data
###############################################################################

with open(outfile, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Unique Parts'])
    for i in output:
        writer.writerow([i])

# with open(outfile_counts, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Common Parts'])
#     for i in commonality_output:
#         writer.writerow([i])

# execfile('C:\Users\\tkress\Desktop\FCA\Parts Analysis\Unique_Parts_by_Configuration.py')