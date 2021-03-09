#Python 2.7

###############################################################################

import csv
import sys

###############################################################################
#Input Values
###############################################################################
#Provide the parts file
infile_parts = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Parts_Detail_FINAL_Exterior_Color_to_TRM_2016.csv"
# #Provide the initial configuration to volume mapping
# infile_initial_config_to_volume_mapping = "C:\\Users\\tkress\\Desktop\\FCA\\IVS\\New Part Distribution\\Total Volume Before IVS Transfer.csv"
#Provide the final configuration to volume mapping
infile_final_config_to_volume_mapping = "C:\\Users\\tkress\\Desktop\\FCA\\IVS\\New Part Distribution\\Total Volume After IVS Transfer.csv"
#Output data CSV file
outfile = "C:\\Users\\tkress\\Desktop\\FCA\\IVS\\New Part Distribution\\Part_Usage_Before_IVS_Volume_Transfer.csv"

######################################
#Columns in infile_parts
configuration_ID_column = 'Configuration_ID'
base_part_column = 'BASE_PART'
part_usage_column = 'Q_PART_USAGE'
#Columns in infile_final_config_to_volume_mapping
mapping_configuration_ID_column = 'Configuration_ID'
mapping_volume_column = 'Final Vol'

###############################################################################
#Start of Data Transformation of infiles
###############################################################################
#Read infile_initial_config_to_volume_mapping
myfile = open(infile_final_config_to_volume_mapping,"rb")
reader = csv.reader(myfile,delimiter=',')
config_to_volume_mapping = list(reader)
myfile.close()

#Create config_to_volume_dict
mapping_configuration_ID_column_index = list(config_to_volume_mapping[0]).index(mapping_configuration_ID_column)
mapping_volume_column_index = list(config_to_volume_mapping[0]).index(mapping_volume_column)
config_to_volume_dict = dict((k[mapping_configuration_ID_column_index],k[mapping_volume_column_index]) for k in config_to_volume_mapping)

#Store the parts corresponding to each configuration_ID
configuration_ID_to_parts_dict = {}
parts_to_usage_dict = {}
parts_to_configs_dict = {}
with open(infile_parts, "rb") as file:
	reader = csv.reader(file)
	for i,x in enumerate(reader):
		#Determine column indices
		if i == 0:
			configuration_ID_column_index = x.index(configuration_ID_column)
			base_part_column_index = x.index(base_part_column)
			part_usage_column_index = x.index(part_usage_column)
		#Store data in configuration_ID_to_parts_dict
		else:
			current_configuration_ID = x[configuration_ID_column_index]
			current_base_part = x[base_part_column_index]
			current_part_usage = float(x[part_usage_column_index])		

			# #Add the current_base_part to configuration_ID_to_parts_dict
			# if current_configuration_ID in configuration_ID_to_parts_dict:
			# 	configuration_ID_to_parts_dict[current_configuration_ID].add(current_base_part)
			# else:
			# 	configuration_ID_to_parts_dict[current_configuration_ID] = {current_base_part}

			# #Add the current_part_usage to parts_to_usage_dict
			# if current_base_part not in parts_to_usage_dict:
			# 	parts_to_usage_dict[current_base_part] = current_part_usage

			#Add the current_base_part to parts_to_configs_dict
			if current_configuration_ID in config_to_volume_dict.keys():
				if current_base_part in parts_to_configs_dict:
					parts_to_configs_dict[current_base_part].add(current_configuration_ID)
				else:
					parts_to_configs_dict[current_base_part] = {current_configuration_ID}

# #Find the final part usages
# output = {}
# for final_configuration_ID in config_to_volume_dict.keys():
# 	if final_configuration_ID != mapping_configuration_ID_column:
# 		final_volume = int(config_to_volume_dict[final_configuration_ID])
# 		current_base_parts = configuration_ID_to_parts_dict[final_configuration_ID]
# 		for part in current_base_parts:
# 			final_part_usage = final_volume*parts_to_usage_dict[part]
# 			if part in output:
# 				output[part] += final_part_usage
# 			else:
# 				output[part] = final_part_usage

# outfile = "C:\\Users\\tkress\\Desktop\\FCA\\IVS\\New Part Distribution\\Parts to Q_PART_USAGE.csv"
# with open(outfile, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Base Part','Q_PART_USAGE'])
#     for i in parts_to_usage_dict:
#         writer.writerow([i,parts_to_usage_dict[i]])
# sys.exit()

# outfile = "C:\\Users\\tkress\\Desktop\\FCA\\IVS\\New Part Distribution\\Final Configurations to Parts Grid.csv"
# all_parts = output.keys()
# all_parts_dict = dict((k, all_parts.index(k)) for k in all_parts)
# parts_output = []
# for final_configuration_ID in config_to_volume_dict.keys():
# 	if final_configuration_ID != mapping_configuration_ID_column:
# 		current_parts = configuration_ID_to_parts_dict[final_configuration_ID]
# 		current_output = len(all_parts)*[0]
# 		for part in current_parts:
# 			current_output[all_parts_dict[part]] = 1
# 		parts_output.append([final_configuration_ID] + current_output)

# with open(outfile, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Configuration_ID'] + all_parts)
#     for i in parts_output:
#         writer.writerow(i)
# sys.exit()

outfile = "C:\\Users\\tkress\\Desktop\\FCA\\IVS\\New Part Distribution\\Final Parts to Configurations Mapping.csv"
with open(outfile, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Base_Part', 'Configs'])
    for i in parts_to_configs_dict:
        writer.writerow([i]+list(parts_to_configs_dict[i]))
sys.exit()

###############################################################################
#Output all of the Data
###############################################################################

with open(outfile, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Base Part','Part Usage'])
    for i in output:
        writer.writerow([i,output[i]])

# execfile('C:\Users\\tkress\Desktop\FCA\IVS\New Part Distribution\Part_Usage.py')