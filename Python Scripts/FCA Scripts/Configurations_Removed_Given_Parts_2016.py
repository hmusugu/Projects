#Python 2.7

###############################################################################
#Input Values
###############################################################################
#Input data CSV file
infile_parts = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Parts_Detail_FINAL_2016.csv"
#Input data CSV file
infile_removed_base_parts = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Removed_Base_Parts.csv"
#Input data CSV file
infile_Config_ID_to_Configuration = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Long Spec VINs to Configuration IDs_3 Fields.csv"
#Input data CSV file
infile_optional_content = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Optional Content Codes 02152017v2.csv"
#Output data CSV file
outfile = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Configurations_Removed_by_Base_Part_2016.csv"

######################################
#Columns in infile_parts
configuration_ID_column = 'Configuration_ID'
base_part_column = 'BASE_PART'
volume_column = 'VOLUME'
#Columns in infile_removed_base_parts
removed_parts_column = 'Removed Parts'
removed_parts_desc_column = 'Part Desc'
removed_per_unit_price_column = 'Per Unit Price'
#Columns in infile_Config_ID_to_Configuration
mapping_configuration_ID_column = 'Configuration ID'
mapping_configuration_column = 'Configuration'

#Threshold percentage for parts on less than volume_parameter of VINs
volume_parameter = [0.01,0.05,0.10,1.00]

###############################################################################

import csv
import sys
import time
from collections import Counter

###############################################################################
#Start of Data Transformation of infiles
###############################################################################
start = time.time()
#Store the configuration IDs corresponding to each base_part
base_part_to_configs_dict = {}
#Store the part volumes corresponding to each base_part
base_part_to_volume_dict = {}
#Store the part volumes corresponding to each configuration_ID
configuration_ID_to_volume_dict = {}
with open(infile_parts, "rb") as file:
	reader = csv.reader(file)
	for i,x in enumerate(reader):
		#Determine column indices
		if i == 0:
			configuration_ID_column_index = x.index(configuration_ID_column)
			base_part_column_index = x.index(base_part_column)
			volume_column_index = x.index(volume_column)
		#Iterate through parts
		else:
			current_configuration_ID = x[configuration_ID_column_index]
			current_base_part = x[base_part_column_index]
			current_volume = int(x[volume_column_index])

			if '2016_' in current_configuration_ID:
				#Add the current_configuration_ID to base_part_to_configs_dict
				if current_base_part in base_part_to_configs_dict:
					base_part_to_configs_dict[current_base_part].append(current_configuration_ID)
				else:
					base_part_to_configs_dict[current_base_part] = [current_configuration_ID]

				#Add the current_volume to base_part_to_volume_dict
				if current_base_part in base_part_to_volume_dict:
					base_part_to_volume_dict[current_base_part] += current_volume
				else:
					base_part_to_volume_dict[current_base_part] = current_volume

				#Add the current_volume to configuration_ID_to_volume_dict
				if current_configuration_ID not in configuration_ID_to_volume_dict:
					configuration_ID_to_volume_dict[current_configuration_ID] = current_volume

#Check that the current_configuration_ID isn't added to base_part_to_configs_dict[current_base_part] more than once
for key in base_part_to_configs_dict.keys():
	current_configuration_list = base_part_to_configs_dict[key]
	unique_current_configuration_list = list(set(current_configuration_list))
	if len(current_configuration_list) != len(unique_current_configuration_list):
		#Update part counts to remove duplicate counting
		current_counts_dict = Counter(current_configuration_list)
		for current_configuration_ID in unique_current_configuration_list:
			current_count = current_counts_dict[current_configuration_ID]
			if current_count > 1:
				base_part_to_volume_dict[key] -= (configuration_ID_to_volume_dict[current_configuration_ID])*(current_count - 1)

		#Update list to be the unique configuration IDs corresponding to each base part
		base_part_to_configs_dict[key] = unique_current_configuration_list
end = time.time()
print (end-start)


#Calculate lowest 1% volume threshold for parts
total_volume = sum(configuration_ID_to_volume_dict.values())
volume_threshold = [i*total_volume for i in volume_parameter]


start = time.time()
#Return the configurations that are removed when a part is removed
output = [[configuration_ID_column]] + [[x] for x in configuration_ID_to_volume_dict.keys()]
output_indices_dict = dict((x[0],i) for i,x in enumerate(output))
with open(infile_removed_base_parts, "rb") as file:
	reader = csv.reader(file)
	for i,x in enumerate(reader):
		if i == 0:
			removed_parts_column_index = x.index(removed_parts_column)
			removed_parts_desc_column_index = x.index(removed_parts_desc_column)
			removed_per_unit_price_column_index = x.index(removed_per_unit_price_column)
		if i != 0:
			current_base_part = x[removed_parts_column_index]
			current_base_part_desc = x[removed_parts_desc_column_index]
			current_base_part_price_per_unit = x[removed_per_unit_price_column_index]
			try:
				current_part_volume = base_part_to_volume_dict[current_base_part]
				current_volume_comparison = [j>current_part_volume for j in volume_threshold]
				current_percentage_band = int(volume_parameter[[j for j,k in enumerate(current_volume_comparison) if k][0]]*100)
			except:
				pass
			else:
				current_configuration_list = base_part_to_configs_dict[current_base_part]
				for i in range(len(output)):
					if i == 0:
						output[i].append(current_base_part + ' (' + current_base_part_desc + ': ' + current_base_part_price_per_unit + ': Take rate less than ' + str(current_percentage_band) + '%: ')
					else:
						output[i].append('')
				for current_configuration_ID in current_configuration_list:
					output[output_indices_dict[current_configuration_ID]][-1] = 1
end = time.time()
print (end-start)


start = time.time()
#Read infile_optional_content
myfile = open(infile_optional_content,"rb")
reader = csv.reader(myfile,delimiter=',')
optional_content_column = list(reader)
optional_content_column = [x[0] for i,x in enumerate(optional_content_column) if i != 0]
myfile.close()


#Calculate the commonality of Configuration IDs for each part
#Create a dictionary of Configuration IDs to Configuration
configuration_ID_to_configuration_dict = {}
with open(infile_Config_ID_to_Configuration, "rb") as file:
	reader = csv.reader(file)
	for i,x in enumerate(reader):
		if i == 0:
			mapping_configuration_ID_column_index = x.index(mapping_configuration_ID_column)
			mapping_configuration_column_index = x.index(mapping_configuration_column)
		if i != 0:
			if '2016_' in x[mapping_configuration_ID_column_index]:
				configuration_ID_to_configuration_dict[x[mapping_configuration_ID_column_index]] = x[mapping_configuration_column_index]

for i,x in enumerate(output[0]):
	if i != 0:
		current_base_part = x[0:x.index(' ')]
		current_configuration_IDs_list = base_part_to_configs_dict[current_base_part]
		current_commonality_codes = []
		for j,current_configuration_ID in enumerate(current_configuration_IDs_list):
			current_configuration = configuration_ID_to_configuration_dict[current_configuration_ID]
			current_commonality_codes.append(current_configuration.split(','))
			if len(current_commonality_codes) > 100 or j == (len(current_configuration_IDs_list)-1):
				current_commonality_codes = [list(set(current_commonality_codes[0]).intersection(*current_commonality_codes))]
				if current_commonality_codes == [[]]:
					current_commonality_codes = [['NONE']]
					break

		optional_current_commonality_codes = []
		for current_code in current_commonality_codes[0]:
			if current_code in optional_content_column:
				optional_current_commonality_codes.append(current_code)
		if optional_current_commonality_codes == []:
			optional_current_commonality_codes = ['NONE']

		output[0][i] += ','.join(optional_current_commonality_codes) + ')' #+= ','.join(current_commonality_codes[0]) + ')'

end = time.time()
print (end-start)

###############################################################################
#Output all of the Data
###############################################################################

# with open(outfile, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for i in output:
#         writer.writerow(i)

###############################################################################
#Post processing requests of the data
###############################################################################

# #Output the base parts that have volume percentages less than 1%

# #Output data CSV file
# outfile_parts_to_configs = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Parts_to_Configurations_Less_1_Take_Rate.csv"

# with open(outfile_parts_to_configs, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Base Part','Configuration_IDs','Count of Configuration_IDs','Volume'])
#     for x in base_part_to_configs_dict.keys():
#     	if base_part_to_volume_dict[x] <= volume_threshold[0]:
#     		current_output = [x,','.join(base_part_to_configs_dict[x]),len(base_part_to_configs_dict[x]),base_part_to_volume_dict[x]]
#         	writer.writerow(current_output)

###############################################################################

#Output answers to the following questions:
# 1.How many total parts are on the KL program for ALL markets?
# 2.How many total parts are on the KL program for ALL markets EXCLUDING Trailhawk?
# 3.How many of those parts are UNIQUE to NA markets
# 4.How many of those parts are UNIQUE to International markets
# 5.How many of those parts are UNIQUE to International markets EXCLUDING Trailhawk?
# 6.How many parts are on EVERY KL configuration?
# 7.How many parts are UNIQUE for each drive type?
# 8.How many of those parts are UNIQUE to each region?
# 9.How many parts are UNIQUE for each International country?

#Input data CSV file
infile_Config_ID_to_Trim_Level = "C:\\Users\\tkress\\Desktop\\FCA\\Tableau Dashboard\\Configuration_ID_to_Trim_Level_no_Drive_and_CPOS.csv"
#Input data CSV file
infile_VIN_to_Country = "C:\\Users\\tkress\\Desktop\\FCA\\Tableau Dashboard\\VIN_to_Country.csv"
#Output data CSV file
outfile_part_count_requests = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Part_Count_Requests_2016_03232017.csv"

#Columns in infile_Config_ID_to_Trim_Level
mapping_CT_configuration_ID_column = 'Configuration ID'
mapping_CT_trim_level_column = 'Trim Level (no drive)'
mapping_CT_CPOS_column = 'CPOS'

#Columns in infile_VIN_to_Country
mapping_VC_configuration_ID_column = 'Configuration ID'
mapping_VC_country_column = 'Country Name'
mapping_VC_VIN_column = 'Vin 8'
mapping_VC_region_column = 'Region (Set)'

#Create a dictionary of Configuration IDs to Trim Level and a dictionary of Configuration IDs to CPOS
configuration_ID_to_trim_level_dict = {}
configuration_ID_to_CPOS_dict = {}
with open(infile_Config_ID_to_Trim_Level, "rb") as file:
	reader = csv.reader(file)
	for i,x in enumerate(reader):
		if i == 0:
			mapping_CT_configuration_ID_column_index = x.index(mapping_CT_configuration_ID_column)
			mapping_CT_trim_level_column_index = x.index(mapping_CT_trim_level_column)
			mapping_CT_CPOS_column_index = x.index(mapping_CT_CPOS_column)
		if i != 0:
			if '2016_' in x[mapping_CT_configuration_ID_column_index]:
				configuration_ID_to_trim_level_dict[x[mapping_CT_configuration_ID_column_index]] = x[mapping_CT_trim_level_column_index]
				configuration_ID_to_CPOS_dict[x[mapping_CT_configuration_ID_column_index]] = x[mapping_CT_CPOS_column_index]

#Create a dictionary of Configuration IDs to Country and a dictionary of Configuration IDs to region
configuration_ID_to_Country_dict = {}
configuration_ID_to_region_dict = {}
with open(infile_VIN_to_Country, "rb") as file:
	reader = csv.reader(file)
	for i,x in enumerate(reader):
		if i == 0:
			mapping_VC_configuration_ID_column_index = x.index(mapping_VC_configuration_ID_column)
			mapping_VC_country_column_index = x.index(mapping_VC_country_column)
			mapping_VC_VIN_column_index = x.index(mapping_VC_VIN_column)
			mapping_VC_region_column_index = x.index(mapping_VC_region_column)
		if i != 0:
			if '2016_' in x[mapping_VC_configuration_ID_column_index]:
				try:
					configuration_ID_to_Country_dict[x[mapping_VC_configuration_ID_column_index]].append(x[mapping_VC_country_column_index])
				except:
					configuration_ID_to_Country_dict[x[mapping_VC_configuration_ID_column_index]] = [x[mapping_VC_country_column_index]]
				try:
					configuration_ID_to_region_dict[x[mapping_VC_configuration_ID_column_index]].append(x[mapping_VC_region_column_index])
				except:
					configuration_ID_to_region_dict[x[mapping_VC_configuration_ID_column_index]] = [x[mapping_VC_region_column_index]]
for i in configuration_ID_to_Country_dict.keys():
	configuration_ID_to_Country_dict[i] = list(set(configuration_ID_to_Country_dict[i]))
for i in configuration_ID_to_region_dict.keys():
	configuration_ID_to_region_dict[i] = list(set(configuration_ID_to_region_dict[i]))

#Answer for 1.
total_parts = 0
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	total_parts += 1
#Answer for 2.
total_parts_trailhawk_only = 0
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	if all([configuration_ID_to_trim_level_dict[j] == 'TRAILHAWK' for j in current_configuration_IDs_list]):
		total_parts_trailhawk_only += 1
total_parts_excluding_trailhawk = total_parts - total_parts_trailhawk_only
#Answer for 3.
total_parts_NA = 0
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	if all(['_US_' in j for j in current_configuration_IDs_list]):
		total_parts_NA += 1
#Answer for 4.
total_parts_international = 0
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	if all(['_INT_' in j for j in current_configuration_IDs_list]):
		total_parts_international += 1
#Answer for 5.
total_parts_international_trailhawk_only = 0
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	if all(['_INT_' in j and configuration_ID_to_trim_level_dict[j] == 'TRAILHAWK' for j in current_configuration_IDs_list]):
		total_parts_international_trailhawk_only += 1
total_parts_international_excluding_trailhawk = total_parts_international - total_parts_international_trailhawk_only
#Answer for 6.
common_parts = 0
number_of_configuration_IDs = 0
for i in configuration_ID_to_configuration_dict.keys():
	number_of_configuration_IDs += 1
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	if len(current_configuration_IDs_list) == number_of_configuration_IDs:
		common_parts += 1
#Answer for 7.
unique_drive_parts = {}
CPOS_to_drive_dict = {'T': 'LH', 'J': 'LH', 'B': 'RH', 'U': 'RH'}
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	current_CPOS = [configuration_ID_to_CPOS_dict[j] for j in current_configuration_IDs_list]
	current_drives = [CPOS_to_drive_dict[j[2]] for j in current_CPOS]
	current_drives = set(current_drives)
	if len(current_drives) == 1:
		try:
			unique_drive_parts[list(current_drives)[0]] += 1
		except:
			unique_drive_parts[list(current_drives)[0]] = 1
#Answer for 8.
unique_region_parts = {}
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	current_regions = [configuration_ID_to_region_dict[j] for j in current_configuration_IDs_list]
	current_regions = set([item for sublist in current_regions for item in sublist])
	if len(current_regions) == 1:
		try:
			unique_region_parts[list(current_regions)[0]] += 1
		except:
			unique_region_parts[list(current_regions)[0]] = 1
#Answer for 9.
unique_country_parts = {}
for i in base_part_to_configs_dict.keys():
	current_configuration_IDs_list = base_part_to_configs_dict[i]
	current_countries = [configuration_ID_to_Country_dict[j] for j in current_configuration_IDs_list]
	current_countries = set([item for sublist in current_countries for item in sublist])
	if len(current_countries) == 1 and list(current_countries)[0] not in ['USA','Canada','Mexico']:
		try:
			unique_country_parts[list(current_countries)[0]] += 1
		except:
			unique_country_parts[list(current_countries)[0]] = 1

with open(outfile_part_count_requests, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    unique_drive_parts_keys = ['Unique Parts in ' + str(i) for i in unique_drive_parts.keys()]
    unique_drive_parts_values = unique_drive_parts.values()
    unique_country_parts_keys = ['Unique Parts in ' + str(i) for i in unique_country_parts.keys()]
    unique_country_parts_values = unique_country_parts.values()
    unique_region_parts_keys = ['Unique Parts in ' + str(i) for i in unique_region_parts.keys()]
    unique_region_parts_values = unique_region_parts.values()
    writer.writerow(['Model Year','Total Parts','Total Parts Excluding Trailhawk','Total Parts Unique NA','Total Parts Unique International','Total Parts International Excluding Trailhawk','Common Parts'] + unique_drive_parts_keys + unique_region_parts_keys + unique_country_parts_keys)
    writer.writerow(['2016',total_parts,total_parts_excluding_trailhawk,total_parts_NA,total_parts_international,total_parts_international_excluding_trailhawk,common_parts] + unique_drive_parts_values + unique_region_parts_values + unique_country_parts_values)

###############################################################################

# #Output the MY and VIN count for the given parts

# #Input data CSV file
# infile_specific_parts = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Take Rate - All Parts_03222017_v1.csv"
# #Output data CSV file
# outfile_specific_parts = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\Take Rate - All Parts_03222017_v1_with_MY_and_VIN_counts.csv"

# output_specific_parts = []
# with open(infile_specific_parts, "rb") as file:
# 	reader = csv.reader(file)
# 	for x in reader:
# 		current_base_part = x[0]
# 		try:
# 			current_configuration_IDs_list = base_part_to_configs_dict[current_base_part]
# 		except:
# 			output_specific_parts.append([current_base_part, "NO INFO"])
# 		else:
# 			current_volume_2016 = 0
# 			for current_configuration_ID in current_configuration_IDs_list:
# 				if '2016_' in current_configuration_ID:
# 					current_volume_2016 += configuration_ID_to_volume_dict[current_configuration_ID]
# 			output_specific_parts.append([current_base_part, current_volume_2016])

# with open(outfile_specific_parts, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Base Part','Volume 2016'])
#     for i in output_specific_parts:
#     	writer.writerow(i)

###############################################################################

# #Average number of parts per Configuration ID
# all_configuration_lists = base_part_to_configs_dict.values()
# all_configuration_lists = [item for sublist in all_configuration_lists for item in sublist]
# configuration_ID_to_part_count_dict = Counter(all_configuration_lists) 

# total_parts = 0
# length = 0
# for i in configuration_ID_to_part_count_dict.keys():
# 	total_parts += configuration_ID_to_part_count_dict[i]
# 	length += 1
# average_parts_per_configuration_ID = float(total_parts)/float(length)
# print average_parts_per_configuration_ID

# #How many parts map to only 1 Configuration ID
# count = 0
# for i in base_part_to_configs_dict.keys():
# 	if len(base_part_to_configs_dict[i]) == 1:
# 		count += 1
# print count

###############################################################################

# #Output the count of VINs for each part used on Configuration IDs

# #Output data CSV file
# outfile_VIN_count = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\VIN_Count_for_Parts_on_2016_Configurations.csv"

# #Which parts are UNIQUE to International markets
# total_parts_unique_international = []
# for i in base_part_to_configs_dict.keys():
# 	current_configuration_IDs_list = base_part_to_configs_dict[i]
# 	if all(['_INT_' in j for j in current_configuration_IDs_list]):
# 		total_parts_unique_international += [i]

# #Which parts are UNIQUE to Trailhawk
# total_parts_unique_trailhawk = []
# for i in base_part_to_configs_dict.keys():
# 	current_configuration_IDs_list = base_part_to_configs_dict[i]
# 	if all([configuration_ID_to_trim_level_dict[j] == 'TRAILHAWK' for j in current_configuration_IDs_list]):
# 		total_parts_unique_trailhawk += [i]

# #Output data
# output_VIN_count = []
# for i in base_part_to_volume_dict.keys():
# 	if i in total_parts_unique_international:
# 		international_flag = 'Yes'
# 	else:
# 		international_flag = 'No'
# 	if i in total_parts_unique_trailhawk:
# 		trailhawk_flag = 'Yes'
# 	else:
# 		trailhawk_flag = 'No'
# 	output_VIN_count.append([i,base_part_to_volume_dict[i],international_flag,trailhawk_flag])

# with open(outfile_VIN_count, 'ab') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Base Part','VIN Count','Within Set of "Total Parts Unique International"?','Within Set of "Total Parts Unique Trailhawk"?'])
#     for i in output_VIN_count:
#     	writer.writerow(i)

# execfile('C:\Users\\tkress\Desktop\FCA\Parts Analysis\Configurations_Removed_Given_Parts_2016.py')