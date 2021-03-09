#Python 2.7

###############################################################################
#Input Values
###############################################################################
#Input data CSV file - VINs with Configuration Strings/IDs
infile_data = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Long Spec VINs to Configuration IDs.csv"
#Input data CSV file - Code/Description to Grouping Mapping
infile_grouping = "C:\\Users\\tkress\\Desktop\\FCA\\Configuration ID to Options Columns\\Options_Grouping.csv"
# #Input data CSV file - CPOS to Engine/Transmission
# infile_CPOS_to_Engine_Transmission = "C:\\Users\\tkress\\Desktop\\FCA\\Configuration ID to Options Columns\\CPOS to Engine Transmission.csv"
#Input data CSV file - Optional/standard codes in Long Spec that define a Configuration
infile_optional_content_codes = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Content Codes 02232017v2.csv"
#Output data CSV file - Configuration IDs with Options and Grouping Columns
outfile = "C:\\Users\\tkress\\Desktop\\FCA\\Configuration ID to Options Columns\\Configuration_IDs_with_Options_and_Groupings_test.csv"

######################################
#Columns in infile_data
configuration_column = 'Configuration'
configuration_ids_column = 'Configuration ID'
CPOS_column = 'CPOS'
Master_Source_column = 'Master Source'
MY_column = 'Model Year'
######################################
#Columns in infile_grouping
grouping_desc_and_code_column = 'Desc/Code Labels'
grouping_options_group_column = 'Options Group'
######################################
# #Columns in infile_CPOS_to_Engine_Transmission
# CET_Master_Source_column = 'Master Source'
# CET_MY_column = 'Model Year'
# CET_CPOS_column = 'CPOS'
# CET_Engine_column = 'Engine'
# CET_Transmission_column = 'Transmission'
######################################
#Column names in infile_optional_content_codes
Optional_Content_codes_column = 'Code'
Optional_Content_desc_column = 'Desc'

###############################################################################

import csv
import sys

###############################################################################
#Start of Data Transformation of infiles
###############################################################################

#Read infile_data
myfile = open(infile_data,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix = list(reader)
myfile.close()

#Determine column indices of inputs
configuration_column_index = input_matrix[0].index(configuration_column)
configuration_ids_column_index = input_matrix[0].index(configuration_ids_column)
CPOS_column_index = input_matrix[0].index(CPOS_column)
Master_Source_column_index = input_matrix[0].index(Master_Source_column)
MY_column_index = input_matrix[0].index(MY_column)

#Configuration strings and IDs columns
configuration = [i[configuration_column_index] for i in input_matrix]
configuration_ids = [i[configuration_ids_column_index] for i in input_matrix]

# #Read infile_CPOS_to_Engine_Transmission
# myfile = open(infile_CPOS_to_Engine_Transmission,"rb")
# reader = csv.reader(myfile,delimiter=',')
# input_matrix_eng_trans = list(reader)
# myfile.close()

# #Determine column indices of inputs
# CET_Master_Source_column_index = input_matrix_eng_trans[0].index(CET_Master_Source_column)
# CET_MY_column_index = input_matrix_eng_trans[0].index(CET_MY_column)
# CET_CPOS_column_index = input_matrix_eng_trans[0].index(CET_CPOS_column)
# CET_Engine_column_index = input_matrix_eng_trans[0].index(CET_Engine_column)
# CET_Transmission_column_index = input_matrix_eng_trans[0].index(CET_Transmission_column)

# #Create (Master_Source,MY,CPOS) to [Engine,Transmission] dict
# CET_dict = {}
# for i in input_matrix_eng_trans:
#     CET_dict[(i[CET_Master_Source_column_index],i[CET_MY_column_index],i[CET_CPOS_column_index])] = [i[CET_Engine_column_index],i[CET_Transmission_column_index]]

#Read Optional Content Codes data
myfile = open(infile_optional_content_codes,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_Optional_Codes = list(reader)
myfile.close()

#Determine column indices of inputs in Optional Content data
Optional_Content_codes_column_index = input_matrix_Optional_Codes[0].index(Optional_Content_codes_column)
Optional_Content_desc_column_index = input_matrix_Optional_Codes[0].index(Optional_Content_desc_column)

#Optional Content Codes
Optional_Content_codes = [i[Optional_Content_codes_column_index] for i in input_matrix_Optional_Codes if i[Optional_Content_codes_column_index] != Optional_Content_codes_column]
Optional_Content_desc = [i[Optional_Content_desc_column_index] for i in input_matrix_Optional_Codes if i[Optional_Content_desc_column_index] != Optional_Content_desc_column]
Optional_Content_desc_with_codes = [i[Optional_Content_desc_column_index] + '(' + i[Optional_Content_codes_column_index] + ')'  for i in input_matrix_Optional_Codes if i[Optional_Content_desc_column_index] != Optional_Content_desc_column]

#Read grouping data
myfile = open(infile_grouping,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_grouping = list(reader)
myfile.close()

#Determine column indices of inputs
grouping_desc_and_code_column_index = input_matrix_grouping[0].index(grouping_desc_and_code_column)
grouping_options_group_column_index = input_matrix_grouping[0].index(grouping_options_group_column)

#Separate the configuration code/desc and corresponding grouping names
input_matrix_grouping_code_and_desc = [i[grouping_desc_and_code_column_index] for i in input_matrix_grouping[1:len(input_matrix_grouping)]]
input_matrix_grouping_code = [x[-4:-1] for i,x in enumerate(input_matrix_grouping_code_and_desc)]
input_matrix_grouping_desc = [x[0:-6] for i,x in enumerate(input_matrix_grouping_code_and_desc)]
input_matrix_grouping_name = [i[grouping_options_group_column_index] for i in input_matrix_grouping[1:len(input_matrix_grouping)]]
grouping_column_names_unique = list(set(input_matrix_grouping_name))

#Group to codes dictionary
group_to_codes_dict = dict((k,[x for i,x in enumerate(input_matrix_grouping_code) if input_matrix_grouping_name[i] == k]) for k in input_matrix_grouping_name)

#Add options and grouping columns
bad_groupings = []
output_dict = {}
output_dict_format = Optional_Content_codes + grouping_column_names_unique
for i,x in enumerate(configuration_ids):
    if i != 0:
        if x not in output_dict:
            current_configuration = configuration[i]
            current_configuration_list = [k.strip() for k in current_configuration.split(',')]
            # #Add in Engine/Transmission to configuration strings
            # current_eng_trans = CET_dict[(input_matrix[i][Master_Source_column_index],input_matrix[i][MY_column_index],input_matrix[i][CPOS_column_index])]
            # current_configuration_list = current_configuration_list + current_eng_trans
            current_output = len(output_dict_format)*['']
            #Codes that are part of the configuration string (Options and Groups)
            for j in current_configuration_list:
                #Codes that have groups
                if j in input_matrix_grouping_code:
                    #Fill in options columns
                    current_output[output_dict_format.index(j)] = 1 #input_matrix_grouping_desc[input_matrix_grouping_code.index(j)]
                    #Fill in grouping columns
                    current_group = input_matrix_grouping_name[input_matrix_grouping_code.index(j)]
                    #Check that there is no overlap when grouping columns
                    if current_output[output_dict_format.index(current_group)] != '':
                        print 'Configuration ID ' + x + ' has at least two codes that are in the same group.'
                        #sys.exit()
                        bad_groupings.append([x, current_output[output_dict_format.index(current_group)], input_matrix_grouping_code_and_desc[input_matrix_grouping_code.index(j)]])
                    else:
                        current_output[output_dict_format.index(current_group)] = input_matrix_grouping_code_and_desc[input_matrix_grouping_code.index(j)]
                #Codes that don't have groups
                else:
                    #Fill in options columns
                    current_output[output_dict_format.index(j)] = 1 #Optional_Content_desc[Optional_Content_codes.index(j)]

            output_dict[x] = current_output

#Output bad_groupings if it exists
if bad_groupings == []:
    pass
else:
    outfile_bad = "C:\\Users\\tkress\\Desktop\\FCA\\Configuration ID to Options Columns\\Bad_Groupings.csv"
    bad_groupings_subset = [i[1:] for i in bad_groupings]
    unique_bad_groupings_subset = []
    for i in bad_groupings_subset:
        if i not in unique_bad_groupings_subset:
            unique_bad_groupings_subset.append(i)
    bad_groupings_with_sample_config = {}
    for i in bad_groupings:
        current_bad_group = i[1:]
        if current_bad_group in unique_bad_groupings_subset:
            if tuple(current_bad_group) not in bad_groupings_with_sample_config:
                bad_groupings_with_sample_config[tuple(current_bad_group)] = i
    with open(outfile_bad, 'ab') as file:
        writer = csv.writer(file, delimiter = ',')
        writer.writerow(['Sample Config_ID','Option 1','Option 2'])
        for i in bad_groupings_with_sample_config.values():
            writer.writerow(i)
    print 'The normal output did not finish. Bad groupings have been output instead.'
    sys.exit()

###############################################################################
#Output all of the Data
###############################################################################

with open(outfile, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Configuration ID'] + Optional_Content_desc_with_codes + grouping_column_names_unique)
    output_dict_keys = output_dict.keys()
    for i in output_dict_keys:
        current_output = [i] + output_dict[i]     
        writer.writerow(current_output)

# execfile('C:\Users\\tkress\Desktop\FCA\Configuration ID to Options Columns\Configuration_ID_to_Options_Columns.py')