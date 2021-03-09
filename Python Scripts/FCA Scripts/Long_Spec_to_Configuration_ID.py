#Python 2.7

###############################################################################
#Input Values
###############################################################################
#Input data CSV file - Long Spec Data
infile_Long_Spec_data = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Long Spec Compiled.csv"
#Input data CSV file - Master Data
infile_Master_data = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\VIN Master Data_v2.csv"
#Input data CSV file - Optional/standard codes in Long Spec that define a Configuration
infile_optional_content_codes = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Content Codes 02232017v2.csv"
#Output data CSV file
outfile = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Long Spec VINs to Configuration IDs.csv"
#Output data CSV file
outfile_Config_ID_to_Long_Spec = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Long Spec to Config_ID Mapping.csv"

#Column names in infile_Long_Spec_data
LS_VIN_Last8_column = 'Last 8 Digits of VIN'
LS_sales_codes_column = 'Sales Codes'
#Column names in infile_Master_data
Master_VIN_Last8_column = 'VIN Last 8'
Master_MY_column = 'ModelYear'
Master_CPOS_column = 'CPOS'
Master_Source_column = 'Data_Source'
#Column names in infile_optional_content_codes
Optional_Content_codes_column = 'Code'

###############################################################################

import csv

###############################################################################
#Start of Data Transformation of infiles
###############################################################################

#Read Long Spec data
myfile = open(infile_Long_Spec_data,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_Long_Spec = list(reader)
myfile.close()

#Read Master data
myfile = open(infile_Master_data,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_Master = list(reader)
myfile.close()

#Read Optional Content Codes data
myfile = open(infile_optional_content_codes,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_Optional_Codes = list(reader)
myfile.close()

#Determine column indices of inputs in Long Spec data
LS_VIN_Last8_column_index = input_matrix_Long_Spec[0].index(LS_VIN_Last8_column)
LS_sales_codes_column_index = input_matrix_Long_Spec[0].index(LS_sales_codes_column)
#Determine column indices of inputs in Master data
Master_VIN_Last8_column_index = input_matrix_Master[0].index(Master_VIN_Last8_column)
Master_MY_column_index = input_matrix_Master[0].index(Master_MY_column)
Master_CPOS_column_index = input_matrix_Master[0].index(Master_CPOS_column)
Master_Source_column_index = input_matrix_Master[0].index(Master_Source_column)
#Determine column indices of inputs in Optional Content data
Optional_Content_codes_column_index = input_matrix_Optional_Codes[0].index(Optional_Content_codes_column)

#Columns from Master data that are required
Master_VINs = [i[Master_VIN_Last8_column_index] for i in input_matrix_Master]
Master_MY = [i[Master_MY_column_index] for i in input_matrix_Master]
Master_CPOS = [i[Master_CPOS_column_index] for i in input_matrix_Master]
Master_Source = [i[Master_Source_column_index] for i in input_matrix_Master]

#Free up memory
#reset_selective -f input_matrix_Master
del input_matrix_Master

#Create Long Spec dictionary
LS_Dict = dict(input_matrix_Long_Spec)

#Optional Content Codes
Optional_Content_codes = [i[Optional_Content_codes_column_index] for i in input_matrix_Optional_Codes if i[Optional_Content_codes_column_index] != Optional_Content_codes_column]

#Loop through Long Spec data to find Configuration string and ID
output_matrix = []
output_Config_ID_to_Long_Spec = []
Config_ID_dict = {}
counter_2015_US = counter_2016_US = counter_2015_CAN = counter_2016_CAN = counter_2015_MEX = counter_2016_MEX = counter_2015_INT = counter_2016_INT = 1
for i,x in enumerate(Master_VINs):
    if i != 0 and Master_MY[i] != '2014':
        #Find Configuration string
        current_LS_string = LS_Dict[x]
        current_output = []
        for j in Optional_Content_codes:
            if j in current_LS_string:
                current_output.append(j)
        current_output.sort()
        current_output = ','.join(current_output)
        
        #Find Configuration ID
        if (Master_MY[i],Master_CPOS[i],Master_Source[i],current_output) not in Config_ID_dict:
            if Master_MY[i] == '2015':
                if Master_Source[i] == 'US_Retail' or Master_Source[i] == 'Fleet':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2015' + '_US_' + str(counter_2015_US)
                    counter_2015_US += 1
                if Master_Source[i] == 'CAN':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2015' + '_CAN_' + str(counter_2015_CAN)
                    counter_2015_CAN += 1
                if Master_Source[i] == 'MEX':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2015' + '_MEX_' + str(counter_2015_MEX)
                    counter_2015_MEX += 1
                if Master_Source[i] == 'INT':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2015' + '_INT_' + str(counter_2015_INT)
                    counter_2015_INT += 1
            elif Master_MY[i] == '2016':
                if Master_Source[i] == 'US_Retail' or Master_Source[i] == 'Fleet':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2016' + '_US_' + str(counter_2016_US)
                    counter_2016_US += 1
                if Master_Source[i] == 'CAN':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2016' + '_CAN_' + str(counter_2016_CAN)
                    counter_2016_CAN += 1
                if Master_Source[i] == 'MEX':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2016' + '_MEX_' + str(counter_2016_MEX)
                    counter_2016_MEX += 1
                if Master_Source[i] == 'INT':
                    Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)] = '2016' + '_INT_' + str(counter_2016_INT)
                    counter_2016_INT += 1
                    
        #Store the output data
        output_matrix.append([x, Master_CPOS[i], Master_Source[i], Master_MY[i], current_output, Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)]])

        #Store the Config_ID to Long Spec data
        output_Config_ID_to_Long_Spec.append([Config_ID_dict[(Master_MY[i],Master_CPOS[i],Master_Source[i],current_output)], current_LS_string])
###############################################################################
#Output all of the Data
###############################################################################

with open(outfile, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['VIN_8','CPOS','Master Source','Model Year','Configuration','Configuration ID']) 
    for i in output_matrix:
        current_output = i
        writer.writerow(current_output)

with open(outfile_Config_ID_to_Long_Spec, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Configuration ID','Long Spec']) 
    for i in output_Config_ID_to_Long_Spec:
        current_output = i
        writer.writerow(current_output)

# execfile('C:\Users\\tkress\Desktop\FCA\Long Spec\Long_Spec_to_Configuration_ID.py')