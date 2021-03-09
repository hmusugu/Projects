#Python 2.7

###############################################################################
#Input Values
###############################################################################
#Input data CSV file - IDR DOLA Data
infile_data = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\Compiled Cherokee_v7.csv"
#Input data CSV file - Code to Description Mapping
infile_mapping = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\KL Sales Codes with Descriptions Compiled.csv"
#Input data CSV file - Code/Description to Grouping Mapping
infile_grouping = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\Options_Grouping.csv"
#Input data CSV files - Incentive Data
infile_incentive_1 = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\IDSS_Cherokee_Incentive_Numbers_Part 1.csv"
infile_incentive_2 = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\IDSS_Cherokee_Incentive_Numbers_Part 2.csv"
infile_incentive_3 = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\IDSS_Cherokee_Incentive_Numbers_Part 3.csv"
#Input data CSV files - Cost Data
infile_cost_1 = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\klcost-part1-aggregated.csv"
infile_cost_2 = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\klcost-part2-aggregated.csv"
infile_cost_3 = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\klcost-part3-aggregated.csv"
#Input data CSV file - Base Options Data
infile_base_options = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\Base_Options_by_CPOS_02062017_v3.csv"
#Input data CSV file - Package Dependencies
infile_package_dependencies = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\Package Dependencies 02082017v2.csv"
#Output data CSV file - IDR DOLA Data with Options Columns
outfile = "C:\\Users\\tkress\\Desktop\\FCA\\IDR DOLA Data Transformed\\Compiled Cherokee_v7 with Options Columns.csv"

######################################
#VINs in infile_data
VIN_column = 'VIN'
#Configuration Code Strings in infile_data
configuration_column = 'Configuration'
#HCO Code Strings in infile_data
HCO_column = 'HCOList'
#Model Year value in infile_data
model_year_column = 'ModelYear'
#Sales Type values in infile_data
sales_type_column = 'SalesType'
#Trim Class values in infile_data
trim_class_column = 'Trim Class'
#CPOS values in infile_data
CPOS_column = 'CPOS'
#Variable Cost values in infile_data
variable_cost_column = 'VariableCost'
######################################
#KL Sales Code in infile_mapping
sales_code_column = 'Code'
#KL Sales Description in infile_mapping
sales_desc_column = 'Desc'
#KL Sales Class Code in infile_mapping
sales_class_code_column = 'Class Code'
######################################
#Desc/Code Labels in infile_grouping
grouping_desc_and_code_column = 'Desc/Code Labels'
#Options Group in infile_grouping
grouping_options_group_column = 'Options Group'
######################################
#VIN Last 8 in infile_incentive files
VIN_last_8_column = 'VIN of last 8 Characters'
#VIN First 9 in infile_incentive files
VIN_first_9_column = 'VIN of first 9 Characters'
#Incentive Class Description in infile_incentive files
incentive_class_desc_column = 'Incentive Class Desc'
######################################
#VINs in infile_cost
VIN_cost_column = 'VIN'
#Labor in infile_cost
labor_cost_column = 'LABOR'
#Burden fixed in infile_cost
burden_fixed_cost_column = 'BURD-FXD'
#Burden variable infile_cost
burden_var_cost_column = 'BURD-VAR'
#IBT in infile_cost
IBT_cost_column = 'IBT'
#Material in infile_cost
material_cost_column = 'TOTAL-MAT'
######################################
#CPOS in infile_base_options
base_CPOS_column = 'CPOS'
#Options Group in infile_base_options
base_options_group_column = 'Options Group'
#Model Year in infile_base_options
base_model_year_column = 'Model Year'
#Base Option Code in infile_base_options
base_options_code_column = 'Code'
#Base Option Description in infile_base_options
base_options_desc_column = 'Desc'
######################################
#Column names in infile_package_dependencies
package_dependencies_MY_column = 'MY'
package_dependencies_CPOS_column = 'CPOS'
package_dependencies_options_code_column = 'Package Code'
package_dependencies_options_desc_column = 'Package Desc'
package_dependencies_group_column = 'Group'
package_dependencies_base_code_column = 'Base Codes'
package_dependencies_base_desc_column = 'Base Desc'
######################################
#Warranty cost MY15
warranty_cost_value_15 = '1444'
#Warranty cost MY16
warranty_cost_value_16 = '729'
#OBT cost MY15
obt_cost_value_15 = '436'
#OBT cost MY16
obt_cost_value_16 = '406'
#LOH value
loh_value = '1871'
#Average Material as a Percent of Variable Cost
material_cost_percentage = .9184
#Average IBT as a Percent of Variable Cost
IBT_cost_percentage = .01015

###############################################################################

import numpy as np
import csv
import sys

###############################################################################
#Start of Data Transformation of infiles
###############################################################################

#Read IDR DOLA Data
myfile = open(infile_data,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix = list(reader)
myfile.close()

#Determine column indices of inputs
VIN_column_index = input_matrix[0].index(VIN_column)
configuration_column_index = input_matrix[0].index(configuration_column)
HCO_column_index = input_matrix[0].index(HCO_column)
model_year_column_index = input_matrix[0].index(model_year_column)
sales_type_column_index = input_matrix[0].index(sales_type_column)
trim_class_column_index = input_matrix[0].index(trim_class_column)
CPOS_column_index = input_matrix[0].index(CPOS_column)
variable_cost_column_index = input_matrix[0].index(variable_cost_column)

#Add Warranty, OBT, and LOH Values
input_matrix = [i+['','',''] for i in input_matrix]
input_matrix_row_len = len(input_matrix[0])
for i,x in enumerate(input_matrix):
    if i == 0:
        input_matrix[i][-3:input_matrix_row_len] = ['Warranty','OBT','LOH']
    elif x[model_year_column_index] == '2015':
        input_matrix[i][-3] = warranty_cost_value_15
        input_matrix[i][-2] = obt_cost_value_15
        input_matrix[i][-1] = loh_value
    elif x[model_year_column_index] == '2016':
        input_matrix[i][-3] = warranty_cost_value_16
        input_matrix[i][-2] = obt_cost_value_16
        input_matrix[i][-1] = loh_value

#Read Cost Data
myfile = open(infile_cost_1,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_cost_1 = list(reader)
myfile.close()

myfile = open(infile_cost_2,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_cost_2 = list(reader)
myfile.close()

myfile = open(infile_cost_3,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_cost_3 = list(reader)
myfile.close()

#Determine column indices of inputs
VIN_cost_column_index = input_matrix_cost_1[0].index(VIN_cost_column)
labor_cost_column_index = input_matrix_cost_1[0].index(labor_cost_column)
burden_fixed_cost_column_index = input_matrix_cost_1[0].index(burden_fixed_cost_column)
burden_var_cost_column_index = input_matrix_cost_1[0].index(burden_var_cost_column)
IBT_cost_column_index = input_matrix_cost_1[0].index(IBT_cost_column)
material_cost_column_index = input_matrix_cost_1[0].index(material_cost_column)

#Add the cost data to input_matrix
cost_aggregated = input_matrix_cost_1 + input_matrix_cost_2[1:len(input_matrix_cost_2)] + input_matrix_cost_3[1:len(input_matrix_cost_3)]
cost_aggregated_dict = dict((i[VIN_cost_column_index],[i[labor_cost_column_index],i[burden_fixed_cost_column_index],i[burden_var_cost_column_index],i[IBT_cost_column_index],i[material_cost_column_index]]) for i in cost_aggregated)
cost_VINs = set([i[VIN_cost_column_index] for i in cost_aggregated])

IDR_DOLA_VINs = [x[VIN_column_index] for x in input_matrix]
IDR_DOLA_VINs_dict = dict((k,i) for i,k in enumerate(IDR_DOLA_VINs))
VINs_intersection = list(cost_VINs.intersection(IDR_DOLA_VINs))
input_matrix = [i+['','','','',''] for i in input_matrix]
input_matrix_row_len = len(input_matrix[0])
input_matrix[0][input_matrix_row_len-5:input_matrix_row_len] = ['Labor','Burden Fixed','Burden Variable','IBT','Material']
for i in VINs_intersection:
    input_matrix[IDR_DOLA_VINs_dict[i]][input_matrix_row_len-5:input_matrix_row_len] = cost_aggregated_dict[i]

#Fill the blank IBT and Material data with IBT_cost_percentage and material_cost_percentage
for i,x in enumerate(input_matrix):
    #IBT blank
    if x[-2] == '0.00':
        input_matrix[i][-2] = str(IBT_cost_percentage*float(input_matrix[i][variable_cost_column_index]))
    #Material blank
    if x[-1] == '0.00':
        input_matrix[i][-1] = str(material_cost_percentage*float(input_matrix[i][variable_cost_column_index]))

#Free up memory
%reset_selective -f input_matrix_cost_1
%reset_selective -f input_matrix_cost_2
%reset_selective -f input_matrix_cost_3
%reset_selective -f cost_aggregated
%reset_selective -f cost_aggregated_dict

#Read Incentive Data
myfile = open(infile_incentive_1,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_incentive_1 = list(reader)
myfile.close()

myfile = open(infile_incentive_2,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_incentive_2 = list(reader)
myfile.close()

myfile = open(infile_incentive_3,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_incentive_3 = list(reader)
myfile.close()

#Determine column indices of inputs
VIN_last_8_column_index = input_matrix_incentive_1[0].index(VIN_last_8_column)
VIN_first_9_column_index = input_matrix_incentive_1[0].index(VIN_first_9_column)
incentive_class_desc_column_index = input_matrix_incentive_1[0].index(incentive_class_desc_column)

#Add the incentive data to input_matrix
incentive_class_aggregated = input_matrix_incentive_1 + input_matrix_incentive_2[1:len(input_matrix_incentive_2)] + input_matrix_incentive_3[1:len(input_matrix_incentive_3)]
employee_VINs = set([i[VIN_first_9_column_index]+i[VIN_last_8_column_index] for i in incentive_class_aggregated if 'Employee' in i[incentive_class_desc_column_index]])

#Free up memory
%reset_selective -f input_matrix_incentive_1
%reset_selective -f input_matrix_incentive_2
%reset_selective -f input_matrix_incentive_3
%reset_selective -f incentive_class_aggregated

#Add the incentive data to input_matrix
VINs_intersection = list(employee_VINs.intersection(IDR_DOLA_VINs))
input_matrix = [i+[''] for i in input_matrix]
input_matrix_row_len = len(input_matrix[0])
input_matrix[0][input_matrix_row_len-1] = 'Employee Incentive'
for i in VINs_intersection:
    input_matrix[IDR_DOLA_VINs_dict[i]][input_matrix_row_len-1] = 'Employee'

#Re-map the Sales Type codes
for i in input_matrix:
    if i[sales_type_column_index] == '1':
        i[sales_type_column_index] = 'Direct Retail Sale'
    elif i[sales_type_column_index] == 'B':
        i[sales_type_column_index] = 'Business Retail Sale'
    elif i[sales_type_column_index] == 'E':
        i[sales_type_column_index] = 'Business Retail Lease'
    elif i[sales_type_column_index] == 'L':
        i[sales_type_column_index] = 'Retail Lease'


##################### UPDATE BASED ON BASE OPTIONS DATA AND PACKAGE DEPENDENCIES #####################
#Read base options data
myfile = open(infile_base_options,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_base_options = list(reader)
myfile.close()

#Determine column indices of inputs
base_CPOS_column_index = input_matrix_base_options[0].index(base_CPOS_column)
base_options_group_column_index = input_matrix_base_options[0].index(base_options_group_column)
base_model_year_column_index = input_matrix_base_options[0].index(base_model_year_column)
base_options_code_column_index = input_matrix_base_options[0].index(base_options_code_column)
base_options_desc_column_index = input_matrix_base_options[0].index(base_options_desc_column)

#Dictionary for mapping between base options and infile_data
base_options_dict = {}
for i,x in enumerate(input_matrix_base_options):
    if i != 0:
        key = (x[base_CPOS_column_index],x[base_model_year_column_index],x[base_options_group_column_index])
        if key not in base_options_dict:
            base_options_dict[key] = x[base_options_code_column_index]

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
input_matrix_grouping_name = [i[grouping_options_group_column_index] for i in input_matrix_grouping[1:len(input_matrix_grouping)]]
grouping_column_names_unique = list(set(input_matrix_grouping_name))

#Group to codes dictionary
group_to_codes_dict = dict((k,[x[-4:-1] for i,x in enumerate(input_matrix_grouping_code_and_desc) if input_matrix_grouping_name[i] == k]) for k in input_matrix_grouping_name)
        
#Find all unique configuration codes in infile_data
#Pull configuration strings and separate out the codes
configuration_column_codes = [i[configuration_column_index] for i in input_matrix]
configuration_column_codes = [i.split(',') for i in configuration_column_codes]
#Add in the HCO codes and remove duplicates
HCO_column_codes = [i[HCO_column_index] for i in input_matrix]
HCO_column_codes = [i.split(',') for i in HCO_column_codes]
for i,x in enumerate(HCO_column_codes):
    if i != 0:
        for j in x:
            if configuration_column_codes[i] == ['-'] and j != '':
                configuration_column_codes[i] = [j]
            if j not in configuration_column_codes[i] and j != '':
                configuration_column_codes[i].append(j)

#Remove 'TBT' code from configuration strings
for i,x in enumerate(configuration_column_codes):
    if 'TBT' in x:
        configuration_column_codes[i].remove('TBT')

#Add in package dependencies
#Read package dependencies data
myfile = open(infile_package_dependencies,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_package_dependencies = list(reader)
myfile.close()

#Determine column indices of inputs
package_dependencies_MY_column_index = input_matrix_package_dependencies[0].index(package_dependencies_MY_column)
package_dependencies_CPOS_column_index = input_matrix_package_dependencies[0].index(package_dependencies_CPOS_column)
package_dependencies_options_code_column_index = input_matrix_package_dependencies[0].index(package_dependencies_options_code_column)
package_dependencies_options_desc_column_index = input_matrix_package_dependencies[0].index(package_dependencies_options_desc_column)
package_dependencies_group_column_index = input_matrix_package_dependencies[0].index(package_dependencies_group_column)
package_dependencies_base_code_column_index = input_matrix_package_dependencies[0].index(package_dependencies_base_code_column)
package_dependencies_base_desc_column_index = input_matrix_package_dependencies[0].index(package_dependencies_base_desc_column)

#Dictionary for (MY,CPOS,Package) to [Group,Dependency]
package_dependencies_dict = {}
for i,x in enumerate(input_matrix_package_dependencies):
    if i != 0:
        package_dependencies_dict[(x[package_dependencies_MY_column_index], x[package_dependencies_CPOS_column_index], x[package_dependencies_options_code_column_index])] = [x[package_dependencies_group_column_index], x[package_dependencies_base_code_column_index]]

#Look for package dependencies not yet included in configuration_column_codes
for i,x in enumerate(configuration_column_codes):
    if i != 0:
        for current_option_code in x:
            current_lookup_key = (input_matrix[i][model_year_column_index], input_matrix[i][CPOS_column_index], current_option_code)
            if current_lookup_key in package_dependencies_dict:
                group_name_dependency = package_dependencies_dict[current_lookup_key][0]
                if not any([val in x for val in group_to_codes_dict[group_name_dependency]]):
                    configuration_column_codes[i].append(package_dependencies_dict[current_lookup_key][1])

#Sort the configuration strings
for i in configuration_column_codes:
    i.sort()
    
#Find the unique configuration codes
configuration_column_codes_unique = list(set([item for sublist in configuration_column_codes for item in sublist]))
configuration_column_codes_unique = [i for i in configuration_column_codes_unique if i != configuration_column and i != '-']

#Find codes in base options data that aren't in the infile_data
base_options_unique_codes = list(set([x[base_options_code_column_index] for i,x in enumerate(input_matrix_base_options) if i != 0]))
new_base_options_codes = [i for i in base_options_unique_codes if i not in configuration_column_codes_unique]

#Compiled codes from infile_data and base options
compiled_options_codes = configuration_column_codes_unique + new_base_options_codes

#Read mapping data
myfile = open(infile_mapping,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_mapping = list(reader)
myfile.close()

#Determine column indices of inputs
sales_code_column_index = input_matrix_mapping[0].index(sales_code_column)
sales_desc_column_index = input_matrix_mapping[0].index(sales_desc_column)
sales_class_code_column_index = input_matrix_mapping[0].index(sales_class_code_column)

#Find the configuration descriptions that correspond to compiled_options_codes
input_matrix_mapping_code = [i[sales_code_column_index] for i in input_matrix_mapping]
input_matrix_mapping_desc = [i[sales_desc_column_index] for i in input_matrix_mapping]
configuration_column_desc_unique = [input_matrix_mapping_desc[input_matrix_mapping_code.index(i)] for i in compiled_options_codes]
configuration_column_codes_and_desc_unique = [configuration_column_desc_unique[i] + ' [' + x + ']' for i,x in enumerate(compiled_options_codes)]

#Update configuration_column_codes and configuration_column_codes_unique based on
#new_base_options_codes, making sure that each configuration_column_codes has a
#compiled_options_codes for every grouping_column_names_unique
for i,x in enumerate(configuration_column_codes):
    if i != 0:
        VIN_type = (input_matrix[i][CPOS_column_index], input_matrix[i][model_year_column_index])
        for k in grouping_column_names_unique:
            current_options_to_be_grouped = group_to_codes_dict[k]
            #If current VIN being analyzed does not have a configuration code for the
            #specific option group, check base options data
            if not any([j in x for j in current_options_to_be_grouped]):
                current_key = VIN_type + (k,)
                if current_key in base_options_dict:
                    current_value = base_options_dict[current_key]
                    input_matrix[i][configuration_column_index] = input_matrix[i][configuration_column_index] + ',' + current_value
                    configuration_column_codes[i].append(current_value)

##################### UPDATE BASED ON BASE OPTIONS DATA AND PACKAGE DEPENDENCIES #####################


#Sorted configuration strings - calculated configuration ID
configuration_ids = []
for i in configuration_column_codes:
    configuration_ids.append(i)

for i in range(len(configuration_ids)):
    current_model_year = input_matrix[i][model_year_column_index]
    current_CPOS = input_matrix[i][CPOS_column_index]
    configuration_ids[i].sort()
    if i != 0:
        input_matrix[i][configuration_column_index] = configuration_ids[i]
    configuration_ids[i] = ''.join(configuration_ids[i] + [current_CPOS] + [current_model_year])

#Mapping configuration strings (and model year) to unique configuration ID
unique_configuration_ids = list(set(configuration_ids))
configuration_ids_mapped = {}
configuration_ids_column = ['Configuration ID']
counter_2016 = counter_2015 = counter_2014 = 1
for i,x in enumerate(configuration_ids):
    if input_matrix[i][model_year_column_index] == '2016':
        if x in configuration_ids_mapped:
            configuration_ids_column.append(configuration_ids_mapped[x])
        else:
            configuration_ids_mapped[x] = str(counter_2016) + '_2016'
            counter_2016 += 1
            configuration_ids_column.append(configuration_ids_mapped[x])
    elif input_matrix[i][model_year_column_index] == '2015':
        if x in configuration_ids_mapped:
            configuration_ids_column.append(configuration_ids_mapped[x])
        else:
            configuration_ids_mapped[x] = str(counter_2015) + '_2015'
            counter_2015 += 1
            configuration_ids_column.append(configuration_ids_mapped[x])
    elif input_matrix[i][model_year_column_index] == '2014':
        if x in configuration_ids_mapped:
            configuration_ids_column.append(configuration_ids_mapped[x])
        else:
            configuration_ids_mapped[x] = str(counter_2014) + '_2014'
            counter_2014 += 1
            configuration_ids_column.append(configuration_ids_mapped[x])

#Add options columns and configuration IDs based on configuration_column_desc_unique
output_matrix = []
for i,x in enumerate(configuration_column_codes):
    if i == 0:
        #Column headers
        output_matrix.append(input_matrix[0] + configuration_column_codes_and_desc_unique + [configuration_ids_column[0]])
    else:
        options_columns_values = [input_matrix_mapping_desc[input_matrix_mapping_code.index(j)] + ' [' + j + ']' for j in x if j != '-']
        options_columns_values_padded = range(len(configuration_column_desc_unique))
        for k in range(len(configuration_column_codes_and_desc_unique)):
            if configuration_column_codes_and_desc_unique[k] in options_columns_values:
                options_columns_values_padded[k] = configuration_column_codes_and_desc_unique[k]
            else:
                options_columns_values_padded[k] = ""
        
        #input_matrix data plus the options columns plus the configuration IDs
        output_matrix.append(list(input_matrix[i]) + options_columns_values_padded + [configuration_ids_column[i]])

#Add grouping columns by combining the options columns
output_matrix_final = output_matrix
for k in grouping_column_names_unique:
    current_options_to_be_grouped = [x for i,x in enumerate(input_matrix_grouping_code_and_desc) if input_matrix_grouping_name[i] == k]
    output_matrix_column_indices_to_be_grouped = [output_matrix[0].index(i) for i in current_options_to_be_grouped]
    output_matrix_columns_grouped = [''.join([i[j] for j in output_matrix_column_indices_to_be_grouped]) for i in output_matrix]
    #Check that output_matrix_columns_grouped doesn't "overlap" options
    unique_output_matrix_columns_grouped = list(set(output_matrix_columns_grouped))
    unique_output_matrix_columns_grouped = [i for i in unique_output_matrix_columns_grouped if i != '' and i != output_matrix_columns_grouped[0]]
    if not all([m in current_options_to_be_grouped for m in unique_output_matrix_columns_grouped]):
        print 'The options in the grouping ' + k + ' have overlap'
        sys.exit()

    #Add grouped column to output_matrix_final
    output_matrix_columns_grouped[0] = k
    output_matrix_final = [x + [output_matrix_columns_grouped[i]] for i,x in enumerate(output_matrix_final)]
    
###############################################################################
#Output all of the Data
###############################################################################

with open(outfile, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(output_matrix_final)):
        current_output = output_matrix_final[i]     
        writer.writerow(current_output) 