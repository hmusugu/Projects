#Python 2.7

###############################################################################
#Input Values
###############################################################################
#Input data CSV file - Master Data
infile_data = "C:\\Users\\tkress\\Desktop\\FCA\\Master Database\\Master Database_US_Retail_03022017_v2.csv"
# #Input data CSV file - Updated LOH DT
# infile_LOH_DT = "C:\\Users\\tkress\\Desktop\\FCA\\Master Database\\MASTER_CONTRIBUTION_PROFIT.csv"
#Output data CSV file - Take Rate Analysis
outfile = "C:\\Users\\tkress\\Desktop\\FCA\\Take Rate Analysis\\Cherokee Take Rate Analysis_US_Retail_03022017_v2.csv"

#Column names in infile_data
VIN_column = 'VIN_8'
model_year_column = 'ModelYear'
total_revenue_column = 'TotalRevenue'
incentives_column = 'Incentives'
total_material_column = 'TOTAL-MAT'
warranty_column = 'Warranty'
obt_column = 'OBT'
ibt_column = 'IBT'
loh_column = 'LOH'

# #Column names in infile_LOH_DT
# DT_VIN_column = 'VIN_8'
# DT_LOH_column = 'LOH_DT'

#Bin size of contribution profit
bin_size = 500

###############################################################################

import csv
from collections import Counter
import copy

###############################################################################
#Start of Data Transformation of infiles
###############################################################################

#Read Master Data
myfile = open(infile_data,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix = list(reader)
myfile.close()

#Determine column indices of inputs
VIN_column_index = input_matrix[0].index(VIN_column)
model_year_column_index = input_matrix[0].index(model_year_column)
total_revenue_column_index = input_matrix[0].index(total_revenue_column)
incentives_column_index = input_matrix[0].index(incentives_column)
total_material_column_index = input_matrix[0].index(total_material_column)
warranty_column_index = input_matrix[0].index(warranty_column)
obt_column_index = input_matrix[0].index(obt_column)
ibt_column_index = input_matrix[0].index(ibt_column)
loh_column_index = input_matrix[0].index(loh_column)

# #Select out only MY 2016
# input_matrix_old = copy.copy(input_matrix)
# del input_matrix
# input_matrix = []
# for i,x in enumerate(input_matrix_old):
#     if i == 0:
#         input_matrix.append(x)
#     else:
#         if x[model_year_column_index] == '2016':
#             input_matrix.append(x)

# ############### Update LOH from LOH DT Data ###############
# #Read LOH DT Data
# myfile = open(infile_LOH_DT,"rb")
# reader = csv.reader(myfile,delimiter=',')
# input_LOH_DT = list(reader)
# myfile.close()

# #Determine column indices of inputs
# DT_VIN_column_index = input_LOH_DT[0].index(DT_VIN_column)
# DT_LOH_column_index = input_LOH_DT[0].index(DT_LOH_column)

# #DT_LOH dictionary
# DT_LOH_dict = {}
# for i,x in enumerate(input_LOH_DT):
#     if i != 0:
#         DT_LOH_dict[x[DT_VIN_column_index]] = x[DT_LOH_column_index]

# #Change input_matrx LOH to be LOH DT Data
# for i,x in enumerate(input_matrix):
#     if i != 0:
#         current_VIN = x[VIN_column_index]
#         updated_LOH_DT_value = DT_LOH_dict[current_VIN]
#         input_matrix[i][loh_column_index] = updated_LOH_DT_value
# ############### Update LOH from LOH DT Data ###############

#Calculate Contribution Margin and Contribution Profit
contribution_margin = [float(x[total_revenue_column_index])-float(x[incentives_column_index])-float(x[total_material_column_index])-float(x[warranty_column_index])-float(x[obt_column_index])-float(x[ibt_column_index]) for i,x in enumerate(input_matrix) if i != 0 and x[model_year_column_index] != '2014']
contribution_profit = [float(x[total_revenue_column_index])-float(x[incentives_column_index])-float(x[total_material_column_index])-float(x[warranty_column_index])-float(x[obt_column_index])-float(x[ibt_column_index])-float(x[loh_column_index]) for i,x in enumerate(input_matrix) if i != 0 and x[model_year_column_index] != '2014']

#Min and max bin ranges
min_bin = min(contribution_profit)
max_bin = max(contribution_profit)
bin_ranges = range(int(min_bin - min_bin%bin_size),int(max_bin- max_bin%bin_size) + 1,bin_size)

#Calculate take rates for different Options Groups
output_dict = {}
for j,m in enumerate(input_matrix[0]):
    if ' (Group)' in m:
        current_option_group = [input_matrix[i][j] for i,x in enumerate(input_matrix) if i != 0 and x[model_year_column_index] != '2014']
        current_option_choices = set(current_option_group)
        overall_counts = dict(Counter(current_option_group))
        for l in bin_ranges:
            current_option_group_subset = [x for i,x in enumerate(current_option_group) if l <= contribution_profit[i] < l+bin_size]
            counts = dict(Counter(current_option_group_subset))
            for k in current_option_choices:
                if k in counts:
                    output_dict[(m,k,str(l))] = [str(counts[k]),str(float(counts[k])/len(current_option_group_subset)),str(float(overall_counts[k])/len(current_option_group))]
                else:
                    output_dict[(m,k,str(l))] = [str(0),str(0),str(float(overall_counts[k])/len(current_option_group))]

###############################################################################
#Output all of the Data
###############################################################################

with open(outfile, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Option Group','Option Choice','Contribution Profit Bin','Count of VINs','Take Rate','Overall Take Rate']) 
    for i in output_dict.keys():
        current_output = list(i) + output_dict[i]    
        writer.writerow(current_output)

# execfile('C:\Users\\tkress\Desktop\FCA\\Take Rate Analysis\Options_Take_Rate_Analysis.py')