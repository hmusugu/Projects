#Python 2.7

###############################################################################
#Input Values
###############################################################################
#Input data CSV file - VINs mapped to sales codes (Long Spec Data)
infile_Long_Spec = "C:\\Users\\tkress\\Desktop\\FCA\\Long Spec\\Unique Long Spec Compiled.csv"
#Input data CSV file - Sales code rules mapped to parts
infile_mapping = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\2016 EPIRS_No_ST_Unique_Part-EI-EngCond.csv"
#Output data CSV file - VINs with (parts,end item,eng cond) indices
outfile_indices = "C:\\Users\\tkress\\Desktop\\FCA\\Parts Analysis\\VINs_to_Indices_Output.csv"

######################################
#Column names in infile_Long_Spec
LS_VIN_Last8_column = 'Last 8 Digits of VIN'
LS_sales_codes_column = 'Sales Codes'
######################################
#Columns in infile_mapping
indice_column = 'INDICE'
part_numbers_column = 'PART NO CHR'
end_item_column = 'EBOM E/I'
eng_cond_column = 'ENGRG COND'

###############################################################################

import csv
import sys

###############################################################################
#Start of Data Transformation of infiles
###############################################################################

#Read infile_Long_Spec
myfile = open(infile_Long_Spec,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_Long_Spec = list(reader)
myfile.close()

#Determine column indices of inputs
LS_VIN_Last8_column_index = input_matrix_Long_Spec[0].index(LS_VIN_Last8_column)
LS_sales_codes_column_index = input_matrix_Long_Spec[0].index(LS_sales_codes_column)

#Columns in input_matrix_Long_Spec
LS_VIN_Last8 = [i[LS_VIN_Last8_column_index] for i in input_matrix_Long_Spec]
LS_sales_codes = [i[LS_sales_codes_column_index] for i in input_matrix_Long_Spec]

#Read infile_mapping
myfile = open(infile_mapping,"rb")
reader = csv.reader(myfile,delimiter=',')
input_matrix_mapping = list(reader)
myfile.close()

#Determine column indices of inputs
indice_column_index = input_matrix_mapping[0].index(indice_column)
part_numbers_column_index = input_matrix_mapping[0].index(part_numbers_column)
end_item_column_index = input_matrix_mapping[0].index(end_item_column)
eng_cond_column_index = input_matrix_mapping[0].index(eng_cond_column)

#Columns in input_matrix_mapping
indice = [i[indice_column_index] for i in input_matrix_mapping]
part_numbers = [i[part_numbers_column_index] for i in input_matrix_mapping]
end_item = [i[end_item_column_index] for i in input_matrix_mapping]
eng_cond_original = [i[eng_cond_column_index] for i in input_matrix_mapping]
eng_cond = [i[eng_cond_column_index] for i in input_matrix_mapping]

#Add spaces inbetween sales codes and logic markers in eng_cond
for i,x in enumerate(eng_cond):
    if i != 0:
        current_eng_cond = x.replace("="," = ")
        current_eng_cond = current_eng_cond.replace("+"," + ")
        current_eng_cond = current_eng_cond.replace("-"," - ")
        current_eng_cond = current_eng_cond.replace("%"," % ")

        #Make sure each string starts with + or -
        if current_eng_cond[0] == ' ':
            current_eng_cond = current_eng_cond[1:]
        else:
            current_eng_cond = '+ ' + current_eng_cond
        eng_cond[i] = current_eng_cond.split(' ')

#Free up memory
#reset_selective -f input_matrix_mapping
del input_matrix_mapping

###############################################################################
#Explanation of 'ENGRG COND' logic
###############################################################################

# A=B=C=D+E+F+G%H%I%J-K%L%M%N

# If you have:
#     One of A,B,C,D
#     All of E,F
#     At least one of G,H,I,J
#     None of K,L,M,N

# The above logic gives the following:

# X88+TBB-XL7%XLB%XLC
#     All of X88,TBB
#     None of XL7,XLB,XLC

# X82+JPD+GTB%GTF-LCJ%LEM%LEZ%GX4%GXD%XBU
#     All of X82,JPD
#     At least one of GTB,GTF
#     None of LCJ,LEM,LEZ,GX4,GXD,XBU

# -M1E-XEW-XFJ
#     None of M1E,XEW,XFJ

###############################################################################

#Create dictionary that maps eng_cond rules to one_of_codes, all_of_codes, at_least_one_of_codes, none_of_codes
eng_cond_dict = {}
for i,current_eng_cond_list in enumerate(eng_cond):
    if i != 0:
        one_of_codes = []
        all_of_codes = []
        at_least_one_of_codes = []
        none_of_codes = []

        #If there is a value in hold_code, clear it before each iteration
        try:
            hold_code
        except:
            None
        else:
            del hold_code

        #Step through logic markers in current_eng_cond_list
        for j,current_logic_marker in enumerate(current_eng_cond_list[::2]):
            if current_logic_marker == '+':
                #If the current_logic_marker is +, wait until the next current_logic_marker is encountered
                #to make a decision about the following code
                
                #If hold_code exists already, store the previous hold_code in all_of_codes (encountered +,A,+,B
                #and therefore A must go into all_of_codes), and update hold_code
                try:
                    hold_code
                except:
                    hold_code = current_eng_cond_list[2*j+1]
                else:
                    all_of_codes.append(hold_code)
                    hold_code = current_eng_cond_list[2*j+1]

                #If the current_logic_marker is + and is the last current_logic_marker, store the code
                #in all_of_codes and break the loop
                if j == (len(current_eng_cond_list[::2]) - 1):
                    all_of_codes.append(hold_code)
                    break

            elif current_logic_marker == '=':
                #If the current_logic_marker is = and hold_code exists, store the value in hold_code in one_of_codes
                #(encountered +,A,=,B and therefore A must go into one_of_codes), store the next code in one_of_codes
                #(B must also go into one_of_codes), and clear hold_code. If no value is in hold_code, only store the next code
                #in one_of_codes (encountered =,A,=B and therefore A is already in one_of_codes but B must also go into 
                #one_of_codes).
                try:
                    hold_code
                except:
                    one_of_codes.append(current_eng_cond_list[2*j+1])
                else:
                    one_of_codes.append(hold_code)
                    one_of_codes.append(current_eng_cond_list[2*j+1])
                    del hold_code

            elif current_logic_marker == '%':
                #If the current_logic_marker is %, follow "the same logic" as = but for at_least_one_of_codes
                try:
                    hold_code
                except:
                    at_least_one_of_codes.append(current_eng_cond_list[2*j+1])
                else:
                    at_least_one_of_codes.append(hold_code)
                    at_least_one_of_codes.append(current_eng_cond_list[2*j+1])
                    del hold_code

            elif current_logic_marker == '-':
                #If the current_logic_marker is -, store all of the remaining codes in none_of_codes and break the loop.
                #Also, if hold_code exists, store hold_code in all_of_codes (encountered +,A,-,B).

                try:
                    hold_code
                except:
                    None
                else:
                    all_of_codes.append(hold_code)

                none_of_codes = current_eng_cond_list[(2*j+1)::2]
                break

            else:
                print str(current_eng_cond_list) + ' has an invalid value for current_logic_marker: ' + current_logic_marker
                sys.exit()
        
        #Store the output in the dictionary      
        eng_cond_dict[tuple(current_eng_cond_list)] = [one_of_codes,all_of_codes,at_least_one_of_codes,none_of_codes]

#Function that maps Long Spec to parts
def Long_Spec_to_Parts(current_Long_Spec):
    current_Long_Spec_list = set([i.strip() for i in current_Long_Spec.split(',')])
    #Loop through eng_cond_dict rules
    current_parts_ei_cond = []
    for i,current_eng_cond_list in enumerate(eng_cond):
        if i != 0:
            current_rules = eng_cond_dict[tuple(current_eng_cond_list)]
            current_one_of_codes = current_rules[0]
            current_all_of_codes = current_rules[1]
            current_at_least_one_of_codes = current_rules[2]
            current_none_of_codes = current_rules[3]

            #If all of the rules conditions are satisfied by the current_Long_Spec_list,
            #the part_numbers corresponding to the current_eng_cond_list is on the current VIN
            current_all_of_codes_intersection = current_Long_Spec_list.intersection(current_all_of_codes)
            if not ((current_all_of_codes == []) or (len(current_all_of_codes_intersection) == len(current_all_of_codes))):
                continue

            current_none_of_codes_intersection = current_Long_Spec_list.intersection(current_none_of_codes)
            if not ((current_none_of_codes == []) or (len(current_none_of_codes_intersection) == 0)):
                continue

            current_one_of_codes_intersection = current_Long_Spec_list.intersection(current_one_of_codes)
            if not ((current_one_of_codes == []) or (len(current_one_of_codes_intersection) == 1)):
                continue
            
            current_at_least_one_of_codes_intersection = current_Long_Spec_list.intersection(current_at_least_one_of_codes)
            if not ((current_at_least_one_of_codes == []) or (len(current_at_least_one_of_codes_intersection) >= 1)):
                continue

            current_parts_ei_cond.append(indice[i])

            # #Overlap between codes in rules and codes in current_Long_Spec_list
            # current_one_of_codes_intersection = current_Long_Spec_list.intersection(current_one_of_codes)
            # current_all_of_codes_intersection = current_Long_Spec_list.intersection(current_all_of_codes)
            # current_at_least_one_of_codes_intersection = current_Long_Spec_list.intersection(current_at_least_one_of_codes)
            # current_none_of_codes_intersection = current_Long_Spec_list.intersection(current_none_of_codes)

            # #If all of the rules conditions are satisfied by the current_Long_Spec_list,
            # #the part_numbers corresponding to the current_eng_cond_list is on the current VIN
            # if (  ((current_one_of_codes == []) or (len(current_one_of_codes_intersection) == 1)) and
            #       ((current_all_of_codes == []) or (len(current_all_of_codes_intersection) == len(current_all_of_codes))) and
            #       ((current_at_least_one_of_codes == []) or (len(current_at_least_one_of_codes_intersection) >= 1)) and
            #       ((current_none_of_codes == []) or (len(current_none_of_codes_intersection) == 0))  ):
            #    current_parts_ei_cond.append(part_numbers[i])

    #Return all unique (parts,end items,eng cond) indice values
    return ','.join(current_parts_ei_cond)

#Loop through Long Specs and find parts
output = [['VIN_8','Parts']]
for i,current_Long_Spec in enumerate(LS_sales_codes):
    if i != 0:
        current_parts_ei_cond = Long_Spec_to_Parts(current_Long_Spec)
        output.append([LS_VIN_Last8[i], current_parts_ei_cond])
    if i%1000 == 0:
        print i
###############################################################################
#Output all of the Data
###############################################################################

with open(outfile_indices, 'ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in output:
        writer.writerow(i)

# execfile('C:\Users\\tkress\Desktop\FCA\Parts Analysis\Parts_Analysis.py')