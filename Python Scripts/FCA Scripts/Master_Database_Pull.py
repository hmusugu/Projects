#Python 2.7

###############################################################################
#Input Values
###############################################################################
#Output data CSV file - Master Database
outfile = "C:\\Users\\tkress\\Desktop\\FCA\\Master Database\\test.csv" #Master Database_US_Retail_03172017.csv"

###############################################################################

import pyodbc
import pandas as pd

###############################################################################

sql_connection_string = "DRIVER={SQL Server};SERVER=USSLTCHER8212sq.dan.sltc.com\SQL01,1440;DATABASE=ProjectFCA"
engine = pyodbc.connect(sql_connection_string)
db_conn = pyodbc.connect(sql_connection_string)
db_conn.autocommit = False
db_cursor = db_conn.cursor()
db_conn.commit()
retail = "'US_Retail'"
fleet = "'Fleet'"
mexico = "'MEX'"
canada = "'CAN'"
international = "'INT'"
#sql_command_1 = 'Select [VIN_8], [Configuration ID], [TrimLevel], [SoldDate], [SoldMonth], [Avg DOL], [ShipmentsDate], [DeliveryDate], [Incentives], [Contribution_Profit] From [DT].[Master_3_17_17] Where Data_Source in (%s,%s)' %(retail, fleet)
#sql_command_1 = 'Select * From [DT].[Master_3_17_17] Where Data_Source in (%s)' %(retail)
sql_command_1 = 'Select * From [DT].[IVS_Input_I]'
df = pd.io.sql.read_sql(sql_command_1,engine)
db_cursor.close()
df.to_csv(outfile)

# execfile('C:\Users\\tkress\Desktop\FCA\Master Database\Master_Database_Pull.py')