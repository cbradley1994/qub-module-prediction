'''

Created by Callum Bradley

File contains core functionality performing Data Quality Testing using Python Library - Great Expectations

'''

# Import Great Expectations
import great_expectations as ge

# Import Pandas DataFrame
import pandas as pd

dataset = pd.read_csv("Callum Updated Dataset.csv")
df = ge.from_pandas(dataset)
#print(df)

    # ------- Birthdate Tests ------- #

print("\nBirthdate Tests:")

# Check Birthdate does not contain NULL data
#print(df.expect_column_values_to_not_be_null('Birthdate'))

# Create Parameter to return True if No null values exist
birthdate_not_empty = df.expect_column_values_to_not_be_null('Birthdate')
if (birthdate_not_empty["success"]):
    print("Birthdate Feature contains No NULL data")
else:
    print("Birthdate Feature contains NULL data")



    # ------- DR% Tests ------- #

print("\nDR% Tests:")

# Check DR% does not contain NULL data
#print(df.expect_column_values_to_not_be_null('DR%'))

# Create Parameter to return True if No null values exist
dr_not_empty = df.expect_column_values_to_not_be_null('DR%')
if (dr_not_empty["success"]):
    print("DR% Feature contains No NULL data")
else:
    print("DR% Feature contains NULL data")

# Create Parameter to return True if data is 0-100%
df_1to100 = df.expect_column_values_to_be_between('DR%', 0, 100)
if (df_1to100["success"]):
    print("DR% Feature has all values 0-100%")
else:
    print("DR% Feature has values outside limits 0-100%")




    # ------- SM% Tests ------- #

print("\nSM% Tests:")

# Check SM% does not contain NULL data
#print(df.expect_column_values_to_not_be_null('SM%'))

# Create Parameter to return True if No null values exist
sm_not_empty = df.expect_column_values_to_not_be_null('SM%')
if (sm_not_empty["success"]):
    print("SM% Feature contains No NULL data")
else:
    print("SM% Feature contains NULL data")

# Create Parameter to return True if data is 1-100%
sm_1to100 = df.expect_column_values_to_be_between('SM%', 0, 100)
if (sm_1to100["success"]):
    print("SM% Feature has all values 0-100%")
else:
    print("SM% Feature has values outside limits 0-100%")




    # ------- WV% Tests ------- #

print("\nWV% Tests:")

# Check WV% does not contain NULL data
#print(df.expect_column_values_to_not_be_null('WV%'))

# Create Parameter to return True if No null values exist
wv_not_empty = df.expect_column_values_to_not_be_null('WV%')
if (wv_not_empty["success"]):
    print("WV% Feature contains No NULL data")
else:
    print("WV% Feature contains NULL data")

# Create Parameter to return True if data is 1-100%
wv_1to100 = df.expect_column_values_to_be_between('WV%', 0, 100)
if (sm_1to100["success"]):
    print("WV% Feature has all values 0-100%")
else:
    print("WV% Feature has values outside limits 0-100%")




    # ------- OVERALL % Tests ------- #

print("\nOVERALL % Tests:")

# Check OVERALL % does not contain NULL data
#print(df.expect_column_values_to_not_be_null('OVERALL %'))

# Create Parameter to return True if No null values exist
overall_not_empty = df.expect_column_values_to_not_be_null('OVERALL %')
if (overall_not_empty["success"]):
    print("OVERALL % Feature contains No NULL data")
else:
    print("OVERALL % Feature contains NULL data")

# Create Parameter to return True if data is 1-100%
overall_1to100 = df.expect_column_values_to_be_between('OVERALL %', 0, 100)
if (overall_1to100["success"]):
    print("OVERALL % Feature has all values 0-100%")
else:
    print("OVERALL % Feature has values outside limits 0-100%")




    # ------- Programming score Tests ------- #

print("\nProgramming score Tests:")

# Check Programming score does not contain NULL data
#print(df.expect_column_values_to_not_be_null('Programming score'))

# Create Parameter to return True if No null values exist
Programming_score_not_empty = df.expect_column_values_to_not_be_null('Programming score')
if (Programming_score_not_empty["success"]):
    print("Programming score Feature contains No NULL data")
else:
    print("Programming score Feature contains NULL data")

# Create Parameter to return True if data is 1-100%
programmingscore_1to100 = df.expect_column_values_to_be_between('Programming score', 0, 100)
if (programmingscore_1to100["success"]):
    print("Programming score Feature has all values 0-100%")
else:
    print("Programming score Feature has values outside limits 0-100%")




    # ------- Acad mode (full / part time) Tests ------- #

print("\nAcad mode (full / part time) Tests:")

# Check Acad mode (full / part time) does not contain NULL data
#print(df.expect_column_values_to_not_be_null('Acad mode (full / part time)'))

# Create Parameter to return True if No null values exist
acadmode_not_empty = df.expect_column_values_to_not_be_null('Acad mode (full / part time)')
if (acadmode_not_empty["success"]):
    print("Acad mode (full / part time) Feature contains No NULL data")
else:
    print("Acad mode (full / part time) Feature contains NULL data")

# Create Parameter to return True if values outside of F/M exist
acadmode_acronym = df.expect_column_values_to_be_in_set('Acad mode (full / part time)', ['F', 'P'])
if (acadmode_acronym["success"]):
    print("Acad mode (full / part time) Feature only contains F or M data")
else:
    print("Acad mode (full / part time) Feature contains data outside of F/M requirements")



    # -- Sex Tests -- #

print("\nSex Tests:")

# Check Sex does not contain NULL data
#print(df.expect_column_values_to_not_be_null('Sex'))

# Create Parameter to return True if No null values exist
sex_not_empty = df.expect_column_values_to_not_be_null('Sex')
if (sex_not_empty["success"]):
    print("Sex Feature contains No NULL data")
else:
    print("Sex Feature contains NULL data")

# Create Parameter to return True if values outside of F/M exist
sex_acronym = df.expect_column_values_to_be_in_set('Sex', ['M', 'F'])
if (sex_acronym["success"]):
    print("Sex Feature only contains F or M data")
else:
    print("Sex Feature contains data outside of F/M requirements")
