from matplotlib.pyplot import axis
import pandas as pd
import base64
import datetime
import sklearn

from datetime import date
from sklearn.preprocessing import LabelEncoder


def get_csv_df(file):

    # Confirms if uploader file is of type CSV
    # Reads CSV file as pandas pd dataframe
    # Returns df True if CSV file read successful

    format = file.name.split('.')[-1].upper()
    if format == 'CSV':
        df = pd.read_csv(file)
        
        return df

def rowContainNum(columnData):

    # Confirms if data in all columns are of type Int or No Data
    # Returns isIntList if True

    isIntList = True
    for item in columnData:
        if not (isinstance(item, int) or isinstance(item, float) or isinstance(item, date) or item == 'NaN'):
            isIntList = False
    return isIntList


def cleanDataFrame(df, clean_method):

    # In pandas dataframe function performs data cleaning
    # Delete row option and Mode option only for data not of Integer type
    # Returns cleaned dataset as pandas dataframe

    for col in clean_method:
        if clean_method[col][1] == 'Delete Row':
            df.dropna(subset=[clean_method[col][0]], inplace=True)
            
    for col in clean_method:
        if clean_method[col][1] == 'Mean':
            x = df[clean_method[col][0]].mean()
            df[clean_method[col][0]].fillna(x, inplace=True)
        elif clean_method[col][1] == 'Median':
            x = df[clean_method[col][0]].median()
            df[clean_method[col][0]].fillna(x, inplace=True)
        elif clean_method[col][1] == 'Mode':
            x = df[clean_method[col][0]].mode()[0]
            df[clean_method[col][0]].fillna(x, inplace=True)

    df = df.reset_index(drop=True)

    # Convert String to Integer #
    # Convert Full-Time (F) to 0 in Trans_Acadmode column
    # Convert Part-Time (P) to 1 in Trans_Acadmode column
    # Convert Male (M) to 0 in Trans_Sex column
    # Convert Female (F) to 1 in Trans_Sex column

    def tran_acadmode(x):
        if x == 'F':
            return 0
        if x == 'P':
            return 1
    
    def tran_sex(y):
        if y == 'M':
            return 0
        if y == 'F':
            return 1

    def tran_age(Birthdate):
        today = date.today()
        age = today.year - Birthdate.year - ((today.month, today.day) < (Birthdate.month, Birthdate.day))
        df['Birthdate'] = pd.to_datetime(df['Birthdate'])
        return age

    # return the new columns to the dataframe
    df['Trans_Acadmode'] = df['Acad mode (full / part time)'].apply(tran_acadmode)

    df['Trans_Sex'] = df['Sex'].apply(tran_sex)

    df['Birthdate'] = pd.to_datetime(df['Birthdate'])   # convert the birthdate column to datetime 
    df['Trans_Age'] = df['Birthdate'].apply(tran_age)

    # drop the current acad_mode / sex / birthdate columns
    df = df.drop('Acad mode (full / part time)', 1)
    df = df.drop('Sex', 1)
    df = df.drop('Birthdate', 1)

    # drop the current Programming Score and return 0 or 1
    # 0 = Programming score < 70
    # 1 = Programming core >= 70
    def tran_programming_score(x):
        if x < 70:
            return 0
        if x >=70:
            return 1

    df['Trans_Programming_Score'] = df['Programming score'].apply(tran_programming_score)

    df = df.drop('Programming score', 1)

    # Drop the Acad Mode and Sex Column String Values
    #df = df.drop(['Acad mode (full / part time)', 'Sex'], axis='columns')
    
    return df

def csv_downnloadlink(df, file_name):

    # download csv file as base64 encode method
    # href used to allow link for download
    # styling applied to add color 
    # pandas dataframe returns data for download
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download =' + file_name + ' style="color: #228B22">Download CSV File</a>'
    
    return href
    