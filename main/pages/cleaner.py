from os import write
import streamlit as st
import pandas as pd
from methods import cleandataset_method as cl  # import cleandataset method


def cleanerPageDisplay():

    st.subheader("Dataset Cleaner")
    

    uploaded_file = st.sidebar.file_uploader(label = "Upload your CSV File (200MB max)", type=['csv'])

    global df
    if uploaded_file is not None:
        st.write("**Original Dataset** - Any String Data Types remain until Datset Cleaning Completed ")
        df = pd.read_csv(uploaded_file, na_values="NaN")
        st.dataframe(df)
        headers = list(df.columns)

        nulldata = df[df.isnull().any(axis=1)].head() #returns no rows if there is 0 null values
        st.write("**Missing Data** - Any Data from the uploaded dataset that has a missing entry in a row")
        st.write(nulldata)

        #if nulldata:
            #st.write(nulldata)
        #else:
            #st.write("No missing Data")


        #with st.sidebar: # (Option to have clean dataset options in sidebar)
        with st.form('dataset_cleanerform'):
                st.subheader('Dataset Cleaning:')
                st.write('*a) Please select an option from each dropdown as to how missing data from each column is to be processed (If Applicable)*')
                st.write('*b) Delete Row option will remove a row with missing data from the cleaned dataset*')
                st.write('*c) Columns that are not of type Integer, will only have option to Delete Row or Generate Mode*')
                clean_method = {}
                index = 0
                for col in headers:
                    clean_method[index] = [col]
                    if cl.rowContainNum(df[col]):
                        clean_method[index].append(st.selectbox(
                        str(index) + ') ' + str(col), [
                        'Delete Row',
                        'Mean',
                        'Mode',
                        'Median']))
            
                    else:
                        clean_method[index].append(st.selectbox( 
                        str(index) + ') ' + str(col), [
                        'Delete Row',
                        'Mode']))
                
                    index += 1


                submitted = st.form_submit_button("Submit")

        if submitted:
            st.write('**Cleaned Dataset** - String to Integer Conversion Completed for ML Methods:')
            st.write('- *Full-Time (F) converted to 0*')
            st.write('- *Part-Time (P) converted to 1*')
            st.write('- *Male (M) converted to 0*')
            st.write('- *Female (F) converted to 1*')
            st.write('- *Birthdate converted to respective age*')
            st.write('- *Programming score converted to 0 (<70) and 1 (70+)*')
            cleaned_df = cl.cleanDataFrame(df, clean_method)
            # Returns the revised/cleaned dataset
            st.write(cleaned_df)
            st.subheader('Download Cleaned Dataset as CSV File')
            # Add cleaned dataset csv to filename. Adds _cleaned so can differentiate new file when dowloaded
            file_name = uploaded_file.name[0:-4] + '_cleaned.csv'
            # Remove space to activate download link - replaced with an underscore to remove the space
            file_name = file_name.replace(" ", "_")
            st.markdown(cl.csv_downnloadlink(cleaned_df, file_name), unsafe_allow_html=True)
            st.session_state.cleaned_df = cleaned_df

    # If no dataset is yet to be uploaded
    else:
        st.write("Upload your Dataset using the Sidebar")




