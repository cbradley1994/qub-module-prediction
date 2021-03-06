'''

Created by Callum Bradley

File contains core functionality for webapp regards front-end and training of ML Models for Group Entry Prediction

'''
#import libraries

import streamlit as st
from seaborn.axisgrid import pairplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns  # Python visualisation library based upon matplotlib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def machinelearningdisplaygroup():

    '''
    
    Comments:
    Creates frontend elements for all model training and prediciton for dropdown: '2b. ML Dataset - Group Entry Prediction'
    
    '''

    df_file = False
    st.subheader("1. Model Training")

    #set CSV uploader function via sidebar
    st.sidebar.subheader('Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader(
        label="Ensure Dataset is cleaned prior to Upload. Visit Clean Dataset Page.", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values="NaN")
        df_file = True

    else:
        st.write("Upload Cleaned Dataset using the Sidebar")

    if df_file == True:
        st.write('**Step 1.1:  Uploaded Cleaned Dataset Dataframe**')
        st.write(df)

        # https://github.com/thefullstackninja/Streamlit_tutorials/blob/master/data_visualization_app.py #

        st.write('**Step 1.2:  Feature Selection**')

        try:
            # seaborn visualisation using correlation plot to identify relationships
            heatmap = st.checkbox("Activate Correlation Matrix")
            if heatmap:
                st.write(
                    "**Correlation Matrix** is a table displaying the correlation coefficients between variables. Each cell demonstrates the correlation between two variables")

                df_corr = df.corr()
                sns.heatmap(df_corr, annot=True, cmap='YlGnBu')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                #Correlation with output variable
                cor_target = abs(df_corr["Trans_Programming_Score"])

                #Selecting highly correlated features
                relevant_features = cor_target[cor_target>0.1]
                st.write('Correlation Ranking > 0.1 : ')
                st.write(relevant_features)

        except ValueError:
            st.write('Dataset is not appropriate for Correlation Matrix')

        try:
            # seaborn visualisation using pair plots to identify relationships
            pair_plot = st.checkbox("Activate Pairs Plot")
            if pair_plot:
                st.write(
                    "**Pairs Plot** is a grid of scatterplots and histograms which show both distribution of single variables and relationships between variables. See sidebar to select Hue, the parameter for dictating colour encoding")
                numeric_columns = list(
                    df.select_dtypes(['float', 'int']).columns)
                vector = st.sidebar.selectbox(
                    'Hue - Pairs Plot', options=numeric_columns)
                fig = sns.pairplot(df, hue=vector)
                st.pyplot(fig)

        except ValueError:
            st.write('Dataset is not appropriate for Pairs Plot')


       # -- MAKING DROP DOWN FEATURE SELECTION -- #

        st.write('**Step 1.3:  Select Features to Train ML Model**')

        try:
            # allow user to chose what features to remain in dataframe for ML Models - Can drop those with poor correlation etc.
            df_dropdrown = df.drop('Trans_Programming_Score', axis=1)
            make_choice = st.multiselect('Select from dropdown', df_dropdrown.columns)
            df_choice = df_dropdrown[make_choice]
            if make_choice:
                st.write('Planned ML Training Dataset Dataframe : ')
                st.write(df_choice)
            
        except ValueError:
            st.write('Error with Feature Selection for Dataframe')

        # -- MAKING DROP DOWN MODEL SELECTION -- # 

        if make_choice:
            
            st.write('**Step 1.4:  Select the Machine Learning Classifier to Train**')
            task = st.selectbox('Select from dropdown', [
                                "< Please select a Classification Model >", "Decision Trees", "Support Vector Machines (SVM)", "Naive Bayes", "Random Forest", "Logistic Regression", "K-Nearest Neighbor"])

# -- Build each ML Model -- #
    
    # -- Support Vector Machines -- #

            if task == "Support Vector Machines (SVM)":
                class SVM_classifier():
                    import pandas as pd

                import numpy as np
                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # creating the target variable - programming score
                X = df_choice #Make X feature for training model match the features selected by user in dropdown
                y = df['Trans_Programming_Score']

                #Start timer to record train time of model
                import time
                start = time.time()

                # Sidebar - Specify parameter settings
                with st.sidebar.subheader('Test/Train Split Ratio'):
                    parameter_test_size = st.sidebar.slider(
                        '(E.g. 20/80 split = 20% Test / 80% Train)', 0.1, 0.9, 0.2, 0.1)

                # Splitting the dataset into test/train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=999)

                # Standardise the dataset - Between 0 and 1. Reduce where there can possibly be outliers that carry weight
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)

                #transform X_trainvalues between 0 and 1 to remove negative values
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()

                #transform X_trainvalues between 0 and 1 to remove negative values
                X_train = min_max_scaler.fit_transform(X_train)

                print(X_train)

                # Length of Train data
                len(X_train)

                # Length of Test data
                len(X_test)

                # Optimise hyper-parameters when training Model

                # GridSearch - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the SVC() (SVC=SVM Model) model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV

                param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
                    1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

                # does 5 fold cross-validation and takes average
                grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
                grid.fit(X_train, y_train)

                #Stop timer to measure train timer for model
                stop = time.time()

                # Find the optimal parameters
                print(grid.best_params_)

                # print how our model looks after hyper-parameter tuning
                print(grid.best_estimator_)
                print('Model after tuning of Hyper-Parameters : ',
                    grid.best_estimator_)

                # Test set predictions - Confusion Matrix and Classification Report
                grid_predictions = grid.predict(X_test)
                print('Confusion Matrix Report ',
                    confusion_matrix(y_test, grid_predictions))
                print('Classification Report ',
                    classification_report(y_test, grid_predictions))

                # Print Model Performance

                st.subheader('2. Model Performance')
                st.markdown('**Step 2.1: Feature Importance (XGBoost)**')
                st.write(
                    "**Feature importance** refers to a class of techniques for assigning a score to an input feature to a predictive model, that indicates the relative importance when making a prediction")

                # xgboost for feature importance on a classification problem
                from xgboost import XGBClassifier

                # define the model
                model = XGBClassifier()

                # fit the model
                model.fit(X_train, y_train)

                # get importance
                importance = model.feature_importances_
                importancelist = importance.tolist()

                # print Feature hearders for bar chart
                st.write(X.columns)
                st.write(importance)
                fig = (importancelist)

                try:
                # seaborn visualisation using correlation plot to identify relationships
                    feature_selection_plot = st.checkbox("Activate Feature Selection Plot")
                    if feature_selection_plot:
                        
                        st.bar_chart(fig)

                except ValueError:
                    st.write('Dataset is not appropriate for Feature Selection')

                st.markdown('**Step 2.2: Hyper-Parameter Tuning Results**')
                st.write('Model after tuning of Hyper-Parameters : ',
                        grid.best_estimator_)
                st.write("Test/Train Split (sidebar) : ", parameter_test_size, " / ", (1-parameter_test_size))

                #print Confusion Marix Chart
                st.markdown('**Step 2.3: Confusion Matrix Report**')

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                confusion_matrix = ConfusionMatrixDisplay(cconf_matrix)
                plt.figure(figsize=(20, 20))
                confusion_matrix.plot(cmap='Reds')
                confusion_matrix.ax_.set(
                    title='Confusion Matrix',
                    xlabel='Predicted Value',
                    ylabel='Actual Value',
                    )
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.markdown('**Step 2.4: Classification Report**')
                report = classification_report(y_test, grid_predictions, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report.head())

                #Measurement for model train time
                st.markdown('**Step 2.5: Training Time(s)**')
                TrainingTime = stop-start
                st.write("%.2f" % TrainingTime)

                #Input of new predictive parameters
                st.write("**Step 2.6: Upload Results for Group Prediction**")
                st.write("Upload the new results via CSV where the features included match those used to train the model in Step 1.3")
                uploaded_results = st.file_uploader(label = "Upload your CSV File containing Group Results (200MB max)", type=['csv'])
                if uploaded_results is not None:
                    df_newresults = pd.read_csv(uploaded_results, delimiter=',')
                    st.dataframe(df_newresults)

                    # -- If Condition to make sure group results features match the trained models features

                    #use set instead of list as set can account for unordered list. Using list wont consider user selecting features in same order as that uploaded
                    if set(df_choice.columns) == set(df_newresults.columns):
                        
                        input_data_group = []
                        for row in df_newresults.values:
                            input_data_as_numpy_array = np.asarray(row)
                            # reshape the array as we are predicting for one instance
                            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            scaler.fit(df_newresults.values)

                            std_data = scaler.transform(input_data_reshaped)
                            #print("STD DATA:",std_data)

                            prediction = grid.predict(std_data)
                            pred_str=np.array_str(prediction)
                            #print(pred_str)

                            if pred_str=="[0]":
                                input_data_group.append("AT RISK")
                            else:
                                input_data_group.append("PASS")

                        df_newresults['Prediction']=input_data_group

                        # Upload function for new set of results to predict from
                        st.write("**Step 2.7: Group Prediction Results**")
                        st.write("Upon upload of results for group prediction, click Predict")
                        prediction_button = st.button("Predict")
                        if prediction_button:
                            
                            st.write(df_newresults)

                            from methods import cleandataset_method as cl  # import cleandataset method

                            st.write("**Step 2.8: Download Prediction Results as CSV File**")
                            # Add cleaned dataset csv to filename. Adds _cleaned so can differentiate new file when dowloaded
                            file_name = uploaded_file.name[0:-4] + '_prediction_results.csv'
                            # Remove space to activate download link - replaced with an underscore to remove the space
                            file_name = file_name.replace(" ", "_")
                            st.markdown(cl.csv_downnloadlink(df_newresults, file_name), unsafe_allow_html=True)
                            st.session_state.cleaned_df = df_newresults

                        else:
                            ("Upload New Results for Group Prediction")
                    
                    else:
                        st.write("**ERROR - Features (Column Names) from uploaded Results do not match features from trained model in Section 1.1 - Planned ML Training Dataset Dataframe. Cannot Continue**")
                        st.write("Please do one of the following:")
                        st.write("1. Re-upload Results where features match that of the trained model")
                        st.write("2. Adjust features in Step 1.3 to Re-train the model to match the features in uploaded Results")

        # -- RANDOM FOREST CLASSIFIER -- #

            elif task == "Random Forest":
                class RandomForest_classifier():
                    import pandas as pd

                import numpy as np
                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # creating the target variable - programming score
                X = df_choice #Make X feature for training model match the features selected by user in dropdown
                y = df['Trans_Programming_Score']

                #Start timer to record train time of model
                import time
                start = time.time()

                # Sidebar - Specify parameter settings
                with st.sidebar.subheader('Test/Train Split Ratio'):
                    parameter_test_size = st.sidebar.slider(
                        '(E.g. 20/80 split = 20% Test / 80% Train)', 0.1, 0.9, 0.2, 0.1)

                # Splitting the dataset into test/train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=999)

                # Standardise the dataset - Between 0 and 1. Reduce where there can possibly be outliers that carry weight
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)

                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()

                #transform X_trainvalues between 0 and 1 to remove negative values
                X_train = min_max_scaler.fit_transform(X_train)

                print(X_train)

                # Length of Train data
                len(X_train)

                # Length of Test data
                len(X_test)

                # Optimise hyper-parameters when training Model

                # GridSearch - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV

                param_grid={'n_estimators':[10,50,100,150,200],'min_samples_leaf':[1,5,10,15,50],'max_features':('auto','sqrt','log2')} 

                # does 5 fold cross-validation and takes average 
                grid = GridSearchCV(RandomForestClassifier(),param_grid, cv=5, refit=True,verbose=2)
                grid.fit(X_train, y_train)

                #Stop timer to measure train timer for model
                stop = time.time()

                # Find the optimal parameters
                print(grid.best_params_)

                # print how our model looks after hyper-parameter tuning
                print(grid.best_estimator_)
                print('Model after tuning of Hyper-Parameters : ',
                    grid.best_estimator_)

                # Test set predictions - Confusion Matrix and Classification Report
                grid_predictions = grid.predict(X_test)
                print('Confusion Matrix Report ',
                    confusion_matrix(y_test, grid_predictions))
                print('Classification Report ',
                    classification_report(y_test, grid_predictions))

                # Print Model Performance

                st.subheader('2. Model Performance')
                st.markdown('**Step 2.1: Feature Importance (XGBoost)**')
                st.write(
                    "**Feature importance** refers to a class of techniques for assigning a score to an input feature to a predictive model, that indicates the relative importance when making a prediction")

                # xgboost for feature importance on a classification problem
                from xgboost import XGBClassifier

                # define the model
                model = XGBClassifier()

                # fit the model
                model.fit(X_train, y_train)

                # get importance
                importance = model.feature_importances_
                importancelist = importance.tolist()

                # print Feature hearders for bar chart
                st.write(X.columns)
                st.write(importance)
                fig = (importancelist)

                try:
                # seaborn visualisation using correlation plot to identify relationships
                    feature_selection_plot = st.checkbox("Activate Feature Selection Plot")
                    if feature_selection_plot:
                        
                        st.bar_chart(fig)

                except ValueError:
                    st.write('Dataset is not appropriate for Feature Selection')

                #print Confusion Marix Chart
                st.markdown('**Step 2.2: Hyper-Parameter Tuning Results**')
                st.write('Model after tuning of Hyper-Parameters : ',
                        grid.best_estimator_)
                st.write("Test/Train Split (sidebar) : ", parameter_test_size, " / ", (1-parameter_test_size))

                #print Confusion Marix Chart
                st.markdown('**Step 2.3: Confusion Matrix Report**')

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                confusion_matrix = ConfusionMatrixDisplay(cconf_matrix)
                plt.figure(figsize=(20, 20))
                confusion_matrix.plot(cmap='Reds')
                confusion_matrix.ax_.set(
                    title='Confusion Matrix',
                    xlabel='Predicted Value',
                    ylabel='Actual Value',
                    )
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.markdown('**Step 2.4: Classification Report**')
                report = classification_report(y_test, grid_predictions, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report.head())

                #Measurement for model train time
                st.markdown('**Step 2.5: Training Time(s)**')
                TrainingTime = stop-start
                st.write("%.2f" % TrainingTime)

                #Input of new predictive parameters
                st.write("**Step 2.6: Upload Results for Group Prediction**")
                st.write("Upload the new results via CSV where the features included match those used to train the model in section 1.1")
                uploaded_results = st.file_uploader(label = "Upload your CSV File containing Group Results (200MB max)", type=['csv'])
                if uploaded_results is not None:
                    df_newresults = pd.read_csv(uploaded_results, delimiter=',')
                    st.dataframe(df_newresults)

                    # -- If Condition to make sure group results features match the trained models features

                    #use set instead of list as set can account for unordered list. Using list wont consider user selecting features in same order as that uploaded
                    if set(df_choice.columns) == set(df_newresults.columns):
                        
                        input_data_group = []
                        for row in df_newresults.values:
                            input_data_as_numpy_array = np.asarray(row)
                            # reshape the array as we are predicting for one instance
                            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            scaler.fit(df_newresults.values)

                            std_data = scaler.transform(input_data_reshaped)
                            #print("STD DATA:",std_data)

                            prediction = grid.predict(std_data)
                            pred_str=np.array_str(prediction)
                            #print(pred_str)

                            if pred_str=="[0]":
                                input_data_group.append("AT RISK")
                            else:
                                input_data_group.append("PASS")

                        df_newresults['Prediction']=input_data_group
                        
                        # Upload function for new set of results to predict from
                        st.write("**Step 2.7: Group Prediction Results**")
                        st.write("Upon upload of results for group prediction, click Predict")
                        prediction_button = st.button("Predict")
                        if prediction_button:
                            
                            st.write(df_newresults)

                            from methods import cleandataset_method as cl  # import cleandataset method

                            st.write("**Step 2.8: Download Prediction Results as CSV File**")
                            # Add cleaned dataset csv to filename. Adds _cleaned so can differentiate new file when dowloaded
                            file_name = uploaded_file.name[0:-4] + '_prediction_results.csv'
                            # Remove space to activate download link - replaced with an underscore to remove the space
                            file_name = file_name.replace(" ", "_")
                            st.markdown(cl.csv_downnloadlink(df_newresults, file_name), unsafe_allow_html=True)
                            st.session_state.cleaned_df = df_newresults

                        else:
                            ("Upload New Results for Group Prediction")
                    

                    else:
                        st.write("**ERROR - Features (Column Names) from uploaded Results do not match features from trained model in Section 1.1 - Planned ML Training Dataset Dataframe. Cannot Continue**")
                        st.write("Please do one of the following:")
                        st.write("1. Re-upload Results where features match that of the trained model")
                        st.write("2. Adjust features in Step 1.3 to Re-train the model to match the features in uploaded Results")

        # -- LOGISTIC REGRESSION -- #

            elif task == "Logistic Regression":
                class LogisticRegression_classifier():
                    import pandas as pd
                    
                import numpy as np
                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # creating the target variable - programming score
                X = df_choice #Make X feature for training model match the features selected by user in dropdown
                y = df['Trans_Programming_Score']

                #Start timer to record train time of model
                import time
                start = time.time()

                # Sidebar - Specify parameter settings
                with st.sidebar.subheader('Test/Train Split Ratio'):
                    parameter_test_size = st.sidebar.slider(
                        '(E.g. 20/80 split = 20% Test / 80% Train)', 0.1, 0.9, 0.2, 0.1)

                # Splitting the dataset into test/train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=999)
         

                # Standardise the dataset - Between 0 and 1. Reduce where there can possibly be outliers that carry weight
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)

                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()

                #transform X_trainvalues between 0 and 1 to remove negative values
                X_train = min_max_scaler.fit_transform(X_train)

                print(X_train)

                # Length of Train data
                len(X_train)

                # Length of Test data
                len(X_test)

                # Optimise hyper-parameters when training Model

                # GridSearch - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV
                import numpy as np

                param_grid={"C":np.logspace(-3,3,7), "penalty":["l2"]} # l1 lasso l2 ridge 

                # does 5 fold cross-validation and takes average
                grid = GridSearchCV(LogisticRegression(),param_grid, cv=5, refit=True,verbose=2)
                grid.fit(X_train, y_train)

                #Stop timer to measure train timer for model
                stop = time.time()

                # Find the optimal parameters
                # print best parameter after tuning
                print(grid.best_params_)

                # print how our model looks after hyper-parameter tuning
                print(grid.best_estimator_)
                print('Model after tuning of Hyper-Parameters : ',
                    grid.best_estimator_)

                # Test set predictions - Confusion Matrix and Classification Report
                grid_predictions = grid.predict(X_test)
                print('Confusion Matrix Report ',
                    confusion_matrix(y_test, grid_predictions))
                print('Classification Report ',
                    classification_report(y_test, grid_predictions))

                # Print Model Performance

                st.subheader('2. Model Performance')
                st.markdown('**Step 2.1: Feature Importance (XGBoost)**')
                st.write(
                    "**Feature importance** refers to a class of techniques for assigning a score to an input feature to a predictive model, that indicates the relative importance when making a prediction")

                # xgboost for feature importance on a classification problem
                from xgboost import XGBClassifier

                # define the model
                model = XGBClassifier()

                # fit the model
                model.fit(X_train, y_train)

                # get importance
                importance = model.feature_importances_
                importancelist = importance.tolist()

                # print Feature hearders for bar chart
                st.write(X.columns)
                st.write(importance)
                fig = (importancelist)

                try:
                # seaborn visualisation using correlation plot to identify relationships
                    feature_selection_plot = st.checkbox("Activate Feature Selection Plot")
                    if feature_selection_plot:
                        
                        st.bar_chart(fig)

                except ValueError:
                    st.write('Dataset is not appropriate for Feature Selection')

                st.markdown('**Step 2.2: Hyper-Parameter Tuning Results**')
                st.write('Model after tuning of Hyper-Parameters : ',
                        grid.best_estimator_)
                st.write("Test/Train Split (sidebar) : ", parameter_test_size, " / ", (1-parameter_test_size))

                #print Confusion Marix Chart
                st.markdown('**Step 2.3: Confusion Matrix Report**')

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                confusion_matrix = ConfusionMatrixDisplay(cconf_matrix)
                plt.figure(figsize=(20, 20))
                confusion_matrix.plot(cmap='Reds')
                confusion_matrix.ax_.set(
                    title='Confusion Matrix',
                    xlabel='Predicted Value',
                    ylabel='Actual Value',
                    )
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.markdown('**Step 2.4: Classification Report**')
                report = classification_report(y_test, grid_predictions, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report.head())

                #Measurement for model train time
                st.markdown('**Step 2.5: Training Time(s)**')
                TrainingTime = stop-start
                st.write("%.2f" % TrainingTime)

                #Input of new predictive parameters
                st.write("**Step 2.6: Upload Results for Group Prediction**")
                st.write("Upload the new results via CSV where the features included match those used to train the model in section 1.1")
                uploaded_results = st.file_uploader(label = "Upload your CSV File containing Group Results (200MB max)", type=['csv'])
                if uploaded_results is not None:
                    df_newresults = pd.read_csv(uploaded_results, delimiter=',')
                    st.dataframe(df_newresults)

                    # -- If Condition to make sure group results features match the trained models features

                    #use set instead of list as set can account for unordered list. Using list wont consider user selecting features in same order as that uploaded
                    if set(df_choice.columns) == set(df_newresults.columns):
                        
                        input_data_group = []
                        for row in df_newresults.values:
                            input_data_as_numpy_array = np.asarray(row)
                            # reshape the array as we are predicting for one instance
                            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            scaler.fit(df_newresults.values)

                            std_data = scaler.transform(input_data_reshaped)
                            #print("STD DATA:",std_data)

                            prediction = grid.predict(std_data)
                            pred_str=np.array_str(prediction)
                            #print(pred_str)

                            if pred_str=="[0]":
                                input_data_group.append("AT RISK")
                            else:
                                input_data_group.append("PASS")

                        df_newresults['Prediction']=input_data_group
                        
                        # Upload function for new set of results to predict from
                        st.write("**Step 2.7: Group Prediction Results**")
                        st.write("Upon upload of results for group prediction, click Predict")
                        prediction_button = st.button("Predict")
                        if prediction_button:
                            
                            st.write(df_newresults)

                            from methods import cleandataset_method as cl  # import cleandataset method

                            st.write("**Step 2.8: Download Prediction Results as CSV File**")
                            # Add cleaned dataset csv to filename. Adds _cleaned so can differentiate new file when dowloaded
                            file_name = uploaded_file.name[0:-4] + '_prediction_results.csv'
                            # Remove space to activate download link - replaced with an underscore to remove the space
                            file_name = file_name.replace(" ", "_")
                            st.markdown(cl.csv_downnloadlink(df_newresults, file_name), unsafe_allow_html=True)
                            st.session_state.cleaned_df = df_newresults

                        else:
                            ("Upload New Results for Group Prediction")

                    else:
                        st.write("**ERROR - Features (Column Names) from uploaded Results do not match features from trained model in Section 1.1 - Planned ML Training Dataset Dataframe. Cannot Continue**")
                        st.write("Please do one of the following:")
                        st.write("1. Re-upload Results where features match that of the trained model")
                        st.write("2. Adjust features in Step 1.3 to Re-train the model to match the features in uploaded Results")

        # -- DECISION TREES -- #

            elif task == "Decision Trees":
                class DecisionTree_classifier():
                    import pandas as pd
                    
                import numpy as np
                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # creating the target variable - programming score
                X = df_choice #Make X feature for training model match the features selected by user in dropdown
                y = df['Trans_Programming_Score']

                #Start timer to record train time of model
                import time
                start = time.time()

                # Sidebar - Specify parameter settings
                with st.sidebar.subheader('Test/Train Split Ratio'):
                    parameter_test_size = st.sidebar.slider(
                        '(E.g. 20/80 split = 20% Test / 80% Train)', 0.1, 0.9, 0.2, 0.1)

                # Splitting the dataset into test/train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=999)
                # random state needs optimised via trial and error. 101 seems good

                # Standardise the dataset - Between 0 and 1. Reduce where there can possibly be outliers that carry weight
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)

                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()

                #transform X_trainvalues between 0 and 1 to remove negative values
                X_train = min_max_scaler.fit_transform(X_train)

                print(X_train)

                # Length of Train data
                len(X_train)

                # Length of Test data
                len(X_test)

                # Optimise hyper-parameters when training Model

                # GridSearch - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV

                dt = DecisionTreeClassifier(random_state=999)

                params={
                        'max_depth': [2, 3, 5, 10, 20],
                        'min_samples_leaf': [5, 10, 20, 50, 100],
                        'criterion': ["gini", "entropy"]
                        }

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
                grid = GridSearchCV(estimator=dt, 
                            param_grid=params, 
                            cv=5, n_jobs=-1, verbose=1, scoring = "accuracy")
                #%%time
                grid.fit(X_train, y_train)

                #Stop timer to measure train timer for model
                stop = time.time()

                # Find the optimal parameters
                print(grid.best_params_)

                # print how our model looks after hyper-parameter tuning
                print(grid.best_estimator_)
                print('Model after tuning of Hyper-Parameters : ',
                    grid.best_estimator_)

                # Test set predictions - Confusion Matrix and Classification Report
                grid_predictions = grid.predict(X_test)
                print('Confusion Matrix Report ',
                    confusion_matrix(y_test, grid_predictions))
                print('Classification Report ',
                    classification_report(y_test, grid_predictions))


                st.subheader('2. Model Performance')
                st.markdown('**Step 2.1: Feature Importance (XGBoost)**')
                st.write(
                    "**Feature importance** refers to a class of techniques for assigning a score to an input feature to a predictive model, that indicates the relative importance when making a prediction")

                # xgboost for feature importance on a classification problem
                from xgboost import XGBClassifier

                # define the model
                model = XGBClassifier()

                # fit the model
                model.fit(X_train, y_train)

                # get importance
                importance = model.feature_importances_
                importancelist = importance.tolist()

                # print Feature hearders for bar chart
                st.write(X.columns)
                st.write(importance)
                fig = (importancelist)

                try:
                # seaborn visualisation using correlation plot to identify relationships
                    feature_selection_plot = st.checkbox("Activate Feature Selection Plot")
                    if feature_selection_plot:
                        
                        st.bar_chart(fig)

                except ValueError:
                    st.write('Dataset is not appropriate for Feature Selection')

                st.markdown('**Step 2.2: Hyper-Parameter Tuning Results**')
                st.write('Model after tuning of Hyper-Parameters : ',
                        grid.best_estimator_)
                st.write("Test/Train Split (sidebar) : ", parameter_test_size, " / ", (1-parameter_test_size))

                #print Confusion Marix Chart        
                st.markdown('**Step 2.3: Confusion Matrix Report**')
                #st.write(confusion_matrix(y_test, grid_predictions))

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                confusion_matrix = ConfusionMatrixDisplay(cconf_matrix)
                plt.figure(figsize=(20, 20))
                confusion_matrix.plot(cmap='Reds')
                confusion_matrix.ax_.set(
                    title='Confusion Matrix',
                    xlabel='Predicted Value',
                    ylabel='Actual Value',
                    )
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.markdown('**Step 2.4: Classification Report**')
                report = classification_report(y_test, grid_predictions, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report.head())

                #Measurement for model train time
                st.markdown('**Step 2.5: Training Time(s)**')
                TrainingTime = stop-start
                st.write("%.2f" % TrainingTime)

                #Input of new predictive parameters
                st.write("**Step 2.6: Upload Results for Group Prediction**")
                st.write("Upload the new results via CSV where the features included match those used to train the model in section 1.1")
                uploaded_results = st.file_uploader(label = "Upload your CSV File containing Group Results (200MB max)", type=['csv'])
                if uploaded_results is not None:
                    df_newresults = pd.read_csv(uploaded_results, delimiter=',')
                    st.dataframe(df_newresults)

                    # -- If Condition to make sure group results features match the trained models features

                    #use set instead of list as set can account for unordered list. Using list wont consider user selecting features in same order as that uploaded
                    if set(df_choice.columns) == set(df_newresults.columns):
                        
                        input_data_group = []
                        for row in df_newresults.values:
                            input_data_as_numpy_array = np.asarray(row)
                            # reshape the array as we are predicting for one instance
                            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            #scaler.fit(df.drop('Trans_Programming_Score',axis=1).values) #.values is to ensure headers not included
                            scaler.fit(df_newresults.values)

                            std_data = scaler.transform(input_data_reshaped)
                            #print("STD DATA:",std_data)

                            prediction = grid.predict(std_data)
                            pred_str=np.array_str(prediction)
                            #print(pred_str)

                            if pred_str=="[0]":
                                input_data_group.append("AT RISK")
                            else:
                                input_data_group.append("PASS")

                        df_newresults['Prediction']=input_data_group
                        
                        # Upload function for new set of results to predict from
                        st.write("**Step 2.7: Group Prediction Results**")
                        st.write("Upon upload of results for group prediction, click Predict")
                        prediction_button = st.button("Predict")
                        if prediction_button:
                            
                            st.write(df_newresults)

                            from methods import cleandataset_method as cl  # import cleandataset method

                            st.write("**Step 2.8: Download Prediction Results as CSV File**")
                            # Add cleaned dataset csv to filename. Adds _cleaned so can differentiate new file when dowloaded
                            file_name = uploaded_file.name[0:-4] + '_prediction_results.csv'
                            # Remove space to activate download link - replaced with an underscore to remove the space
                            file_name = file_name.replace(" ", "_")
                            st.markdown(cl.csv_downnloadlink(df_newresults, file_name), unsafe_allow_html=True)
                            st.session_state.cleaned_df = df_newresults

                        else:
                            ("Upload New Results for Group Prediction")
                    

                    else:
                        st.write("**ERROR - Features (Column Names) from uploaded Results do not match features from trained model in Section 1.1 - Planned ML Training Dataset Dataframe. Cannot Continue**")
                        st.write("Please do one of the following:")
                        st.write("1. Re-upload Results where features match that of the trained model")
                        st.write("2. Adjust features in Step 1.3 to Re-train the model to match the features in uploaded Results")

        # -- NAIVE BAYES -- #

            elif task == "Naive Bayes":
                class NaiveBayes_classifier():
                    import pandas as pd
            
                import numpy as np
                from sklearn.naive_bayes import BernoulliNB
                from sklearn.naive_bayes import GaussianNB
                from sklearn.naive_bayes import MultinomialNB
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import classification_report, confusion_matrix
                from matplotlib import pyplot as plt 

                # returns first 5 rows as dataframe
                df.head()

                # creating the target variable - programming score
                X = df_choice #Make X feature for training model match the features selected by user in dropdown
                y = df['Trans_Programming_Score']

                #Start timer to record train time of model
                import time
                start = time.time()

                # Sidebar - Specify parameter settings
                with st.sidebar.subheader('Test/Train Split Ratio'):
                    parameter_test_size = st.sidebar.slider(
                        '(E.g. 20/80 split = 20% Test / 80% Train)', 0.1, 0.9, 0.2, 0.1)

                # Splitting the dataset into test/train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=999)

                # Standardise the dataset - Between 0 and 1. Reduce where there can possibly be outliers that carry weight
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)

                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()

                #transform X_trainvalues between 0 and 1 to remove negative values
                X_train = min_max_scaler.fit_transform(X_train)       

                print(X_train)

                # Length of Train data
                len(X_train)

                # Length of Test data
                len(X_test)

                #Find best Naive Bayes Model

                #Bernoulli
                BernNB = BernoulliNB(binarize=True)
                BernNB.fit(X_train, y_train)
                print(BernNB)

                y_expect = y_test
                y_pred = BernNB.predict(X_test)
                print(accuracy_score(y_expect, y_pred))

                #MultiNomial - X_Train needs standardised between 0 and 1
                MultiNB = MultinomialNB()

                MultiNB.fit(X_train, y_train)
                print(MultiNB)

                y_expect = y_test
                y_pred = MultiNB.predict(X_test)
                print(accuracy_score(y_expect, y_pred))

                #Gaussian - best performing
                GausNB = GaussianNB()
                GausNB.fit(X_train, y_train)
                print(GausNB)

                #y_expect = GausNB.predict(X_test)
                #print(accuracy_score(y_expect, y_pred))


                # Train The Model - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV
                from sklearn.model_selection import RepeatedStratifiedKFold

                cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)

                from sklearn.preprocessing import PowerTransformer
                param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

                # What fit does is a bit more involved than usual. First, it runs the same loop with cross-validation, to find the best parameter combination.
                # Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
                # You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best_estimator_ attribute:

                # does 5 fold cross-validation and takes average
                grid = GridSearchCV(GaussianNB(),param_grid = param_grid,cv=cv_method,verbose=1,scoring='accuracy')
                
                Data_transformed = PowerTransformer().fit_transform(X_test)
                grid.fit(Data_transformed, y_test)

                #Stop timer to measure train timer for model
                stop = time.time()

                # Find the optimal parameters
                print(grid.best_params_)

                # print how our model looks after hyper-parameter tuning
                print(grid.best_estimator_)
                print('Model after tuning of Hyper-Parameters : ',
                    grid.best_estimator_)

                # Test set predictions - Confusion Matrix and Classification Report
                grid_predictions = grid.predict(X_test)
                print('Confusion Matrix Report ',
                    confusion_matrix(y_test, grid_predictions))
                print('Classification Report ',
                    classification_report(y_test, grid_predictions))

                # Print Model Performance

                st.subheader('2. Model Performance')
                st.markdown('**Step 2.1: Feature Importance (XGBoost)**')
                st.write(
                    "**Feature importance** refers to a class of techniques for assigning a score to an input feature to a predictive model, that indicates the relative importance when making a prediction")

                # xgboost for feature importance on a classification problem
                from xgboost import XGBClassifier

                # define the model
                model = XGBClassifier()

                # fit the model
                model.fit(X_train, y_train)

                # get importance
                importance = model.feature_importances_

                importancelist = importance.tolist()
                # print Feature hearders for bar chart
                st.write(X.columns)
                st.write(importance)
                
                fig = (importancelist)

                try:
                # seaborn visualisation using correlation plot to identify relationships
                    feature_selection_plot = st.checkbox("Activate Feature Selection Plot")
                    if feature_selection_plot:
                        
                        st.bar_chart(fig)

                except ValueError:
                    st.write('Dataset is not appropriate for Feature Selection')

                st.markdown('**Step 2.2: Hyper-Parameter Tuning Results**')
                st.write('Model after tuning of Hyper-Parameters : ',
                        grid.best_estimator_)
                st.write("Test/Train Split (sidebar) : ", parameter_test_size, " / ", (1-parameter_test_size))

                st.markdown('**Step 2.3: Confusion Matrix Report**')
                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                confusion_matrix = ConfusionMatrixDisplay(cconf_matrix)
                plt.figure(figsize=(20, 20))
                confusion_matrix.plot(cmap='Reds')
                confusion_matrix.ax_.set(
                    title='Confusion Matrix',
                    xlabel='Predicted Value',
                    ylabel='Actual Value',
                    )
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.markdown('**Step 2.4: Classification Report**')
                report = classification_report(y_test, grid_predictions, output_dict=True)
                df = pd.DataFrame(report).transpose()
                st.write(df.head())

                #Measurement for model train time
                st.markdown('**Step 2.5: Training Time(s)**')
                TrainingTime = stop-start
                st.write("%.2f" % TrainingTime)

                #Input of new predictive parameters
                st.write("**Step 2.6: Upload Results for Group Prediction**")
                st.write("Upload the new results via CSV where the features included match those used to train the model in section 1.1")
                uploaded_results = st.file_uploader(label = "Upload your CSV File containing Group Results (200MB max)", type=['csv'])
                if uploaded_results is not None:
                    df_newresults = pd.read_csv(uploaded_results, delimiter=',')
                    st.dataframe(df_newresults)

                    # -- If Condition to make sure group results features match the trained models features

                    #use set instead of list as set can account for unordered list. Using list wont consider user selecting features in same order as that uploaded
                    if set(df_choice.columns) == set(df_newresults.columns):
                        
                        input_data_group = []
                        for row in df_newresults.values:
                            input_data_as_numpy_array = np.asarray(row)
                            # reshape the array as we are predicting for one instance
                            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            #scaler.fit(df.drop('Trans_Programming_Score',axis=1).values) #.values is to ensure headers not included
                            scaler.fit(df_newresults.values)

                            std_data = scaler.transform(input_data_reshaped)
                            #print("STD DATA:",std_data)

                            prediction = grid.predict(std_data)
                            pred_str=np.array_str(prediction)
                            #print(pred_str)

                            if pred_str=="[0]":
                                input_data_group.append("AT RISK")
                            else:
                                input_data_group.append("PASS")

                        df_newresults['Prediction']=input_data_group
                        
                        # Upload function for new set of results to predict from
                        st.write("**Step 2.7: Group Prediction Results**")
                        st.write("Upon upload of results for group prediction, click Predict")
                        prediction_button = st.button("Predict")
                        if prediction_button:
                            
                            st.write(df_newresults)

                            from methods import cleandataset_method as cl  # import cleandataset method

                            st.write("**Step 2.8: Download Prediction Results as CSV File**")
                            # Add cleaned dataset csv to filename. Adds _cleaned so can differentiate new file when dowloaded
                            file_name = uploaded_file.name[0:-4] + '_prediction_results.csv'
                            # Remove space to activate download link - replaced with an underscore to remove the space
                            file_name = file_name.replace(" ", "_")
                            st.markdown(cl.csv_downnloadlink(df_newresults, file_name), unsafe_allow_html=True)
                            st.session_state.cleaned_df = df_newresults

                        else:
                            ("Upload New Results for Group Prediction")
                    
                    else:
                        st.write("**ERROR - Features (Column Names) from uploaded Results do not match features from trained model in Section 1.1 - Planned ML Training Dataset Dataframe. Cannot Continue**")
                        st.write("Please do one of the following:")
                        st.write("1. Re-upload Results where features match that of the trained model")
                        st.write("2. Adjust features in Step 1.3 to Re-train the model to match the features in uploaded Results")
                        

        # -- K-NEAREST NEIGHBOR -- #

            elif task == "K-Nearest Neighbor":
                class KNearestNeighbor():
                    import pandas as pd
                    
                import numpy as np
                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # creating the target variabel - programming score
                X = df_choice #Make X feature for training model match the features selected by user in dropdown
                y = df['Trans_Programming_Score']

                #Start timer to record train time of model
                import time
                start = time.time()

                # Sidebar - Specify parameter settings
                with st.sidebar.subheader('Test/Train Split Ratio'):
                    parameter_test_size = st.sidebar.slider(
                        '(E.g. 20/80 split = 20% Test / 80% Train)', 0.1, 0.9, 0.2, 0.1)

                # Splitting the dataset into test/train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=parameter_test_size, random_state=101)

                # Standardise the dataset - Between 0 and 1. Reduce where there can possibly be outliers that carry weight
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)

                # scale the data set between 0 and 1 to remove negative values
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()

                X_train = min_max_scaler.fit_transform(X_train)

                print(X_train)

                # Length of Train data
                len(X_train)

                # Length of Test data
                len(X_test)

                # Optimise hyper-parameters when training Model

                # GridSearch - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma
                
                from sklearn.model_selection import GridSearchCV

                #List Hyperparameters that we want to tune
                grid_params = { 'n_neighbors' : [5,7,9,11,13,15,17],
                        'weights' : ['uniform','distance'],
                        'metric' : ['minkowski','euclidean','manhattan']
                            }

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
                grid = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=5, n_jobs = -1)
                
                # fit the model on our train set
                g_res = grid.fit(X_train, y_train)

                # find the best score
                g_res.best_score_

                # get the hyperparameters with the best score
                g_res.best_params_

                #Stop timer to measure train timer for model
                stop = time.time()

                # use the best hyperparameters
                knn = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform',algorithm = 'brute',metric = 'minkowski')
                knn.fit(X_train, y_train)

                # Find the optimal parameters
                print(grid.best_params_)

                # print how our model looks after hyper-parameter tuning
                print(grid.best_estimator_)
                print('Model after tuning of Hyper-Parameters : ',
                    grid.best_estimator_)

                # Test set predictions - Confusion Matrix and Classification Report
                grid_predictions = grid.predict(X_test)
                print('Confusion Matrix Report ',
                    confusion_matrix(y_test, grid_predictions))
                print('Classification Report ',
                    classification_report(y_test, grid_predictions))

                st.subheader('2. Model Performance')
                st.markdown('**Step 2.1: Feature Importance (XGBoost)**')
                st.write(
                    "**Feature importance** refers to a class of techniques for assigning a score to an input feature to a predictive model, that indicates the relative importance when making a prediction")

                # xgboost for feature importance on a classification problem
                from xgboost import XGBClassifier

                # define the model
                model = XGBClassifier()

                # fit the model
                model.fit(X_train, y_train)

                # get importance
                importance = model.feature_importances_
                importancelist = importance.tolist()

                # print Feature hearders for bar chart
                st.write(X.columns)
                st.write(importance)
                fig = (importancelist)

                try:
                # seaborn visualisation using correlation plot to identify relationships
                    feature_selection_plot = st.checkbox("Activate Feature Selection Plot")
                    if feature_selection_plot:
                        
                        st.bar_chart(fig)

                except ValueError:
                    st.write('Dataset is not appropriate for Feature Selection')

                st.markdown('**Step 2.2: Hyper-Parameter Tuning Results**')
                st.write('Model after tuning of Hyper-Parameters : ',
                        grid.best_estimator_)
                st.write("Test/Train Split (sidebar) : ", parameter_test_size, " / ", (1-parameter_test_size))

                #print Confusion Marix Chart
                st.markdown('**Step 2.3: Confusion Matrix Report**')

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                confusion_matrix = ConfusionMatrixDisplay(cconf_matrix)
                plt.figure(figsize=(20, 20))
                confusion_matrix.plot(cmap='Reds')
                confusion_matrix.ax_.set(
                    title='Confusion Matrix',
                    xlabel='Predicted Value',
                    ylabel='Actual Value',
                    )
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.markdown('**Step 2.4: Classification Report**')
                report = classification_report(y_test, grid_predictions, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report.head())

                #Measurement for model train time
                st.markdown('**Step 2.5: Training Time(s)**')
                TrainingTime = stop-start
                st.write("%.2f" % TrainingTime)

                #Input of new predictive parameters
                st.write("**Step 2.6: Upload Results for Group Prediction**")
                st.write("Upload the new results via CSV where the features included match those used to train the model in section 1.1")
                uploaded_results = st.file_uploader(label = "Upload your CSV File containing Group Results (200MB max)", type=['csv'])
                if uploaded_results is not None:
                    df_newresults = pd.read_csv(uploaded_results, delimiter=',')
                    st.dataframe(df_newresults)

                    # -- If Condition to make sure group results features match the trained models features

                    #use set instead of list as set can account for unordered list. Using list wont consider user selecting features in same order as that uploaded
                    if set(df_choice.columns) == set(df_newresults.columns):
                        
                        input_data_group = []
                        for row in df_newresults.values:
                            input_data_as_numpy_array = np.asarray(row)
                            # reshape the array as we are predicting for one instance
                            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            #scaler.fit(df.drop('Trans_Programming_Score',axis=1).values) #.values is to ensure headers not included
                            scaler.fit(df_newresults.values)

                            std_data = scaler.transform(input_data_reshaped)
                            #print("STD DATA:",std_data)

                            prediction = grid.predict(std_data)
                            pred_str=np.array_str(prediction)
                            #print(pred_str)

                            if pred_str=="[0]":
                                input_data_group.append("AT RISK")
                            else:
                                input_data_group.append("PASS")

                        df_newresults['Prediction']=input_data_group
                        
                        # Upload function for new set of results to predict from
                        st.write("**Step 2.7: Group Prediction Results**")
                        st.write("Upon upload of results for group prediction, click Predict")
                        prediction_button = st.button("Predict")
                        if prediction_button:
                            
                            st.write(df_newresults)

                            from methods import cleandataset_method as cl  # import cleandataset method

                            st.write("**Step 2.8: Download Prediction Results as CSV File**")
                            # Add cleaned dataset csv to filename. Adds _cleaned so can differentiate new file when dowloaded
                            file_name = uploaded_file.name[0:-4] + '_prediction_results.csv'
                            # Remove space to activate download link - replaced with an underscore to remove the space
                            file_name = file_name.replace(" ", "_")
                            st.markdown(cl.csv_downnloadlink(df_newresults, file_name), unsafe_allow_html=True)
                            st.session_state.cleaned_df = df_newresults

                        else:
                            ("Upload New Results for Group Prediction")

                    else:
                        st.write("**ERROR - Features (Column Names) from uploaded Results do not match features from trained model in Section 1.1 - Planned ML Training Dataset Dataframe. Cannot Continue**")
                        st.write("Please do one of the following:")
                        st.write("1. Re-upload Results where features match that of the trained model")
                        st.write("2. Adjust features in Step 1.3 to Re-train the model to match the features in uploaded Results")
        
        else:
            ("Please select a classifier from the dropdown")
