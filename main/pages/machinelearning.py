from os import write
from matplotlib import pyplot
from scipy.sparse import data
from seaborn.axisgrid import pairplot

from methods import machine_learn_method as ml
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns  # Python visualisation library based upon matplotlib
import plotly_express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def machinelearningdisplay():

    # dataframe as false before if conditions
    df_file = False

    st.subheader("1. Model Training")

    st.sidebar.subheader('Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader(
        label="Ensure Dataset is cleaned prior to Upload. Visit Clean Dataset Page.", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values="NaN")
        df_file = True

    else:
        st.write("Upload Cleaned Dataset using the Sidebar")
        #st.button("Use already cleaned dataset???")

    if df_file == True:
        st.write('**Step 1.1:  Uploaded Cleaned Dataset Dataframe**')
        st.write(df)
        headers = list(df.columns)

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
                st.write('Correlation Ranking > 0.1')
                st.write(relevant_features)

        except ValueError:
            st.write('Dataset is not appropriate for Correlation Matrix')

        #global numeric_columns
        #global non_numeric_columns
        #try:
            # seaborn visualisation using graph plot
            #graph = st.checkbox("Activate Data Plot")
            #if graph:
                #st.write('**Data Plot**')
                #chart_select = st.sidebar.selectbox(
                    #label="Select the chart type",
                    #options=['Scatterplots', 'Histogram', 'Boxplot'])

                #numeric_columns = list(
                    #df.select_dtypes(['float', 'int']).columns)
                #non_numeric_columns = list(
                    #df.select_dtypes(['object']).columns)
                #non_numeric_columns.append(None)
                #print(non_numeric_columns)

                #if chart_select == 'Scatterplots':
                    #st.sidebar.subheader("Scatterplot Settings")
                    #try:
                        #x_values = st.sidebar.selectbox(
                            #'X axis', options=numeric_columns)
                        #y_values = st.sidebar.selectbox(
                            #'Y axis', options=numeric_columns)
                        #color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                        #plot = px.scatter(
                            #data_frame=df, x=x_values, y=y_values)

                        # display the chart
                        #st.plotly_chart(plot)

                    #except Exception as e:
                        #print(e)

                #elif chart_select == 'Histogram':
                    #st.sidebar.subheader("Histogram Settings")
                    #try:
                        #x_values = st.sidebar.selectbox(
                            #'X axis', options=numeric_columns)
                        #y_values = st.sidebar.selectbox(
                            #'Y axis', options=numeric_columns)
                        #color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                        #plot = px.histogram(data_frame=df, x=x_values, y=y_values)
                        #st.plotly_chart(plot)
                    #except Exception as e:
                        #print(e)

                #elif chart_select == 'Boxplot':
                    #st.sidebar.subheader("Boxplot Settings")
                #try:
                    #y = st.sidebar.selectbox("Y axis", options=numeric_columns)
                    #x = st.sidebar.selectbox(
                        #"X axis", options=non_numeric_columns)
                    #color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                    #plot = px.box(data_frame=df, y=y, x=x)
                    #st.plotly_chart(plot)
                #except Exception as e:
                    #print(e)

        #except ValueError:
            #st.write('Dataset is not appropriate for Graph Plot')

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
            #df = df.drop_duplicates()
            df_dropdrown = df.drop('Trans_Programming_Score', axis=1)
            make_choice = st.multiselect('Select from dropdown', df_dropdrown.columns)
            df_choice = df_dropdrown[make_choice]
            if make_choice:
                st.write('Planned ML Training Dataset Dataframe : ')
                st.write(df_choice)
            
        except ValueError:
            st.write('Error with Feature Selection for Dataframe')

        if make_choice:

            st.write('**Step 1.4:  Select the Machine Learning Classifier to Train**')
            task = st.selectbox("Select the Machine Learning Classifier to Train", [
                                "< Please select a Classification Model >", "Decision Trees", "Support Vector Machines (SVM)", "Naive Bayes", "Random Forest", "Logistic Regression", "K-Nearest Neighbor"])

    # -- Support Vector Machines -- #

 
            if task == "Support Vector Machines (SVM)":
                class SVM_classifier():
                    import pandas as pd

                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # returns first 5 rows as dataframe
                df.head()

                # creating the target variable - programming score
                #X = df.drop(columns='Trans_Programming_Score', axis=1)
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

                # Scale/normalize the dataset - Between 0 and 1. This normalizes the data where there can possibly be outliers that carry weight
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

                # Train The Model - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the SVC() (SVC=SVM Model) model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV

                param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
                    1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

                # What fit does is a bit more involved than usual. First, it runs the same loop with cross-validation, to find the best parameter combination.
                # Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
                # You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best_estimator_ attribute:

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
                grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=10)
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

                # - Making a Predictive System - #

                # Below 70 Programming score (Outcome 0)
                # 70+ Programming score (Outcome 1)

                with st.sidebar.subheader('New Input Parameters'):
                    input_data=[]
                    for x in make_choice:
                        #sidebar_params = st.sidebar.slider(x)
                        if x == 'Trans_Acadmode': # Required for those features with with either a 0 or 1 for scaling
                            al_parameter = st.sidebar.slider('Acad Mode: 0 = Full-Time, 1 = Part-Time', 0, 1, 0, 1)
                        elif x == 'Trans_Sex': # Required for those features with with either a 0 or 1 for scaling
                            st.sidebar.slider('Sex: 0 = Male, 1 = Female', 0, 1, 0, 1)
                        else:
                            al_parameter=st.sidebar.slider(x)
                        input_data.append(al_parameter)

                # changing the input_data to numpy array
                import numpy as np

                input_data_as_numpy_array = np.asarray(input_data)

                # reshape the array as we are predictin g for one instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                # standardize the input data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                #scaler.fit(df.drop('Trans_Programming_Score', axis=1).values)
                scaler.fit(df_choice.values)
                std_data = scaler.transform(input_data_reshaped)
                print(std_data)

                # print prediction based on input data
                prediction = grid.predict(std_data)
                print(prediction)

                # Print Model Performance

                st.subheader('2. Model Performance')

                #st.markdown('**2.1. Standardised Training Dataset**')
                #df_input_names = pd.DataFrame(std_data, columns = make_choice)
                #st.table(df_input_names)
                #st.write(std_data)

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
                #st.write(confusion_matrix(y_test, grid_predictions))

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                #dl = list(set(df_newresults[model_class]))
                #dl = sorted(dl)

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

                st.write("**Step 2.6: New Input Parameters for Prediction**")
                st.write("Update the sidebar with new parameters for prediction")
                
                df_inputs = pd.DataFrame(input_data_reshaped, columns = make_choice) 
                st.table(df_inputs)

                st.write("**Step 2.7: Single Entry Prediction Result**")
                st.write("Upon update of sidebar with new input parameters, click Predict")
                prediction_button = st.button("Predict")

                if prediction_button:

                    if (prediction[0] == 0):
                        st.info('Programming Module Prediction: **AT RISK**')
                    else:
                        st.info('Programming Module Prediction: **PASS**')
                        st.balloons()

                else:
                    ("Enter Results in Sidebar and Click Predict")

        # -- RANDOM FOREST CLASSIFIER -- #

            elif task == "Random Forest":
                class RandomForest_classifier():

                    import pandas as pd

                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # creating the target variable - programming score
                #X = df.drop(columns='Trans_Programming_Score', axis=1)
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

                # Scale/normalize the dataset - Between 0 and 1. This normalizes the data where there can possibly be outliers that carry weight
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

                # Train The Model - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the SVC() (SVC=SVM Model) model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV

                param_grid={'n_estimators':[10,50,100,150,200],'min_samples_leaf':[1,5,10,15,50],'max_features':('auto','sqrt','log2')} 

                # What fit does is a bit more involved than usual. First, it runs the same loop with cross-validation, to find the best parameter combination.
                # Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
                # You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best_estimator_ attribute:

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
                grid = GridSearchCV(RandomForestClassifier(),param_grid, cv=5, refit=True,verbose=2)
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

                # - Making a Predictive System - #

                # Below 70 Programming score (Outcome 0)
                # 70+ Programming score (Outcome 1)

                with st.sidebar.subheader('New Input Parameters'):
                    input_data=[]
                    for x in make_choice:
                        #sidebar_params = st.sidebar.slider(x)
                        if x == 'Trans_Acadmode': # Required for those features with with either a 0 or 1 for scaling
                            al_parameter = st.sidebar.slider('Acad Mode: 0 = Full-Time, 1 = Part-Time', 0, 1, 0, 1)
                        elif x == 'Trans_Sex': # Required for those features with with either a 0 or 1 for scaling
                            st.sidebar.slider('Sex: 0 = Male, 1 = Female', 0, 1, 0, 1)
                        else:
                            al_parameter=st.sidebar.slider(x)
                        input_data.append(al_parameter)
                
                # changing the input_data to numpy array
                import numpy as np

                input_data_as_numpy_array = np.asarray(input_data)

                # reshape the array as we are predictin g for one instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                # standardize the input data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                #scaler.fit(df.drop('Trans_Programming_Score', axis=1).values)
                scaler.fit(df_choice.values)
                std_data = scaler.transform(input_data_reshaped)
                print(std_data)

                # print prediction based on input data
                prediction = grid.predict(std_data)
                print(prediction)

                # Print Model Performance

                st.subheader('2. Model Performance')

                #st.markdown('**2.1. Standardised Training Dataset**')
                #df_input_names = pd.DataFrame(std_data, columns = make_choice)
                #st.table(df_input_names)
                #st.write(std_data)

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
                #st.write(confusion_matrix(y_test, grid_predictions))

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                #dl = list(set(df_newresults[model_class]))
                #dl = sorted(dl)

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

                st.write("**Step 2.6: New Input Parameters for Prediction**")
                st.write("Update the sidebar with new parameters for prediction")
                df_inputs = pd.DataFrame(input_data_reshaped, columns = make_choice) 
                st.table(df_inputs)

                st.write("**Step 2.7: Single Entry Prediction Result**")
                st.write("Upon update of sidebar with new input parameters, click Predict")
                prediction_button = st.button("Predict")

                if prediction_button:

                    if (prediction[0] == 0):
                        st.info('Programming Module Prediction: **AT RISK**')
                    else:
                        st.info('Programming Module Prediction: **PASS**')
                        st.balloons()

                else:
                    ("Enter Results in Sidebar and Click Predict")

        # -- LOGISTIC REGRESSION -- #

            elif task == "Logistic Regression":
                class LogisticRegression_classifier():

                    import pandas as pd
                    

                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # createing the target variable - programming score
                #X = df.drop(columns='Trans_Programming_Score', axis=1)
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
         

                # Scale/normalize the dataset - Between 0 and 1. This normalizes the data where there can possibly be outliers that carry weight
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

                # Train The Model - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the SVC() (SVC=SVM Model) model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV
                import numpy as np

                param_grid={"C":np.logspace(-3,3,7), "penalty":["l2"]} # l1 lasso l2 ridge 

                # What fit does is a bit more involved than usual. First, it runs the same loop with cross-validation, to find the best parameter combination.
                # Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
                # You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best_estimator_ attribute:

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
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

                # - Making a Predictive System - #

                # Below 70 Programming score (Outcome 0)
                # 70+ Programming score (Outcome 1)

                with st.sidebar.subheader('New Input Parameters'):
                    input_data=[]
                    for x in make_choice:
                        #sidebar_params = st.sidebar.slider(x)
                        if x == 'Trans_Acadmode': # Required for those features with with either a 0 or 1 for scaling
                            al_parameter = st.sidebar.slider('Acad Mode: 0 = Full-Time, 1 = Part-Time', 0, 1, 0, 1)
                        elif x == 'Trans_Sex': # Required for those features with with either a 0 or 1 for scaling
                            st.sidebar.slider('Sex: 0 = Male, 1 = Female', 0, 1, 0, 1)
                        else:
                            al_parameter=st.sidebar.slider(x)
                        input_data.append(al_parameter)
          
                # changing the input_data to numpy array
                import numpy as np

                input_data_as_numpy_array = np.asarray(input_data)

                # reshape the array as we are predictin g for one instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                 # standardize the input data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                #scaler.fit(df.drop('Trans_Programming_Score', axis=1).values)
                scaler.fit(df_choice.values)
                std_data = scaler.transform(input_data_reshaped)
                print(std_data)

                # print prediction based on input data
                prediction = grid.predict(std_data)
                print(prediction)

                # Print Model Performance

                st.subheader('2. Model Performance')

                #st.markdown('**2.1. Standardised Training Dataset**')
                #df_input_names = pd.DataFrame(std_data, columns = make_choice)
                #st.table(df_input_names)
                #st.write(std_data)

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
                #st.write(confusion_matrix(y_test, grid_predictions))

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                #dl = list(set(df_newresults[model_class]))
                #dl = sorted(dl)

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

                st.write("**Step 2.6: New Input Parameters for Prediction**")
                st.write("Update the sidebar with new parameters for prediction")
                
                df_inputs = pd.DataFrame(input_data_reshaped, columns = make_choice) 
                st.table(df_inputs)

                st.write("**Step 2.7: Single Entry Prediction Result**")
                st.write("Upon update of sidebar with new input parameters, click Predict")
                prediction_button = st.button("Predict")

                if prediction_button:

                    if (prediction[0] == 0):
                        st.info('Programming Module Prediction: **AT RISK**')
                    else:
                        st.info('Programming Module Prediction: **PASS**')
                        st.balloons()

                else:
                    ("Enter Results in Sidebar and Click Predict")

        # -- DECISION TREES -- #

            elif task == "Decision Trees":
                class DecisionTree_classifier():

                    import pandas as pd
                    

                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # createing the target variable - programming score
                #X = df.drop(columns='Trans_Programming_Score', axis=1)
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

                # Scale/normalize the dataset - Between 0 and 1. This normalizes the data where there can possibly be outliers that carry weight
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

                # Train The Model - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the SVC() (SVC=SVM Model) model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma

                from sklearn.model_selection import GridSearchCV
                #import numpy as np

                dt = DecisionTreeClassifier(random_state=999)

                params={
                        'max_depth': [2, 3, 5, 10, 20],
                        'min_samples_leaf': [5, 10, 20, 50, 100],
                        'criterion': ["gini", "entropy"]
                        }

                # What fit does is a bit more involved than usual. First, it runs the same loop with cross-validation, to find the best parameter combination.
                # Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
                # You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best_estimator_ attribute:

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
                grid = GridSearchCV(estimator=dt, 
                            param_grid=params, 
                            cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
                #%%time
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

                # - Making a Predictive System - #

                # Below 70 Programming score (Outcome 0)
                # 70+ Programming score (Outcome 1)

                with st.sidebar.subheader('New Input Parameters'):
                    input_data=[]
                    for x in make_choice:
                        #sidebar_params = st.sidebar.slider(x)
                        if x == 'Trans_Acadmode': # Required for those features with with either a 0 or 1 for scaling
                            al_parameter = st.sidebar.slider('Acad Mode: 0 = Full-Time, 1 = Part-Time', 0, 1, 0, 1)
                        elif x == 'Trans_Sex': # Required for those features with with either a 0 or 1 for scaling
                            st.sidebar.slider('Sex: 0 = Male, 1 = Female', 0, 1, 0, 1)
                        else:
                            al_parameter=st.sidebar.slider(x)
                        input_data.append(al_parameter)

                # changing the input_data to numpy array
                import numpy as np
    
                input_data_as_numpy_array = np.asarray(input_data)

                # reshape the array as we are predicting for one instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                # standardize the input data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                #scaler.fit(df.drop('Trans_Programming_Score', axis=1).values)
                scaler.fit(df_choice.values)
                std_data = scaler.transform(input_data_reshaped)
                print(std_data)

                # print prediction based on input data
                prediction = grid.predict(std_data)
                print(prediction)

                # Print Model Performance

                st.subheader('2. Model Performance')

                #st.markdown('**2.1. Standardised Training Dataset**')
                #df_input_names = pd.DataFrame(std_data, columns = make_choice)
                #st.table(df_input_names)
                #st.write(std_data)

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
                #st.write(confusion_matrix(y_test, grid_predictions))

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                #dl = list(set(df_newresults[model_class]))
                #dl = sorted(dl)

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

                st.write("**Step 2.6: New Input Parameters for Prediction**")
                st.write("Update the sidebar with new parameters for prediction")
                
                df_inputs = pd.DataFrame(input_data_reshaped, columns = make_choice) 
                st.table(df_inputs)

                st.write("**Step 2.7: Single Entry Prediction Result**")
                st.write("Upon update of sidebar with new input parameters, click Predict")
                prediction_button = st.button("Predict")

                if prediction_button:

                    if (prediction[0] == 0):
                        st.info('Programming Module Prediction: **AT RISK**')
                    else:
                        st.info('Programming Module Prediction: **PASS**')
                        st.balloons()

                else:
                    ("Enter Results in Sidebar and Click Predict")

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

                from sklearn import metrics
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import classification_report, confusion_matrix
                from matplotlib import pyplot as plt 

                # returns first 5 rows as dataframe
                df.head()

                # createing the target variable - programming score
                #X = df.drop(columns='Trans_Programming_Score', axis=1)
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

                # Scale/normalize the dataset - Between 0 and 1. This normalizes the data where there can possibly be outliers that carry weight
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
                # Call the SVC() (SVC=SVM Model) model from sklearn and fit the model to the training data
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

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
                grid = GridSearchCV(GaussianNB(),param_grid = param_grid,cv=cv_method,verbose=1,scoring='accuracy')
                
                Data_transformed = PowerTransformer().fit_transform(X_test)
                grid.fit(Data_transformed, y_test);

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

                # - Making a Predictive System - #

                # Below 70 Programming score (Outcome 0)
                # 70+ Programming score (Outcome 1)

                # Predict New Input Dataset using trained Model:
               
                with st.sidebar.subheader('New Input Parameters'):
                    input_data=[]
                    for x in make_choice:
                        #sidebar_params = st.sidebar.slider(x)
                        if x == 'Trans_Acadmode': # Required for those features with with either a 0 or 1 for scaling
                            al_parameter = st.sidebar.slider('Acad Mode: 0 = Full-Time, 1 = Part-Time', 0, 1, 0, 1)
                        elif x == 'Trans_Sex': # Required for those features with with either a 0 or 1 for scaling
                            st.sidebar.slider('Sex: 0 = Male, 1 = Female', 0, 1, 0, 1)
                        else:
                            al_parameter=st.sidebar.slider(x)
                        input_data.append(al_parameter)

                # changing the input_data to numpy array
                import numpy as np
                
                input_data_as_numpy_array = np.asarray(input_data)

                # reshape the array as we are predicting for one instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                # standardize the input data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                #scaler.fit(df.drop('Trans_Programming_Score', axis=1).values)
                scaler.fit(df_choice.values)
                std_data = scaler.transform(input_data_reshaped)
                print("Std Data : ", std_data)

                # print prediction based on input data
                prediction = grid.predict(std_data)
                print("Prediction : ", prediction)

                # Print Model Performance

                st.subheader('2. Model Performance')

                #st.markdown('**2.1. Standardised Training Dataset**')
                #df_input_names = pd.DataFrame(std_data, columns = make_choice)
                #st.table(df_input_names)
                #st.write(std_data)

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
                #st.write(confusion_matrix(y_test, grid_predictions))

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                #dl = list(set(df_newresults[model_class]))
                #dl = sorted(dl)

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

                st.write("**Step 2.6: New Input Parameters for Prediction**")
                st.write("Update the sidebar with new parameters for prediction")
                
                df_inputs = pd.DataFrame(input_data_reshaped, columns = make_choice) 
                st.table(df_inputs)

                st.write("**Step 2.7: Single Entry Prediction Result**")
                st.write("Upon update of sidebar with new input parameters, click Predict")
                prediction_button = st.button("Predict")

                if prediction_button:

                    if (prediction[0] == 0):
                        st.info('Programming Module Prediction: **AT RISK**')
                    else:
                        st.info('Programming Module Prediction: **PASS**')
                        st.balloons()

                else:
                    ("Enter Results in Sidebar and Click Predict")
        

        # -- K-NEAREST NEIGHBOR -- #

            elif task == "K-Nearest Neighbor":
                
                class KNearestNeighbor():

                    import pandas as pd
                    

                from matplotlib import pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                from sklearn.metrics import classification_report, confusion_matrix

                # returns first 5 rows as dataframe
                df.head()

                # createing the target variabel - programming score
                #X = df.drop(columns='Trans_Programming_Score', axis=1)
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
                # random state needs optimised via trial and error. 101 seems good

                # Scale/normalize the dataset - Between 0 and 1. This normalizes the data where there can possibly be outliers that carry weight
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

                # Train The Model - https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
                # Call the SVC() (SVC=SVM Model) model from sklearn and fit the model to the training data
                # Determine which kernel performs the best based on the performance metrics such as precision, recall and f1 score.
                # Fine Tuning the hyper-parameters using grid search. By not setting a value for cv GridSearchCV uses 5 fold cross validation - https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

                # Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma
                
                from sklearn.model_selection import GridSearchCV

                #List Hyperparameters that we want to tune
                grid_params = { 'n_neighbors' : [5,7,9,11,13,15,17],
                        'weights' : ['uniform','distance'],
                        'metric' : ['minkowski','euclidean','manhattan']
                            }

                # What fit does is a bit more involved than usual. First, it runs the same loop with cross-validation, to find the best parameter combination.
                # Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
                # You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best_estimator_ attribute:

                # does 5 fold cross-validation and takes average. Can be changed to 10 if needs be. Doesnt make any diff here
                grid = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=10, n_jobs = -1)
                
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

                # - Making a Predictive System - #

                # Below 70 Programming score (Outcome 0)
                # 70+ Programming score (Outcome 1)
                
                with st.sidebar.subheader('New Input Parameters'):
                    input_data=[]
                    for x in make_choice:
                        #sidebar_params = st.sidebar.slider(x)
                        if x == 'Trans_Acadmode': # Required for those features with with either a 0 or 1 for scaling
                            al_parameter = st.sidebar.slider('Acad Mode: 0 = Full-Time, 1 = Part-Time', 0, 1, 0, 1)
                        elif x == 'Trans_Sex': # Required for those features with with either a 0 or 1 for scaling
                            st.sidebar.slider('Sex: 0 = Male, 1 = Female', 0, 1, 0, 1)
                        else:
                            al_parameter=st.sidebar.slider(x)
                        input_data.append(al_parameter)

                # changing the input_data to numpy array
                import numpy as np

                input_data_as_numpy_array = np.asarray(input_data)

                # reshape the array as we are predictin g for one instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                # standardize the input data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                #scaler.fit(df.drop('Trans_Programming_Score', axis=1).values)
                scaler.fit(df_choice.values) # fit the trained features from user dropdown

                std_data = scaler.transform(input_data_reshaped)
                print(std_data)

                # print prediction based on input data
                prediction = grid.predict(std_data)
                print(prediction)

                # Print Model Performance

                st.subheader('2. Model Performance')

                #st.markdown('**2.1. Standardised Training Dataset**')
                #df_input_names = pd.DataFrame(std_data, columns = make_choice)
                #st.table(df_input_names)
                #st.write(std_data)

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
                #st.write(confusion_matrix(y_test, grid_predictions))

                from sklearn.metrics import ConfusionMatrixDisplay

                cconf_matrix = confusion_matrix(y_test, grid_predictions)
                #dl = list(set(df_newresults[model_class]))
                #dl = sorted(dl)

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

                st.write("**Step 2.6: New Input Parameters for Prediction**")
                st.write("Update the sidebar with new parameters for prediction")
                
                df_inputs = pd.DataFrame(input_data_reshaped, columns = make_choice) 
                st.table(df_inputs)

                st.write("**Step 2.7: Single Entry Prediction Result**")
                st.write("Upon update of sidebar with new input parameters, click Predict")
                prediction_button = st.button("Predict")

                if prediction_button:

                    if (prediction[0] == 0):
                        st.info('Programming Module Prediction: **AT RISK**')
                    else:
                        st.info('Programming Module Prediction: **PASS**')
                        st.balloons()

                else:
                    ("Enter Results in Sidebar and Click Predict")
