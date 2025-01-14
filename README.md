## Loan Prediction System
CDS6314 Data Mining Project (Group G27)

Application Link:
https://loan-prediction-system-g27.streamlit.app/

### Overview
The model predict the loan status, either approved or denied, based on details entered by user such as age, number of dependents, income, expenses, and more. The model trained and used in this application is Random Forest. The dataset is obtained from Kaggle named "Loan Approval Dataset". It contains more than 50000 rows of data and 27 columns.

Dataset link: https://www.kaggle.com/datasets/arbaaztamboli/loan-approval-dataset

The model trained is saved into a pkl file using pickle, and then splitted into multiple pkl files due to the large size file of the original file. The pkls file is uploaded in Github, inside "model_parts" folder. When the model is loaded, it will all combines and forms rf_model_reassembled.pkl file.

### Model development
There are total 3 models tested in this project, which are Random Forest, Decision Tree, and K-Nearest Neighbors (KNN). All models are fine tuned to find the best hyperparameters using GridSearch before comparing their performances. Those best parameters are stored in "best_params.json". After comparison, Random Forest was selected as the best-performing model, achieving an accuracy of 85%. This slightly outperformed the other models, making it the preferred choice for this classification task.
