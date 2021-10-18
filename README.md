# Spam_Email_detection_using_NLP
The project aims to classify incoming email messages as spam or ham (not spam) using natural language processing and machine learning.

About the dataset:
The dataset consists of 2893 records and 3 columns.
The columns are subject(email subject), message(body of the email) and label(target class 0 and 1). 0 indicates ham(or no spam) and 1 indicates spam email. 

Our objective is to correctly predict the label 1 (instances of spam email).

The file has been divided into 3 sections namely:
1. Exploratory Data Analysis
2. Text Pre-processing
3. Model Building by using sklearn's pipeline

Three machine learning models have been used for the classification task namely:
1. Multinomial Naive bayes
2. Decision Tree Classifier
3. Random Forest Classifier

Out of all 3, the Random forest classifier and Multinomial Naive Bayes models were found to generate the best scores on both train and test data.
