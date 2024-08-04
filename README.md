# Data_Analytics_Project
Airbnb Data Analysis and Machine Learning Project
Overview
This project involves the analysis of an Airbnb dataset, including data preprocessing, data cleaning, and implementing various machine learning models for predictive analysis.

Table of Contents
Data Preparation and Cleansing
Natural Language Processing (NLP)
Data Encoding
Data Analysis
Machine Learning and Deep Learning Models
Data Preparation and Cleansing
Handling Missing Values
Calculated Host Listings Count:

Filled missing values using the median value of 'calculated host listings count' per host.
Used the overall median as a fallback.
Applied a function to fill missing values accordingly.
Availability_365 Column:

Corrected out-of-range values in the 'availability 365' column.
Replaced values less than 0 with 0 and values greater than 365 with 365.
Replaced null values with 0.
Host Identity Verified Column:

Replaced null values with 'unconfirmed', assuming null indicates the host didn't verify their identity.
Host Since Column:

Converted to datetime format.
Filled missing values with the median date.
Host Response Rate Column:

Converted percentage strings to numerical values.
Filled missing values with the median response rate.
Host Acceptance Rate Column:

Converted percentage strings to numerical values.
Filled missing values with the median acceptance rate.
First Review Column:

Converted to datetime format.
Filled missing values with the median date of the first review.
Last Review Column:

Converted to datetime format.
Filled missing values with the median date of the last review.
Review Scores Rating Column:

Filled missing values with the median review score.
Neighbourhood Group Column:

Filled missing values with 'Unknown'.
Neighbourhood Column:

Filled missing values with 'Unknown'.
Latitude and Longitude Columns:

Checked for and handled any outliers.
Price Column:

Converted to numerical values by removing currency symbols.
Removed outliers based on statistical methods.
Filled missing values with the median price.
Minimum Nights Column:

Checked for out-of-range values and corrected them.
Replaced null values with the median minimum nights.
Number of Reviews Column:

Filled missing values with 0.
Reviews per Month Column:

Filled missing values with 0.
Calculated Host Listings Count Column:

Filled missing values using the median value per host.
Availability 365 Column:

Corrected out-of-range values.
Replaced null values with 0.
Outlier Detection and Correction
Price Column:

Removed outliers based on statistical methods.
Used domain knowledge to set realistic price ranges.
Review Scores Rating:

Filled null values with the median score.
Natural Language Processing
Reviews Text Processing:
Tokenization and stemming of text data.
Removed stop words.
Vectorized the text using TF-IDF (Term Frequency-Inverse Document Frequency).
Data Encoding
Categorical Data Encoding:
Applied one-hot encoding to categorical features such as 'neighbourhood', 'room type', and 'cancellation policy'.
Used label encoding for ordinal features.
Data Analysis
Exploratory Data Analysis (EDA):
Visualized the distribution of various features.
Identified correlations between different features.
Used histograms, scatter plots, and heatmaps for better understanding of data.
Machine Learning and Deep Learning Models
Deep Neural Network (DNN):

Defined, compiled, and trained a DNN model.
Evaluated the model using RMSE, MAE, and R-squared metrics.
Achieved significant results with fine-tuning and hyperparameter optimization.
Recurrent Neural Network (RNN):

Implemented an RNN model with embedding and SimpleRNN layers.
Used the model for time series data prediction.
Evaluated using RMSE, MAE, and R-squared metrics.
Comparison of Models:

Compared the performance of DNN and RNN models.
Summarized the findings and highlighted the strengths of each model.
Conclusion
The project provides a comprehensive analysis of Airbnb data, highlighting the importance of data preprocessing and the application of advanced machine learning models for predictive analysis. The models developed can be further enhanced with more data and feature engineering.

Authors
Moamen Emam
