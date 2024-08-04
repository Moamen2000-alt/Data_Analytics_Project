# DATA ANALYTICS Project

# Airbnb Price prediction


## Table of Contents
1. [Data Preparation and Preprocessing Steps](#data-preparation-and-preprocessing-steps)
2. [Data Splitting](#data-splitting)
3. [NLP Processing](#nlp-processing)
4. [Encoding Categorical Data](#encoding-categorical-data)
5. [Relationship Between Strict House Rules and Listing Prices](#relationship-between-strict-house-rules-and-listing-prices)
6. [Relationship Between Reviews and Listing Prices](#relationship-between-reviews-and-listing-prices)
7. [Model Performance](#model-performance)
8. [Deep Learning Models](#deep-learning-models)
9. [Summary](#summary)

## Data Preparation and Preprocessing Steps

1. **Handling Missing Values**:
   - **host_identity_verified**: Replace `Null` values with `'unconfirmed'`, assuming null value means the host didn't verify themselves.
   - **instant_bookable**: Replace `Null` values with `'False'`, assuming null value means it hasn't Instant_bookable Option.
   - **cancellation_policy**: Remove rows with missing values in this column.
   - **lat and long**: Drop rows with null values in both `lat` and `long` columns.
   - **neighbourhood_group**: Drop rows with null values in this column.

2. **Imputation**:
   - **Construction year**: Fill null values in the construction year column with the median construction year.
   - **minimum_nights**: Fill null values with `1`, assuming minimum nights is `1` if not specified.
   - **calculated_host_listings_count**:
     - Calculate the median of `calculated_host_listings_count` for each host name.
     - Use the overall median as a fallback for missing values.
     - Apply a function to fill missing values with the host-specific median or the overall median if the host-specific median is not available.

3. **Handling Out-of-Range Values**:
   - **availability_365**:
     - Check for out-of-range values.
     - Correct out-of-range values by setting negative values to `0` and values above `365` to `365`.
     - assume null is 1, as it must be at least 1 night available 

4. **Dropping Columns**:
   - Drop columns: `host_name`, `license`, `country`, `country_code`, `id`, `host_id`.

5. **Encoding Categorical Variables**:
   - **room_type**: Encode room types into numerical values for model compatibility.
   - **cancellation_policy**: Encode cancellation policies into numerical values.

6. **Check other Features**:
   - **price**: Ensure all prices are in a consistent currency and format.
   - **service_fee**: Ensure all service fees are in a consistent currency and format.
   - **number_of_reviews**: Ensure all review counts are integers and properly formatted.
   - **reviews_per_month**: Ensure reviews per month are in a consistent format.
   - **review_rate_number**: Ensure review rate numbers are properly formatted.
   - **minimum_nights**: Ensure minimum nights are properly formatted.
   - **instant_bookable**: Ensure `instant_bookable` values are boolean.
   - **cancellation_policy**: Ensure cancellation policies are in a consistent format.
   - **room_type**: Ensure room types are properly encoded.
   - **Construction year**: Ensure construction years are properly formatted and imputed.
   - **calculated_host_listings_count**: Ensure `calculated_host_listings_count` are properly imputed.
   - **availability_365**: Ensure `availability_365` values are within the range `0-365`.

## Data Splitting
**Description**: The dataset is split into training and testing sets to ensure that the model can be evaluated on data it hasn't seen during training. This split is crucial for validating the model's performance and generalizability.

## NLP Processing
**Description**: Natural Language Processing (NLP) is applied to the `NAME` and `house_rules` columns to extract meaningful information from the text data. The following steps are performed:

1. **Text Preprocessing**:
   - **Combining Text Columns**: The `NAME` and `house_rules` columns are combined into a single text column to consolidate the information.
   - **Tokenization**: The combined text is tokenized, which involves breaking down the text into individual words or tokens. This step is essential for converting text into a format that can be further processed.
   - **Padding**: After tokenization, the sequences of tokens are padded to ensure they all have the same length. Padding is necessary because machine learning models require input data of uniform dimensions.

## Encoding Categorical Data
**Description**: Categorical data is encoded into numerical format using `OneHotEncoder`. This encoding process is necessary because machine learning algorithms typically require numerical input. The following steps are performed:

1. **Identifying Categorical Columns**: Columns containing categorical data, such as `host_identity_verified`, `neighbourhood_group`, `neighbourhood`, and `room_type`, are identified for encoding.
2. **Applying OneHotEncoder on X_train and x_text separate to avoid data leakage**
3. **Applying OneHotEncoder**: The identified categorical columns are transformed into a series of binary columns using `OneHotEncoder`. This process converts each category into a binary vector, making the data suitable for machine learning models.
4. **Integrating Encoded Data**: The encoded categorical data is then integrated back into the original dataset, ensuring that the dataset contains both the original numerical features and the newly created binary features from the categorical data.

## Relationship Between Strict House Rules and Listing Prices

**Objective**: To analyze the relationship between the sentiment of house rules and the listing prices. The sentiment of house rules is assessed using TextBlob to derive a sentiment polarity score, which is then used as a feature in a regression model to predict listing prices.

1. **Sentiment Analysis**:
   - **Goal**: To determine the sentiment polarity of the `house_rules` text.
   - **Method**:
     - Fill missing values in the `house_rules` column with an empty string to avoid errors during sentiment analysis.
     - Ensure all values in the `house_rules` column are strings.
     - Use TextBlob to analyze the sentiment of each entry in the `house_rules` column, which returns a polarity score ranging from -1 (negative sentiment) to 1 (positive sentiment).
     - Add the resulting sentiment scores to the dataset as a new column `house_rules_sentiment`.

2. **Data Splitting**:
   - **Goal**: To split the dataset into training and testing sets for model evaluation.
   - **Method**:
     - Use `train_test_split` to divide the data, with the `house_rules_sentiment` column as the feature and `price` as the target variable.
     - Specify a test size of 20% and set a random state for reproducibility.

3. **Model Training**:
   - **Goal**: To train a regression model to predict listing prices based on the sentiment of house rules.
   - **Method**:
     - Define a `GradientBoostingRegressor` model.
     - Set up a parameter grid for hyperparameter tuning, including different values for the number of estimators, learning rate, and maximum depth.
     - Use `GridSearchCV` to perform a grid search with cross-validation to find the best hyperparameters.
     - Fit the model to the training data and select the best estimator.

4. **Model Evaluation**:
   - **Goal**: To evaluate the performance of the trained model.
   - **Method**:
     - Use the best model to make predictions on the test set.
     - Calculate evaluation metrics including RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared score to assess model performance.
     - Plot the actual prices against the predicted prices to visualize the model's accuracy.

5. **Visualization and Correlation Analysis**:
   - **Goal**: To visualize the relationship between house rules sentiment and listing prices, and to calculate their correlation.
   - **Method**:
     - Create a scatter plot to visualize the relationship between `house_rules_sentiment` and `price`.
     - Calculate the correlation coefficient to quantify the strength and direction of the relationship.

**Results**:
- **Correlation**: The correlation between house rules sentiment and listing price is found to be -0.014, indicating a very weak negative relationship, So there is no direct relation 
   between house rules and listing price

## Relationship Between Reviews and Listing Prices

**Objective**: To analyze the relationship between the review ratings of listings and their prices. This analysis uses visualization techniques to explore and illustrate the relationship.

1. **Categorizing Review Ratings**:
   - **Goal**: To ensure that the review rating numbers are treated as categorical variables for appropriate analysis and visualization.
   - **Method**: Convert the `review_rate_number` column to a categorical data type to facilitate categorical analysis.

2. **Violin Plot**:
   - **Goal**: To visualize the distribution of listing prices across different review rating categories.
   - **Method**:
     - Create a violin plot with `review_rate_number` on the x-axis and `price` on the y-axis.
     - Use an inner box plot within the violin plot to show summary statistics.
     - Adjust the scale and palette for better visualization.
     - Title and label the axes appropriately.

3. **Box Plot**:
   - **Goal**: To compare the listing prices across different review rating categories using a box plot.
   - **Method**:
     - Create a box plot with `review_rate_number` on the x-axis and `price` on the y-axis.
     - Use a different dataset variant if necessary (processed_data_1 in this case) to include specific review rate categories.
     - Title and label the axes appropriately.

**Results**:
- **Visual Insights**: The violin plot and box plot provide insights into how listing prices vary across different review rating categories. These visualizations highlight some trends  in the data but didn't ensure that review ratings have strong correlate with higher listing prices.

## Modeling 

1. **Linear Regression**: Simple and fastest one, also less consumer for resources 
   - **RMSE**: 24.70
   - **MAE**: 2.97
   - **R-squared**: 0.9945

2. **Gradient Boosting**: provides very good accuracy but it consumes the resources very much 
   - **Best Parameters**: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50}`
   - **RMSE**: 24.57
   - **MAE**: 3.93
   - **R-squared**: 0.9945

3. **XGBoost**: use XGBoost as it has better performance than normal Gradient Boosting and also built the model to be simple as I already graped the best parameters from the Gradient 
     Boosting  
   - **Same Parameters**: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50}`
   - **RMSE**: 24.50
   - **MAE**: 3.90
   - **R-squared**: 0.9946

4. **Random Forest (Full Grid Search)**:
   - Note: The grid search did not converge and was resource-intensive.

5. **Random Forest (Simplified Model)**: try to build simple Random Forest based on the good parameters from number 4 
   - **Parameters**: `{'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1}`
   - **RMSE**: 25.25
   - **MAE**: 3.05
   - **R-squared**: 0.9942

**Analysis**:
- **Performance**: All models perform similarly, with very high R-squared values, indicating a good fit. The RMSE and MAE values are quite close across the models except in number 4.
- **Efficiency**: Linear Regression is the simplest and most efficient, but it may not capture complex patterns.
- **Complex Models**: Gradient Boosting, XGBoost, and Random Forest offer better performance with minimal improvements over Linear Regression but They are more computationally intensive.

**Recommendations**:
- **For Simplicity and Efficiency**: Linear Regression is a good choice.
- 
## Deep Learning Models

1. **Deep Neural Network (DNN)**:
   - **Model Architecture**:
     - Input: 128 units (ReLU) → Dropout → 64 units (ReLU) → Dropout → 32 units (ReLU) → Output
   - **Results**:
     - **RMSE**: 101.73
     - **MAE**: 78.85
     - **R-squared**: 0.9062

2. **Recurrent Neural Network (RNN)**:
   - **Model Architecture**:
     - Embedding → SimpleRNN (64 units) → Dropout → SimpleRNN (32 units) → Output
   - **Results**:
     - **RMSE**: 336.25
     - **MAE**: 289.68
     - **R-squared**: -0.0244

**Analysis**:
- **DNN Performance**:
  - The DNN model performs relatively well with lower RMSE and MAE compared to the RNN. The R-squared value, while not as high as the traditional models, indicates that the DNN can 
    capture some patterns in the data.
  - The high RMSE and MAE indicate that the DNN model might still have room for improvement or that the target variable might have a wide range of values.

- **RNN Performance**:
  - The RNN model significantly underperforms compared to the other models. The high RMSE, MAE, and negative R-squared value suggest that it isn't capturing the underlying patterns well.
  - RNNs are typically used for sequential or time-series data. If your data isn’t sequential, an RNN might not be suitable.

**Recommendations**:
- **DNN Tuning**:
  - Consider tuning hyperparameters such as the number of layers, units, dropout rates, and learning rates to potentially improve the model’s performance.
  - Experiment with different architectures and regularization techniques to prevent overfitting.

## Summary
- **Machine Learning Models**:
  - Perform well with high R-squared values and low RMSE/MAE.
  - XGBoost and Gradient Boosting are slightly better than other ML models.
  - Random Forest's full grid search was too resource-intensive, but the simplified model performed well.

- **Deep Learning Models**:
  - The DNN has higher RMSE and MAE compared to ML models and a lower R-squared value.
  - The RNN performs poorly, indicating it may not be appropriate for the given data type.

The ML models generally outperform the DL models for your dataset, suggesting that simpler models may be more effective.

Authors and load of work:
Salma: Data Exploration, Data Cleansing and Q2
Yasmin: Data analysis & Check correlation between data, Data Cleansing and Q2
Moamen Eman: NLP Part, Encoding part, Q1 & Q3
