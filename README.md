## Sentiment_analysis
Sentiment Analysis on IMDb Movie Reviews using TF-IDF and Logistic Regression


# Step 1: Data Preparation

Imported necessary libraries such as Pandas for data manipulation.
Loaded the IMDb Movie Reviews dataset into a Pandas DataFrame.
Checked the dataset size, inspected the first few rows, and identified any missing values.
Explored the distribution of sentiments (positive/negative) in the dataset.
Visualized the distribution of review lengths.

# Step 2: Data Exploration
Checked the first few rows of the dataset to understand its structure.
Determined the size of the dataset (number of rows and columns).
Checked for any missing values in the dataset.
Visualized the distribution of sentiments to assess class balance.
Explored the distribution of review lengths to understand the text data.

# Step 3: Feature Extraction with TF-IDF
Imported the TfidfVectorizer from scikit-learn for feature extraction.
Initialized the TF-IDF vectorizer with a maximum number of features (adjustable).
Fit and transformed the text data to obtain TF-IDF vectors.
The resulting TF-IDF matrix was used as input features for the model.

# Step 4: Model Building and Training (Logistic Regression)
Imported necessary libraries for model building and evaluation.
Split the dataset into training and testing sets.
Initialized and trained a Logistic Regression model on the TF-IDF vectors.
Made predictions on the test set using the trained model.
Evaluated the model's performance using accuracy and a classification report.
Generated a confusion matrix for detailed evaluation.
