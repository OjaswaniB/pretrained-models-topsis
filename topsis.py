import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups

# Load the 20 newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Create a DataFrame for easier handling
data = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

# Split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# Define the pre-trained models and their corresponding names
models = [
    ('Multinomial Naive Bayes', MultinomialNB()),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Support Vector Machine', SVC())
]

# Create a DataFrame to store the evaluation results
results = pd.DataFrame(columns=['Model', 'Accuracy'])

# Evaluate each model and store the results
for model_name, model in models:
    # Vectorize the text data
    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(train_data)
    test_features = vectorizer.transform(test_data)
    
    # Train the model
    model.fit(train_features, train_target)
    
    # Make predictions
    predictions = model.predict(test_features)
    
    # Evaluate accuracy
    accuracy = accuracy_score(test_target, predictions)
    
    # Append the results to the DataFrame
    results = results.append({'Model': model_name, 'Accuracy': accuracy}, ignore_index=True)

# Normalize the accuracy values
results['Normalized_Accuracy'] = results['Accuracy'] / results['Accuracy'].sum()

# Calculate the ideal and negative-ideal solutions
ideal_solution = results[['Normalized_Accuracy']].max()
negative_ideal_solution = results[['Normalized_Accuracy']].min()

# Calculate the Euclidean distances to ideal and negative-ideal solutions
results['Distance_to_Ideal'] = ((results['Normalized_Accuracy'] - ideal_solution) ** 2).sum(axis=1) ** 0.5
results['Distance_to_Negative_Ideal'] = ((results['Normalized_Accuracy'] - negative_ideal_solution) ** 2).sum(axis=1) ** 0.5

# Calculate the TOPSIS score
results['TOPSIS_Score'] = results['Distance_to_Negative_Ideal'] / (results['Distance_to_Ideal'] + results['Distance_to_Negative_Ideal'])

# Rank the models based on the TOPSIS score
results = results.sort_values(by='TOPSIS_Score', ascending=False)

# Print the ranked models
print("TOPSIS Ranking:")
print(results[['Model', 'TOPSIS_Score']])
