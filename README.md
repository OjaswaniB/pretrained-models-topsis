# Text Classification Model Selection using TOPSIS

## Overview

This Python script demonstrates how to use the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method to rank pre-trained models for text classification. The script evaluates different models on a subset of the 20 newsgroups dataset and ranks them based on accuracy.

## Dependencies

Make sure you have the following dependencies installed:

- pandas
- scikit-learn

## Usage
- Clone the repository
git clone https://github.com/ojaswaniB/topsis-text-classification.git

- Run the Script
cd topsis-text-classification

## Customization
- Dataset: The code uses the 20 newsgroups dataset for demonstration purposes. You can replace it with your own dataset by modifying the fetch_20newsgroups section.
- Models: The script evaluates four pre-trained models: Multinomial Naive Bayes, Logistic Regression, Random Forest, and Support Vector Machine. You can add or remove models as needed by modifying the models list.
- Criteria: Currently, the script considers only accuracy as the evaluation criterion. If you have additional criteria, you can extend the code accordingly.

## Acknowledgements
- The code uses the 20 newsgroups dataset, which is available through scikit-learn
