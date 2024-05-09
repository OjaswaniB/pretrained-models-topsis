# Pre-trained-Model-Comparison-for-Text-Classification-using-Topsis

## Overview
This repository provides a simple and efficient tool for comparing pre-trained models using the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) method. TOPSIS is a decision-making technique that helps evaluate and rank alternative solutions based on their proximity to the ideal solution.

## Features
- 1. Easy Comparison: Quickly compare multiple pre-trained models.
- 2. Customizable Metrics: Evaluate models based on various metrics (e.g., accuracy, F1 score, precision, recall).
- 3. Models: Real-world pretrained models, such as bert-base-uncased, roberta_base, distilbert-base-uncased,xlnet-base-cased, t5-small, and albert-base-v2, are included in the comparison. These models are widely used in text classification tasks.

## Steps
Start --> Load Data --> Preprocess --> Select Metrics --> Apply Topsis --> Rank Models --> Visualize Results --> End

## Output
- 1. roberta-base: Robust in classification tasks, demonstrating high F1 score and accuracy. It is considered best pre-trained model for our dataset.

- 2. xlnet-base-cased: Excels in text classification tasks with 90% accuracy.It is the second best pre-trained model.

- 3. bert-base-uncased: A versatile model with an 89% accuracy, balanced F1 score, and potential for improved text similarity metrics. It is the third best pre-trained model.

- 4. albert-base-v2: Model with an 88% accuracy. It is a computationally efficient version of BERT.

- 5. distilbert-base-uncased: Model with an 87% accuracy and applicability in real-time scenarios.

- 6. t5-small: Model has high accuracy than some other models but it has the poorest topsis score.Thus,it is ranked last for our dataset.




