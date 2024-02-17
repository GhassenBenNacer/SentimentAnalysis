# Twitter Sentiment Analysis

## Introduction
This project focuses on sentiment analysis using the sentiment140 dataset, which contains 1.6 million tweets extracted using the Twitter API. The goal is to classify the sentiment of tweets as either negative, neutral, or positive.

## Dataset
The sentiment140 dataset can be found [here](https://www.kaggle.com/kazanova/sentiment140). It contains the following fields:
- **target**: The polarity of the tweet (0 = negative, 2 = neutral, 4 = positive).
- **ids**: The ID of the tweet.
- **date**: The date of the tweet.
- **flag**: The query (if applicable).
- **user**: The user who tweeted.
- **text**: The text of the tweet.

## Approach
- **Data Preprocessing**:
  - **Text Cleaning**: Removed punctuation and links using regular expressions.
  - **Text Normalization**: Converted text to lowercase.
  - **Stemming**: Applied stemming to reduce words to their root forms.
- **Model Selection**: Trained two classification models: Naive Bayes and Logistic Regression.
- **Model Evaluation**: Evaluated model performance using accuracy as the metric.
- **Results**: Achieved 76% accuracy using Naive Bayes and 77% accuracy using logistic regression.

## Files
- **sentiment_analysis.ipynb**: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- **sentiment140.csv**: The sentiment140 dataset used for training and testing.

## Dependencies
- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- Nltk 

## Usage
1. Clone the repository.
2. Open `sentiment_analysis.ipynb` in Jupyter Notebook.
3. Run the notebook cells to preprocess the data, train the models, and evaluate performance.

## References
- Original Dataset: [Sentiment140 Dataset](https://www.kaggle.com/kazanova/sentiment140)
- Paper: [Twitter Sentiment Classification using Distant Supervision](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

## Acknowledgements
- Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.
