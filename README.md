# 1. Business Understanding

**Objective**

The main business goal is to analyze customer sentiments expressed as tweets about various brands or products. By building a sentiment classification model, the company aims to:

Measure public opinion of its products or services.

Identify negative feedback early to improve on customer experience.

Track brand reputation and customer satisfaction over time.

Support marketing and product strategy decisions based on data-driven insights.

In this project, we are aiming to build a machine learning model that can determine the sentiment of a tweet based on the content, whether it is positive, negative, or neutral.This is a Natural Language Processing (NLP) problem, useful for applications such as social media monitoring, brand-analysis, and customer-feedback tracking.

# Key Business Questions

What proportion of customer mentions express positive vs. negative emotions?

How can the business improve customer satisfaction based on sentiment trends?

# Success Criteria

Achieve a classification accuracy of at least 67% (baseline observed: 0.6707).

Generate a reliable sentiment distribution (positive, negative, neutral) for business reporting.

Enable automated monitoring of customer feedback at a certain scale.

# 2.Data Understanding

**Data Source**

The dataset has been sourced from data.world. The data contains 9,070 rows and 2 columns

**Data Overview**

From the classification report, the dataset includes at least four sentiment categories:

I can't tell **—** unclear sentiment

Negative emotion **—** dissatisfaction, complaints

No emotion toward brand or product **—** neutral comments

Positive emotion **—** satisfaction

# Observations
We discovered that the class distribution seems imbalanced, with the “No emotion toward brand or product” class having the highest support (1674 samples).

We saw that the accuracy_score (67.07%) indicates that the model performs moderately well but may struggle with minority classes ("I can't tell").

We saw that the data included noise or ambiguous labels due to the subjective nature of sentiment.

# 3. Data Preparation

**Data Cleaning**

Handle Encoding Issues: Fix utf-8 by using appropriate encoding (latin1).

Remove Noise: Strip URLs, user mentions (@username), hashtags, punctuation, and emojis.

Normalize Text: Convert to lowercase and remove extra whitespace.

Handle Missing Values: Drop rows with missing text or sentiment labels.(1 row of tweet_text was dropped and the whole column of emotion_in_a_tweet_is_directed_at was dropped).

**Text Preprocessing**

Tokenization: Split text into words using nltk.word_tokenize().

Stopword Removal: Exclude common non-informative words.

Stemming: Reduce words to their base form.

Vectorization: Transform text into numerical form using TF-IDF.

**Data Splitting**

Train-Test Split: Divide the dataset (We used 70% train, 30% test).

Stratification: Ensure proportional representation of sentiment classes in both sets(stratify = y)

**Feature Engineering**

**Text Vectorization**: Converted tweet text into numeric representations that machine learning models can use(TF-IDF Vectorizer).

**Tweet Length**: Added a numeric feature representing the number of words or characters in each tweet.

# **Outcome**
We prepared the dataset to be suitable for machine learning models such as:

Logistic Regression

Naive Bayes

SMOTE

Support Vector Machine (SVM)


