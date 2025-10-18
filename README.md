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

# **Model Performance Summary**

After experimenting with several models and techniques (balancing, SMOTE, Naive Bayes, SVM, and tuned TF-IDF), we can now compare their performance in one clear table.

We'll evaluate both:

Accuracy — general correctness

Macro F1-score — fairness across all sentiment classes (important for imbalanced data

Linear SVM achieved the highest overall accuracy (~66%) and balanced sentiment detection.

Naive Bayes performed almost as well, making it a great lightweight alternative.

Balanced/SMOTE Logistic Regression improved recall for minority classes but at the cost of accuracy.

In text tasks, SVM and Naive Bayes often outperform Logistic Regression on sparse data like TF-IDF.

# 6.**Building a Machine Learning Pipeline**

To make the model training and prediction process more efficient, we combined our preprocessing, vectorization, and classification steps into a single Pipeline. A pipeline ensures that the same text cleaning and feature extraction steps are applied during training and when making predictions on new data.

We will build a pipeline with the following components:

**TFIDF Vectorizer**: Converts text into numerical features.

**Linear SVM Classifier**: Learns to predict the sentiment of tweets.

**Custom Text Cleaning Function**: Cleans and normalizes tweets (removes URLs, mentions, punctuation, etc.).

# 7.** DEPLOYMENT**

Model Deployment with Streamlit

After training and evaluating our model, the next step is deployment there by making it accessible to users through a simple web app.

We will use Streamlit, which is an interactive framework that allows us to create web interfaces for machine learning models with minimal code.

Our Streamlit app will:

Load the saved SVM pipeline (.pkl file)

Accept a user’s tweet or text input

show a bar code of the tweets

Preprocess and classify the text

Display the predicted sentiment

**create a streamlit app scrptt**

We will now create a new Python file named app.py that will handle:

Loading the trained model

Getting user input

Making predictions

Displaying the sentiment result

**Create a retrain_model**

We will now creat a new python file named **retrain_model.py** that will handle:

Loading the trained model

Loading the pipeline

Allow changes to be made of the pipeline

**Create BERT sentiment model Loader**

We used the Cardiff NLP RoBERTa since our dataset is tweet-based. The model was intergrated in the SVM-model and run side by side in the streamlit app without overwriting the load_model(). The models run independently. Why we added BERT model:

BERT understands meaning and not just words as in the SVM-model which counts words and has no concept of context.

BERT is trained on massive text datasets and fine-tuned for sentiment on millions of tweets(can understand emotion).

It is pre-trained on huge data.

BERT uses something called self-attention, meaning it learns relationships between words in a sentence.

**Summary**

The SVM-model performs well but has some limitations when it came to give out sentiments and therefore we opted for BERT to run our sentiment predictions.


