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

I can't tell — unclear sentiment

Negative emotion — dissatisfaction, complaints

No emotion toward brand or product — neutral comments

Positive emotion — satisfaction

# Observations
We discovered that the class distribution seems imbalanced, with the “No emotion toward brand or product” class having the highest support (1674 samples).

We saw that the accuracy_score (67.07%) indicates that the model performs moderately well but may struggle with minority classes ("I can't tell").

We saw that the data included noise or ambiguous labels due to the subjective nature of sentiment.


