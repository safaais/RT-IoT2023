# RT-IoT2023

## Dataset description.

The RT-IoT2022, is a proprietary dataset derived from a real-time IoT infrastructure, is introduced as a comprehensive resource integrating a diverse range of IoT devices and sophisticated network attack methodologies. 

## Problem definition.

RT IoT 2023 dataset has challenges of missing and noisy data, and inconsistencies that are not friendly to most machine learning algorithms. Further, because the data and feature space have a high dimensionality, the set becomes overfit and model training become challenging, it becomes quite important to employ a strong pre-processing technique that will efficiently improve the data. This is performed to handle missing values, scale the data and possibly; decrease dimensionality in order to prepare the data for the next stage of analysis.


## Implementation.

Here’re the necessary steps to implement classification model to predict attack type in RT-IOT 2023 dataset:
- Import Libraries.
- Loud Dataset and Handle Missing Values. 
- Define Features and Target Variable.
- Data Cleaning – remove outliers.
- Split Data into Training and Testing Sets.
- Scale Numerical Features.
- Train and Evaluate Classifier.


Add Class Weighting to avoid imbalanced classes issue

![image alt](https://github.com/safaais/RT-IoT2023/blob/3f2613bdaaf454709fafa6508683814b62c82b86/ClassWeighting.png)



## Results.
![image alt](https://github.com/safaais/RT-IoT2023/blob/main/ClassifierComparison.png?raw=true)


Results showed poor performance for Gradient Boosting classifier.
Thererfor, remove Gradient Boosting classifier and add Voting Classifier which uses the outputs of several base classifiers in an ensemble form to arrive a decision on the true positive and negative results.


![image alt](https://github.com/safaais/RT-IoT2023/blob/123c3fc9ae5964885854393ff399e9dd80597177/VotingMatric.png)
