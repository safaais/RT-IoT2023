# RT-IoT2023

## Dataset description.

The RT-IoT2022, is a proprietary dataset derived from a real-time IoT infrastructure, is introduced as a comprehensive resource integrating a diverse range of IoT devices and sophisticated network attack methodologies. 

## Problem definition.

RT IoT 2023 dataset has challenges of missing and noisy data, and inconsistencies that are not friendly to most machine learning algorithms. Further, because the data and feature space have a high dimensionality, the set becomes overfit and model training become challenging, it becomes quite important to employ a strong pre-processing technique that will efficiently improve the data. This is performed to handle missing values, scale the data and possibly; decrease dimensionality in order to prepare the data for the next stage of analysis.


## Implementation.

Here’re the necessary steps to implement classification model to predict attack type in RT-IOT 2023 dataset:
### Import Libraries:
Import necessary libraries for data manipulation, visualization and commonly used classifiers for IoT datasets, include:


### Loud Dataset and Handle Missing Values: 
is an essential step in building a classification model including preparing the RT IoT 2023 because it retains the authenticity of data, improves predictive models, minimizes biases, ranks features, and makes the measure used to assess the model more accurate.

### Define Features and Target Variable:
The system uses 48 features for classification, including:
- Network flow metrics (duration, rate, header length)
- Protocol flags (FIN, SYN, RST, PSH, ACK, ECE, CWR)
- Protocol types (HTTP, HTTPS, DNS, etc.)
- Statistical measures (mean, standard deviation, variance)
- Network behavior indicators (IAT, magnitude, radius)

The original dataset contained detailed attack labels which were consolidated into 8 main categories:
- DDoS (Distributed Denial of Service)
- DoS (Denial of Service)
- Mirai (IoT malware variants)
- Recon (Reconnaissance attacks)
- Spoofing (DNS and ARP spoofing)
- Web (Various web-based attacks)
- BruteForce (Dictionary-based attacks)
- Benign (Normal network traffic)

### Data Cleaning – remove outliers:
Outlier elimination is useful because it increases model performance, facilitates better statistical analysis, reduces fragility, refines promising conclusions, improves the quality of data submitted for analysis, and expedites convergence.
- Using z-score
 
The cleaned DataFrame has 0 rows and 48 columns. Z score is too aggressive and removes all rows from DataFrame, that means there are no samples left in the dataset to train or test model.

To solve this:
- Display class distribution to provide an overview of class distribution and the impact of outlier removal.
- Use class adaptive Z-score thresholds based on frequency to avoid removing true positive outliers and applying less permissive thresholds for rare attack classes and more permissive thresholds for common attacks.


### Split Data into Training and Testing Sets:
Split the cleaned dataset into training and testing subsets, ensuring class distribution is maintained.

### Scale Numerical Features:
The purpose of this step is to improve model performance by standardize features by removing the mean and scaling to unit variance.

### Train and Evaluate Classifier:
Evaluate the performance for each classifier using metrics like: accuracy, recall, precision, and F1 score.

## Results
