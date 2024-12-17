
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                           classification_report, confusion_matrix)
from time import time
from scipy.stats import zscore
from collections import Counter

# load the dataset
df = pd.read_csv('RT-IOT2023.csv')

# check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# handle missing values
# option 1: Drop rows with missing target values
df = df.dropna(subset=['label'])

# define features (X) and target (y)
X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight',
]
y_column = 'label'

# create the 8-class mapping
dict_8classes = {
    'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS', 'DoS-UDP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai', 'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon',
    'DNS_Spoofing': 'Spoofing', 'MITM-ArpSpoofing': 'Spoofing',
    'BenignTraffic': 'Benign', 'BrowserHijacking': 'Web',
    'Backdoor_Malware': 'Web', 'XSS': 'Web', 'Uploading_Attack': 'Web',
    'SqlInjection': 'Web', 'CommandInjection': 'Web',
    'DictionaryBruteForce': 'BruteForce'
}

# apply the 8-class mapping to the target variable
df['label_8classes'] = df[y_column].map(dict_8classes)

# define adaptive thresholds based on attack frequency
class_counts = df['label_8classes'].value_counts()
total_samples = len(df)
thresholds = {}

for attack_type in class_counts.index:
    frequency = class_counts[attack_type] / total_samples
    # Use more conservative thresholds for rare attacks
    if frequency < 0.05:  # rare attacks
        thresholds[attack_type] = 5.0  # more permissive
    elif frequency < 0.15:  # uncommon attacks
        thresholds[attack_type] = 4.0
    else:  # common attacks
        thresholds[attack_type] = 3.0

# implement smart outlier detection
def detect_outliers(group_df, attack_type, columns):
    threshold = thresholds[attack_type]
    z_scores = zscore(group_df[columns])
    
    # for rare attacks, require multiple features to be outliers
    if threshold > 3.0:
        return (np.abs(z_scores) > threshold).sum(axis=1) >= 3
    else:
        return (np.abs(z_scores) > threshold).any(axis=1)

# track statistics before cleaning
print("Class distribution before cleaning:")
print(df['label_8classes'].value_counts())

# apply outlier detection per attack type
outliers_mask = pd.Series(False, index=df.index)
for attack_type in df['label_8classes'].unique():
    group_mask = df['label_8classes'] == attack_type
    group_df = df[group_mask]
    
    if len(group_df) > 1:  # need at least 2 samples for z-score
        group_outliers = detect_outliers(group_df[X_columns], attack_type, X_columns)
        outliers_mask[group_df[group_outliers].index] = True

# remove outliers and prepare data
X_cleaned = df[X_columns][~outliers_mask]
y_cleaned = df['label_8classes'][~outliers_mask]

# print cleaning statistics
print("\nOutliers removed per class:")
removed_counts = df[outliers_mask]['label_8classes'].value_counts()
for attack_type in df['label_8classes'].unique():
    original = class_counts[attack_type]
    removed = removed_counts.get(attack_type, 0)
    remaining = original - removed
    print(f"{attack_type}: {removed}/{original} removed ({(removed/original)*100:.1f}%)")

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y_cleaned, test_size=0.2, random_state=42, stratify=y_cleaned
)

# scale numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define classifiers with class weight consideration
classifiers = [
    AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
    RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
    SVC(probability=True, class_weight='balanced'),
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
]

# loop through classifiers
for model in classifiers:
    print(f"\nTraining {model.__class__.__name__}:")
    
    start = time()
    model.fit(X_train_scaled, y_train)
    train_time = time() - start

    start = time()
    y_pred = model.predict(X_test_scaled)
    predict_time = time() - start

    print(f"\tTraining time: {train_time:.3f}s")
    print(f"\tPrediction time: {predict_time:.3f}s")

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\tModel Accuracy Score: {accuracy:.3f}")
    print(f"\tModel Recall Score: {recall:.3f}")
    print(f"\tModel Precision Score: {precision:.3f}")
    print(f"\tModel F1 Score: {f1:.3f}")

    # classification Report
    print("\tClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))