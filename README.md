# Phishing Detection using AI

## Introduction
Phishing is a common cybersecurity threat where malicious actors deceive users into providing sensitive information through fraudulent websites. To mitigate this risk, machine learning models can be developed to detect phishing websites based on various features extracted from URLs, such as the presence of an IP address, domain age, or URL length.

This project aims to create a machine learning pipeline for detecting phishing websites using various algorithms. The dataset used contains website attributes and a binary label that classifies the website as either *phishing* or *safe*.

The project follows a comprehensive process, including exploratory data analysis, data preprocessing, model training, evaluation, and feature importance assessment. Multiple machine learning classifiers are compared to find the best-performing model.

## Data Overview
The dataset `PhishingDetectionData.csv` contains 17 columns. The columns represent different features extracted from website URLs, such as:
- **Have_IP**: Whether the URL contains an IP address (often indicative of phishing).
- **Have_At**: Whether the URL contains the '@' symbol, a common feature in phishing URLs.
- **URL_Length**: The length of the URL, which is typically longer for phishing websites.
- **DNS_Record**: The DNS record of the domain; absence may indicate phishing.
- **Web_Traffic**: Indicates the website's popularity. Lower traffic often correlates with phishing.
- **Domain_Age**: Newly registered domains may be linked to phishing attacks.
- **Label**: The target column; '1' represents a phishing website, '0' represents a safe website.

These features are used to train machine learning models that can predict whether a given website is phishing or safe.

## Step-by-Step Process

### 1. **Loading the Dataset**
The dataset is loaded into a Pandas DataFrame. The **head**, **shape**, and **info** functions provide insights into the structure of the data, revealing the number of rows, columns, and data types.

### 2. **Exploratory Data Analysis (EDA)**
EDA is crucial in cybersecurity because it helps to understand the underlying patterns in phishing URLs. The analysis includes:
   - **Shape Inspection**: Determines the dimensions of the dataset.
   - **Column Analysis**: Lists all the column names, helping to identify feature types (e.g., categorical vs. numerical).
   - **Descriptive Statistics**: Provides a summary of each feature, showing the distribution of values, including the mean, standard deviation, and range. This step helps identify if phishing websites are generally longer in URL length, have more redirections, or use specific techniques like tiny URLs.
   - **Heatmap**: A correlation matrix of the features is plotted. Features that are highly correlated (e.g., `Domain_Age` and `DNS_Record`) might be indicative of strong relationships in predicting phishing.

### 3. **Visualizing Data Distributions**
   - **Histograms**: For each feature, a histogram visualizes its distribution for phishing and safe websites. This helps analyze which features contribute to phishing behavior. For example, features like `Have_IP` may show significant differences between phishing and safe websites.

### 4. **Preprocessing**
   - **Dropping Unnecessary Features**: The `Domain` column is dropped since it contains unique domain names that are irrelevant for modeling. 
   - **Shuffling the Data**: The dataset is shuffled to ensure randomness in the split between training and test data, preventing the model from overfitting on ordered data.
   - **Handling Missing Data**: Missing values are checked to ensure data completeness. In cybersecurity, handling missing data is critical to avoid skewing the model's performance.

### 5. **Train/Test Split**
The data is split into training and testing sets. 
   - **Training Data (80%)**: Used to fit the machine learning models. 
   - **Testing Data (20%)**: Used for evaluating the model's performance.
   
The split ensures that the model is validated on unseen data, simulating a real-world environment where new phishing websites are encountered.

### 6. **Model Selection**
Several machine learning models are used to predict phishing websites. Each model has its unique way of processing and learning from the data. Here’s a breakdown of the models implemented:

#### 6.1 **Decision Tree Classifier**
   - **Goal**: Split the data based on feature importance and create a flowchart-like model that makes predictions based on feature values.
   - **Interpretability**: Decision Trees provide insights into which features (e.g., `Have_IP`, `URL_Length`) are most important in phishing detection. This can be crucial in identifying specific phishing techniques and patterns.
   - **Overfitting Concern**: Since decision trees can overfit the data, `max_depth` is set to 5 to prevent overly complex trees that perform well on training data but poorly on new phishing websites.

#### 6.2 **Random Forest Classifier**
   - **Goal**: A collection of Decision Trees (ensemble method) improves accuracy by reducing variance. Each tree votes, and the majority vote classifies the URL as phishing or safe.
   - **Feature Importance**: Random Forests provide insights into which features contribute the most to phishing detection, which can help cybersecurity professionals focus on specific characteristics of phishing URLs.
   
#### 6.3 **Multilayer Perceptron (MLP) Classifier**
   - **Goal**: A neural network with multiple hidden layers to capture complex patterns in the dataset. 
   - **Relevance in Cybersecurity**: MLPs can detect subtle phishing patterns that simpler models might miss, such as relationships between URL structure and web traffic. The model uses 3 hidden layers, each with 100 neurons.
   
#### 6.4 **Support Vector Machine (SVM)**
   - **Goal**: Use hyperplanes to classify phishing vs. safe websites. SVM is particularly good for high-dimensional data and can identify phishing websites based on subtle feature combinations.
   - **Kernel Trick**: Linear kernel is used to separate phishing and safe websites in feature space.

#### 6.5 **K-Nearest Neighbors (KNN)**
   - **Goal**: Classify a website by finding the K closest websites in the dataset. If most neighbors are phishing, the website is classified as phishing.
   - **Computational Efficiency**: In cybersecurity, KNN might not be the most scalable model due to its high computational cost during inference, especially when dealing with real-time phishing detection.

#### 6.6 **Naive Bayes**
   - **Goal**: Use probability theory and conditional independence assumptions to classify websites. 
   - **Relevance**: Naive Bayes is fast and often performs well in cybersecurity tasks like spam detection, where features are largely independent.

#### 6.7 **Logistic Regression**
   - **Goal**: Predict the probability of a website being phishing or safe using a linear combination of features.
   - **Use in Cybersecurity**: Logistic Regression is simple yet effective in binary classification tasks like phishing detection, offering interpretable outputs (probabilities).

#### 6.8 **Neural Networks (Using TensorFlow/Keras)**
   - **Goal**: A deeper neural network is built using TensorFlow to classify websites. The model uses dense layers, with sigmoid activation functions to ensure binary classification outputs.
   - **Cybersecurity Application**: Neural networks can capture highly complex relationships in data and are useful in detecting advanced phishing techniques, especially when combined with large-scale datasets.

### 7. **Model Evaluation**
   - **Accuracy Calculation**: Each model is evaluated based on accuracy, which measures the percentage of correct predictions on both the training and test sets.
   - **Model Comparison**: The accuracies are stored for comparison, allowing for the identification of the most effective phishing detection model.
   
### 8. **Feature Importance Analysis**
   - For both the **Decision Tree** and **Random Forest** models, feature importance is calculated and plotted. Features like `Have_IP` or `Domain_Age` might have a significant influence on the model’s decision-making process.
   - In cybersecurity, this analysis can help prioritize which features of URLs should be monitored to improve phishing detection systems.

### 9. **Results Summary**
   - All models’ training and testing accuracies are compiled into a DataFrame for comparison.
   - The models are sorted based on test accuracy to identify the most effective at detecting phishing websites in new, unseen data.

### 10. **Best Model Selection**
After comparing the models, the one with the highest test accuracy is considered the best-performing model. In cybersecurity, selecting the most accurate model is critical, as false negatives (phishing websites classified as safe) can lead to significant security breaches.

## Conclusion
This phishing detection project highlights how different machine learning algorithms can be employed to detect malicious websites. By evaluating multiple models, we can identify the most effective one for real-world applications. Feature importance analysis further provides insights into the key characteristics of phishing websites, offering valuable information for cybersecurity professionals to design better detection systems.

## Future Work
- **Feature Engineering**: More advanced features, such as time-based features (domain registration time) or external threat intelligence data, could be incorporated to enhance detection accuracy.
- **Model Optimization**: Hyperparameter tuning of models like SVM and Random Forest could further improve accuracy and model robustness.
- **Real-Time Phishing Detection**: The model could be integrated into a real-time system, flagging suspicious URLs as they are accessed by users.
  
By automating the detection of phishing websites, this project contributes to a proactive cybersecurity strategy, helping prevent phishing attacks before they compromise users' data.