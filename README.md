# **ML Model for SpaceX Falcon 9 First Stage Landing Prediction using Python**
## **Overview**
This project is part of the final requirement for the IBM Data Science and Machine Learning Capstone Project. The objective of this project is to develop a machine learning model capable of predicting the success of the SpaceX Falcon 9 first stage landing. Predicting successful landings is critical as it has significant implications for reducing the cost of space travel.

## **Project Summary**
The project involves applying supervised learning algorithms to create a predictive model. After evaluating various machine learning models using a grid search method, the Decision Tree model emerged as the best fit for the dataset. The model achieved the following metrics:

* Accuracy: 0.875
* Average F1-Score: 0.8148
* Jaccard Score: 0.8000
## **Process Breakdown**
1. Data Wrangling
* Data Retrieval and Preparation:
* Collected raw data from SpaceX's historical launch records.
* Cleaned and preprocessed the data to ensure accuracy and consistency.
2. Exploratory Data Analysis (EDA)
* Correlation Analysis: Identified key features influencing the landing outcome through correlation matrices.
* Distribution Analysis: Analyzed the distribution of variables to understand their impact on the landing prediction.
3. Model Analysis
* Data Splitting: Split the dataset into training and testing subsets to validate model performance.
* Model Development: Developed and trained various machine learning models, including Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Decision Tree.
* Grid Search: Performed a grid search to optimize model hyperparameters and identify the best-performing model.
4. Model Evaluation
* Confusion Matrix: Constructed a confusion matrix to assess the model's accuracy and error rates.
* Distribution Analysis: Evaluated the model's prediction distribution to ensure balanced predictions.
* Jaccard Analysis: Used the Jaccard index to measure the similarity between predicted and actual outcomes.

## **Tools and Technologies**
* Programming Language: Python
* Interactive Programming Tool: Jupyter Notebook
* Data Manipulation and Analysis: Pandas, Numpy
* Data Visualization: Seaborn, Matplotlib
* Data Modeling and Evaluation: Scikit-learn

## **Conclusion**
This project demonstrates the application of machine learning techniques to predict complex real-world outcomes, such as the success of a rocket landing. The Decision Tree model, after extensive evaluation, proved to be the most effective in predicting the Falcon 9 first stage landing with high accuracy and reliability.

