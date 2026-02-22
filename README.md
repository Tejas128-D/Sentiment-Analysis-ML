# Sentiment Analysis Project

## 📌 Project Objective

The objective of this project is to classify product reviews into **Positive, Negative, and Neutral** sentiments using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 📊 Exploratory Data Analysis (EDA)

During the EDA phase, the following steps were performed:

- Loaded and inspected the dataset
- Checked for missing values
- Analyzed rating distribution
- Created sentiment labels from the rating column
- Visualized sentiment distribution using bar plots
- Analyzed review length distribution
- Generated WordClouds for Positive, Negative, and Neutral reviews

EDA helped in understanding:
- Class imbalance
- Data distribution
- Text patterns
- Overall data quality

---

## 🛠 Text Preprocessing & Feature Engineering

The following preprocessing steps were applied:

- Removed special characters and punctuation
- Converted text to lowercase
- Removed stopwords using NLTK
- Created a cleaned text column
- Converted text into numerical format using **TF-IDF Vectorization**

TF-IDF was used to transform textual data into feature vectors suitable for machine learning models.

---

## 🤖 Model Building

Multiple machine learning models were trained and evaluated:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

Models were evaluated using:

- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

---

## ⚙ Hyperparameter Tuning

Hyperparameter tuning was performed on **Logistic Regression** using **GridSearchCV with 5-fold cross-validation**.

Parameters tuned:

- C: [0.01, 0.1, 1, 10]
- Penalty: L2
- Solver: liblinear, lbfgs

### Best Parameters Found:

- C = 10  
- Penalty = L2  
- Solver = liblinear  

### Best Cross-Validation Score:

~77.5%

After hyperparameter tuning, **Logistic Regression achieved the best overall performance** and was selected as the final model.

---

## 🚀 Model Deployment

The final tuned Logistic Regression model was:

- Saved using `pickle`
- Deployed using **Streamlit**
- Integrated into a web application for real-time sentiment prediction

Users can input review text and get instant sentiment prediction.

---

## 📦 Technologies Used

- Python
- Pandas
- Numpy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- GridSearchCV
- Matplotlib & Seaborn
- Streamlit

---

## 📈 Project Highlights

✔ Complete NLP pipeline  
✔ Feature engineering & preprocessing  
✔ Multiple model comparison  
✔ Hyperparameter tuning using GridSearchCV  
✔ Cross-validation  
✔ Model deployment using Streamlit  

---

## 👨‍💻 Author

Tejas R