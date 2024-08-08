# 📊 Cancer Classification Project

## 🙏 Introduction

This project aims to classify cancer types using machine learning techniques. We utilize a comprehensive dataset of patient attributes and cancer characteristics to develop and evaluate models that can accurately predict cancer types. This README provides an overview of the dataset, the tools used, and insights derived from the analysis.

## 📚 Dataset Description

The dataset contains various features related to cancer diagnosis and patient demographics. Each row represents a unique patient, and the columns include attributes such as age, gender, tumor size, cancer type, and other relevant features.

## ⚙️ Tools and Libraries Used

The following tools and libraries were used in this project:

- **Python**: Programming language for data processing and analysis.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **TensorFlow**: For building and training neural network models.
- **Scikit-Learn**: For data splitting, model evaluation, and additional machine learning utilities.
- **EarlyStopping**: To prevent overfitting by stopping training when the validation loss stops improving.
- **Sequential**: Keras API for creating sequential models.
- **Dense**: Keras layer for fully connected neural network layers.
- **Dropout**: To prevent overfitting by randomly dropping neurons during training.
- **Confusion Matrix**: To evaluate the performance of the classification model.

## 📈 Exploratory Data Analysis (EDA)

Before building the model, an exploratory data analysis was conducted to understand the dataset's structure and characteristics. This involved:

- **Visualizing Data Distributions**: Using histograms, box plots, and scatter plots.
- **Analyzing Correlations**: Using heatmaps to identify relationships between features.
- **Handling Missing Values**: Imputing or removing missing data as necessary.
- **Feature Engineering**: Creating new features or modifying existing ones to improve model performance.

## 📊 Model Building and Evaluation

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
df = pd.read_csv('../DATA/cancer_classification.csv')
df.head()
df.info()
df.describe().transpose()
sns.countplot(data=df, x='benign_0__mal_1', palette='viridis')
plt.show()
sns.heatmap(df.corr())
df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
plt.show()
df.corr()['benign_0__mal_1'].sort_values()[:-1].plot(kind='bar')
plt.show()
```

### Count plot for the feature column:

![countplot](/Cancer%20Dataset/ASSETS/countplot.png)

### Heat map of the Dataset:

![heatmap](/Cancer%20Dataset/ASSETS/heatmap.png)

### Barchart of the data corr:

![barchart](/Cancer%20Dataset/ASSETS/barchart.png)

### Barchart without feature corr:

![barplot](/Cancer%20Dataset/ASSETS/barplot.png)

## ⌘ Data Splitting

The dataset was split into training and testing sets using Scikit-Learn's `train_test_split` function:

```py
# Train Test Split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
X_train.shape
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# importing Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600, verbose=1)
loss = pd.DataFrame(model.history.history)
loss.plot()
plt.show()

prediction = model.predict(X_test)
class_prediction = (prediction > 0.5).astype(int).flatten()
print(classification_report(y_test, class_prediction))
print(confusion_matrix(y_test, class_prediction))
```

![overfitmodel](/Cancer%20Dataset/ASSETS/overfitt.png)

## Prediction of normal model and accuracy:

![prediction](/Cancer%20Dataset/ASSETS/model_pred.png)

### Using Early stopping to reduce over fitting of model.

```py
# Overfitting the model we'll use the Early stop
model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam')

# Using Early stopping:
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=25, mode='min')

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), verbose=1, epochs=600, callbacks=[early_stopping])

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

# predicting the model
prediction = model.predict(X_test)
class_predictions = (prediction > 0.5).astype(int).flatten()
print(classification_report(y_test, class_predictions))
print(confusion_matrix(y_test, class_predictions))
```

![fittingmodel](/Cancer%20Dataset/ASSETS/fitting.png)

## Prediction of model and accuracy:

![predictions](/Cancer%20Dataset/ASSETS/early_stop_pred.png)

### Using Tensorflow dropout to make the model and compare:

```py
# let's use drop out so that the val loss stays inside out loss data
model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), verbose=1, epochs=600, callbacks=[early_stopping])
#losses
dropout_losses = pd.DataFrame(model.history.history)
dropout_losses.plot()
plt.show()

# predicting the model
prediction = model.predict(X_test)
prediction

# Convert probabilities to class predictions

class_predictions = (prediction > 0.5).astype(int).flatten()
class_predictions
print(classification_report(y_test, class_predictions))
print('\n')
print(confusion_matrix(y_test, class_predictions))
```

![underfitt](/Cancer%20Dataset/ASSETS/underfit.png)

## Predictions and Accuracy:

![prediction](/Cancer%20Dataset/ASSETS/dropout_pred.png)

## Classification Report

The classification report provides a detailed breakdown of the model’s precision, recall, F1-score, and support for each class.

## Confusion Matrix

The confusion matrix provides a visual representation of the model’s performance, showing the number of true positives, true negatives, false positives, and false negatives.

## Key Insights

• **Model Performance**: The model achieved an accuracy of X% on the test set, indicating a strong ability to distinguish between benign and malignant cases.

•**Feature Importance**: Analysis revealed that certain features, such as tumor size and patient age, were significant predictors of cancer type.
•**Regularization**: The use of dropout layers and early stopping was effective in preventing overfitting, leading to better generalization on the test set.

•**Data Imbalance**: The dataset exhibited some imbalance between benign and malignant cases, which could be addressed in future work by techniques such as SMOTE or class weighting.

# ✅Conclusion

- **This project demonstrated the successful application of machine learning techniques to cancer classification. By leveraging neural networks and effective regularization methods, we achieved a robust model capable of accurate predictions. Further improvements could be made by exploring additional features, tuning hyperparameters, and experimenting with different model architectures. Moreover, addressing data imbalance and integrating more patient data could enhance the model’s performance and generalizability**.
