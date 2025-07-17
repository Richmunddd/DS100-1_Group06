import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'HR-Employee-Attrition.csv'  
data = pd.read_csv(file_path)  

print("Dataset preview:")
print(data.head())  

data_encoded = pd.get_dummies(data, drop_first=True)  

X = data_encoded.drop(columns='Attrition_Yes')  
y = data_encoded['Attrition_Yes']  

#(70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

#training set
rf_classifier.fit(X_train, y_train)

#predictions on the test set
y_pred = rf_classifier.predict(X_test)

# model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))

#classification report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix (True vs Predicted values)
cm = confusion_matrix(y_test, y_pred)

fig, axs = plt.subplots(1, 1, figsize=(18, 8))  

# Confusion Matrix  
def show_confusion_matrix():
    fig.clf()  
    axs = fig.add_subplot(111)  
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Attrition", "Attrition"], yticklabels=["No Attrition", "Attrition"], ax=axs, annot_kws={"size": 16})
    axs.set_title("Confusion Matrix", fontsize=18) 
    axs.set_xlabel("Predicted", fontsize=14)
    axs.set_ylabel("True", fontsize=14) 
    plt.draw()  

def show_feature_importance():
    features = X.columns  
    importances = rf_classifier.feature_importances_  

    feature_importance_df = pd.DataFrame({
        'Feature': features,  
        'Importance': importances 
    })

    # Sort the features based on importance, from most important to least important
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    fig.clf()  # Clear the entire figure
    axs = fig.add_subplot(111)  
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=axs, palette="Blues_d", hue='Feature', legend=False)
    axs.set_title("Feature Importance", fontsize=18)  
    axs.set_xlabel("Importance", fontsize=14)  
    axs.set_ylabel("Feature", fontsize=14)  
    plt.draw()  

show_confusion_matrix()

def on_key(event):
    if event.key == 'right':  
        show_feature_importance()
    elif event.key == 'left':  
        show_confusion_matrix()

fig.canvas.mpl_connect('key_press_event', on_key)  

plt.show()
