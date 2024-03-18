from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
# importing standard scalling method from sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


app=Flask(__name__)
data = pd.read_csv("D:\Programming\Projects\Pavan\Online_Fraud.csv")
accuracies = []
accur = {}
type_new = pd.get_dummies(data['type'], drop_first=True)
data_new = pd.concat([data, type_new], axis=1)
print(f"Concatinating Dummy value \n{data_new.head()}")

X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']
print(f"X Shape: {X.shape} ; y Shape: {y.shape}")
# import library to split training and testing data
from sklearn.model_selection import train_test_split

    # X_train - Training Dataset
    # y_train - Expected Training Results

    # X_test - Testing Dataset
    # y_test - Expected Testing Results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
models = {
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_preds = model.predict_proba(X_train)[:, 1]
    train_acc = roc_auc_score(y_train, train_preds)
    test_preds = model.predict_proba(X_test)[:, 1]
    test_acc = roc_auc_score(y_test, test_preds)
    accuracies.append((name, train_acc, test_acc))
    accur[name] = (train_acc, test_acc)

    # Find the model with the highest test accuracy
best_model_name = max(accur, key=lambda k: accur[k][1])
best_model_train_accuracy = accur[best_model_name][0]
best_model_test_accuracy = accur[best_model_name][1]
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def train():
    global X_train, X_test, y_train, y_test,model
    # Reading the Dataset
    

    # Print some sample data from dataset
    print(f"First 5 Columns \n{data.head()}")
    
    print(f"Dataset info \n{data.info()}")
    
    #plot
    sns.countplot(x='type', data=data)
    plt.savefig('D:\Programming\Projects\Pavan\static\plot1.png')

    sns.barplot(x='type', y='amount', data=data)
    plt.savefig('D:\Programming\Projects\Pavan\static\plot2.png')

    #Confirming the rows of the class labels
    print(data['isFraud'].value_counts())

    #plot for the entire data
    plt.figure(figsize=(15,6))
    sns.displot(data['step'], bins=50)
    plt.savefig('D:\Programming\Projects\Pavan\static\plot3.png')

    #plotting the heatmap for the dataset
    '''plt.figure(figsize=(12,6))
    sns.heatmap(data.corr(), cmap='BrBG', fnt='.2f', linewidths=2, annot=True)
    plt.savefig("plot4.png")'''



    '''models = [LogisticRegression(), XGBClassifier(),
              SVC(kernel='rbf', probability=True),
              RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)]

    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        print(f'{models[i]}: ')

        train_preds = models[i].predict_proba(X_train)[:, 1]
        print('Training Accuracy : ',ras(y_train, train_preds))

        y_preds = models[i].predict_proba(X_test)[:, 1]
        print('Validation Accuracy : ', ras(y_test, y_preds))
        print()'''

    


    best_model = models[best_model_name]
    filename = 'best_model.pkl'
    pickle.dump(best_model, open(filename, 'wb'))


    return render_template("train.html")


@app.route("/accuracy")
def accuracy():
    return render_template("accuracy.html",accuracies=accur, best_model_name=best_model_name,
                           best_model_train_accuracy=best_model_train_accuracy,
                           best_model_test_accuracy=best_model_test_accuracy)


@app.route("/recommend")
def recommend():
    return render_template("predict.html")


@app.route("/predictCrop",methods=['POST'])
def predictCrop():
    v1=request.form['t1']
    v2=request.form['t2']
    v3=request.form['t3']
    v4=request.form['t4']
    v5=request.form['t5']
    v6=request.form['t6']
    v7=request.form['t7']
    v8=request.form['t8']
    v9=request.form['t9']
    v10=request.form['t10']
    filename = 'cropmodel.pkl'
    model = pickle.load(open(filename, 'rb'))
    sea=0
    if v9=='Kharif':
        sea=1
    elif v9=='Autumn':
        sea=2
    elif v9=='Rabi':
        sea=3
    elif v9=='summer':
        sea=4
    elif v9=='winter':
        sea=5
    elif v9=='Whole Year':
        sea=6

    x=model.predict([[v1, v2, v3, v4, v5, v6,v7, v8, float(sea),v10]])
    print(x)

    return render_template("predictcrop.html",rc=x)


app.run(debug=True)
