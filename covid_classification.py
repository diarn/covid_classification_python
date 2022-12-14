# importing lib
from urllib import request
import numpy as np
import matplotlib
# matplotlib.use('GTK4Agg')
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from json import JSONEncoder
from flask import Flask,json,request
from yellowbrick.classifier import ClassificationReport



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = Flask(__name__)
@app.route("/classification")
def main():
    my_symptoms = request.args.get("symptoms",type=list)
    if(len(my_symptoms) < 20):
        return app.response_class(
            response=json.dumps(
                {
                    "status":"fail",
                    "message":"maaf anda harus mengkonfirmasi 20 gejala",
                    "symptoms_length":len(my_symptoms)
                }
                ),
            status=200,
            mimetype="application/json"
        )
    if(len(my_symptoms) > 20):
        return app.response_class(
            response=json.dumps(
                {
                    "status":"fail",
                    "message":"maaf anda memasukkan data lebih dari 20 gejala",
                    "symptoms_length":len(my_symptoms)
                }
                ),
            status=200,
            mimetype="application/json"
        )
    my_symptoms_int = []
    for i in range(len(my_symptoms)):
        my_symptoms_int.append(int(my_symptoms[i]))

    #importing dataset
    dataset = pd.read_csv("data/large_data.csv")

    # split the data into inputs and outputs
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
    y = dataset.iloc[:, 20].values

    #training and testing data
    from sklearn.model_selection import train_test_split

    #assign test data size 25%
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8, random_state=0)

    # print("==============y_test==============")
    # print(y_test)
    # print("==============y_test.shape==============")
    # print(y_test.shape)

    #importing standard scaler
    from sklearn.preprocessing import StandardScaler

    #scaling the input data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    # print("=======================================X TEST=========================================")
    # print(X_test)
    X_test = sc_X.fit_transform(X_test)
    # print("=======================================FITTED X TEST=========================================")
    # print(X_test)
    # print("=======================================FITTED X TEST SHAPE=========================================")
    # print(X_test.shape)


    #import bernoulli naive bayes classifier
    from sklearn.naive_bayes import BernoulliNB
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.linear_model import LogisticRegression

    #create bernoulli claassifier

    classifier1 = BernoulliNB()
    # classifierKNN = KNeighborsClassifier()
    # classifierLR = LogisticRegression()

    #train the model
    classifier1.fit(X_train,y_train)
    # classifierKNN.fit(X_train,y_train)
    # classifierLR.fit(X_train,y_train)

    #testing the model
    y_pred1 = classifier1.predict(X_test)
    # plt.figure(figsize=(6, 10))
    # ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
    # sns.distplot(y_pred1, hist=False, color="b", label="Fitted Values" , ax=ax1)
    # plt.title('DIST PLOT Random Forest')
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.show()
    # plt.close()
    # y_predKNN = classifierKNN.predict(X_test)
    # y_predLR = classifierLR.predict(X_test)

    # importing accuracy score
    from sklearn.metrics import accuracy_score, classification_report
    print("=======================================CLASSIFICATION REPORT TEST DATA NV=========================================")
    print(classification_report(y_test,y_pred1,zero_division=1))
    # classes = ["FLU","COVID","COLD","ALLERGY"]
    # print("=======================================CLASSIFICATION REPORT TEST DATA KNN=========================================")
    # print(classification_report(y_test,y_predKNN,zero_division=1))
    # print("=======================================CLASSIFICATION REPORT TEST DATA LR=========================================")
    # print(classification_report(y_test,y_predLR,zero_division=1))


    my_np_input = np.array([my_symptoms_int])

    my_output1 = classifier1.predict_proba(my_np_input)

    jsonData1 = json.dumps({
        "status":"success",
        "message":"success to retrive data",
        "data":{
            "ALLERGY":round(my_output1[0][0]*100,3),
            "COLD":round(my_output1[0][1]*100,3),
            "COVID":round(my_output1[0][2]*100,3),
            "FLU":round(my_output1[0][3]*100,3),
            }
        })
    response = app.response_class(
        response = jsonData1,
        status=200,
        mimetype="application/json"
    )
    return response

@app.route("/variant-classification")
def variantClassiviation():
    my_symptoms = request.args.get("symptoms",type=list)
    if(len(my_symptoms) < 15):
        return app.response_class(
            response=json.dumps(
                {
                    "staatus":"fail",
                    "message":"maaf anda harus mengkonfirmasi 15 gejala",
                    "symptoms_length":len(my_symptoms)
                }
                ),
            status=200,
            mimetype="application/json"
        )
    if(len(my_symptoms) > 15):
        return app.response_class(
            response=json.dumps(
                {
                    "staatus":"fail",
                    "message":"maaf anda memasukkan data lebih dari 15 gejala",
                    "symptoms_length":len(my_symptoms)
                }
                ),
            status=200,
            mimetype="application/json"
        )

    
    my_symptoms_int = []
    for i in range(len(my_symptoms)):
        my_symptoms_int.append(int(my_symptoms[i]))



    #importing dataset
    dataset = pd.read_csv("data/variant_dataset.csv")
    # split the data into inputs and outputs
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]].values
    y = dataset.iloc[:, 15].values

    #training and testing data
    from sklearn.model_selection import train_test_split

    #assign test data size 25%
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.7, random_state=0)

    #importing standard scaler
    from sklearn.preprocessing import StandardScaler

    #scaling the input data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    #import gaussian naive bayes classifier
    from sklearn.naive_bayes import GaussianNB,BernoulliNB

    #create gaussian claassifier

    classifier1 = BernoulliNB()

    #train the model
    classifier1.fit(X_train,y_train)

    #testing the model
    y_pred1 = classifier1.predict(X_test)

    # importing accuracy score
    from sklearn.metrics import accuracy_score, classification_report

    print(classification_report(y_test,y_pred1,zero_division=1))


    my_np_input = np.array([my_symptoms_int])
    my_output1 = classifier1.predict_proba(my_np_input)

    jsonData1 = json.dumps({
        "status":"success",
        "message":"success to retrive data",
        "data":{
            "ALPHA":round(my_output1[0][0]*100,3),
            "DELTA":round(my_output1[0][1]*100,3),
            "OMICRON":round(my_output1[0][2]*100,3),
            }
        })
    response = app.response_class(
        response = jsonData1,
        status=200,
        mimetype="application/json"
    )
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)