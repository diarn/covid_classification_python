# importing lib
from urllib import request
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from json import JSONEncoder
from flask import Flask,json,request
from yellowbrick.classifier import ClassificationReport
from IPython.display import display



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# app = main()
# @app.route("/classification")

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)



def main():
    my_symptoms_int = [0,1,1,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1]
    dataset = pd.read_csv("data/large_data.csv")
    # display(dataset.head())
    # display(dataset.tail())

    # split the data into inputs and outputs
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
    y = dataset.iloc[:, 20].values
    allergyOfy = np.where(y == "ALLERGY")
    coldOfy = np.where(y == "COLD")
    covidOfy = np.where(y == "COVID")
    fluOfy = np.where(y == "FLU")
    # print(np.shape(allergyOfy))
    # print(np.shape(coldOfy))
    # print(np.shape(covidOfy))
    # print(np.shape(fluOfy))

    #training and testing data
    from sklearn.model_selection import train_test_split

    #assign test data size 25%
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5, random_state=0)
    # print(X_train[0])
    # print(X_train[1])
    # print(X_train[2])
    # print(X_train[3])
    # print(X_train[4])

    #importing standard scaler
    from sklearn.preprocessing import StandardScaler

    #scaling the input data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)


    #import bernoulli naive bayes classifier
    from sklearn.naive_bayes import BernoulliNB

    #create bernoulli claassifier

    classifier1 = BernoulliNB()


    #train the model
    classifier1.fit(X_train,y_train)

    #testing the model
    y_pred1 = classifier1.predict(X_test)
    y_pred2 = classifier1.predict(X_train)
    # print(y_pred2[0])
    # print(y_pred2[1])
    # print(y_pred2[2])
    # print(y_pred2[3])
    # print(y_pred2[4])
    print(classifier1.classes_)
    print(classifier1.class_count_)
    print(classifier1.feature_count_)



    # importing accuracy score
    import sklearn
    from sklearn.metrics import classification_report, plot_confusion_matrix
    # print("=======================================CLASSIFICATION REPORT TEST DATA NV=========================================")
    # print(classification_report(y_test,y_pred1,zero_division=1))

    # color = "black"
    # matrix = plot_confusion_matrix(classifier1,X_test,y_test, cmap=plt.cm.Blues)
    # matrix.ax_.set_title('Confusion Matrix', color=color)
    # plt.xlabel('Predicted Label', color=color)
    # plt.ylabel('True Label', color=color)
    # plt.gcf().axes[0].tick_params(colors=color)
    # plt.gcf().axes[1].tick_params(colors=color)
    # plt.show()


    # clf_report = classification_report(y_test,y_pred1,output_dict=True)
    # sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    # plt.show()


    # my_np_input = np.array([my_symptoms_int])
    # print(classifier1.predict_proba(X_train))


if __name__ == "__main__":
    main()
    # app.run()
    # app.run(host="0.0.0.0", port=8000)