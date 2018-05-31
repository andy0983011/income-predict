import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.cross_validation as cross_validation
import sklearn.preprocessing as preprocessing

from keras.models import Sequential
from keras.layers import Dense, Activation

import statsmodels as sm
import sklearn as skl
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns

# for random forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation  import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
#X_train為 X TRAINING features
#X_test 為 X TRAINING 的測試資料
#Y_train 為 Y 的TRAINING feature

def normalize(X_all, X_test):

    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma
    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def preproessing_onehot(original_data, input_name):

#    print(list(original_data))
    if "sex" in list(original_data):
        sex_mapping = {'Male': 0, 'Female': 1}
        original_data['sex'] = original_data['sex'].map(sex_mapping)
        
    if "income" in list(original_data):    
        target_mapping = {'<=50K': 0,  '>50K': 1}
        original_data['income'] = original_data['income'].map(target_mapping)
        
    i = 0
    for data_name in original_data.columns.values:
        if original_data[data_name].unique().dtype !="object":
            original_data[[data_name]].values.astype(float)   
            
        if original_data[data_name].unique().dtype != "int64":
            i = i + original_data[data_name].unique().shape[0]
            onehot_encodeing = pd.get_dummies(original_data[data_name])
            original_data = original_data.drop(data_name,1)
            original_data = pd.concat([onehot_encodeing, original_data], axis=1)  
            
        elif original_data[data_name].unique().dtype == "int64":
            if data_name != "income":
                i = i + 1;
            if data_name !="sex" or data_name !="income":
#                print("=========================DATA NAME => " + data_name)
#                print(original_data[data_name].unique())
                normal = original_data[[data_name]].values.astype(float)
                
    print("Input Data Name: " + input_name)            
    print("Data Columns Count: " + str(i))
    return original_data

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

if __name__ == '__main__':
    
    train_original_data = pd.read_csv("train.csv",
                                    sep=r'\s*,\s*',
                                    engine='python',
                                    na_values="?")
    
    test_original_data = pd.read_csv("test.csv",
                                    sep=r'\s*,\s*',
                                    engine='python',
                                    na_values="?")
    
#    print(train_original_data.head())
    
    training_data = preproessing_onehot(train_original_data, "train_original_data")
    training_data.to_csv("preprocessing_training_data.csv")
    
    testing_data = preproessing_onehot(test_original_data, "test_original_data")
#    testing_data.to_csv("X_test.csv")
    
    ##X_train data preprocessing
    X_train = training_data.loc[:, training_data.columns != 'income']
    X_train.to_csv("X_train.csv")
    X_train = X_train.values
    print("X_train.shape = ", X_train.shape)
    print("X_train = ", X_train)
    
    ##Y_train data preprocessing
    Y_train = training_data[["income"]]
    Y_train.to_csv("Y_train.csv")
    Y_train = Y_train.values
    print("Y_train.shape = ",Y_train.shape)
    print("Y_train = ", Y_train)
    
    ##X_test data preprocessing
    testing_data.to_csv("X_test.csv")
    X_test = testing_data.values
    print("X_test.shape = ", X_test.shape)
    print("X_test = ", X_test)   
    
#    Y_train.to_csv("Y_train.csv",  index = False)
      
    
    
#    x = training_data.loc[:, training_data.columns != 'income']
#    y = training_data[["income"]]
#    x.to_csv("training_data.csv")
#    y.to_csv("testing_data.csv")
#    print(x.shape)
#    print(y.shape)
#    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, train_size=0.70)
#    scaler = preprocessing.StandardScaler()
#    X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
#    X_test = scaler.transform(X_test.astype("float64"))
#
#    cls = linear_model.LogisticRegression()
#    
#    cls.fit(X_train, y_train)
#    y_pred = cls.predict(X_test)
#    cm = metrics.confusion_matrix(y_test, y_pred)
#  
#    print ("F1 score: %f" % skl.metrics.f1_score(y_test, y_pred))


#    plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
#    plt.xlabel('petal length [standardized]')
#    plt.ylabel('petal width [standardized]')
#    plt.legend(loc='upper left')
#    plt.show()
    
    
#    fig = plt.figure(figsize=(20,15))
#    cols = 5
#    rows = ceil(float(train_original_data.shape[1]) / cols)
#    for i, column in enumerate(train_original_data.columns):
#        ax = fig.add_subplot(rows, cols, i + 1)
#        ax.set_title(column)
#        if train_original_data.dtypes[column] == np.object:
#            train_original_data[column].value_counts().plot(kind="bar", axes=ax)
#        else:
#            train_original_data[column].hist(axes=ax)
#            plt.xticks(rotation="vertical")
#    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    

#    print(original_data.columns)
#    print(original_data.iloc[0])
#    print(original_data.Workclass)
#    print(original_data.Workclass.value_counts())
#    print(original_data.Workclass.unique())

    
#    y_train.to_csv("y_train.csv")
#    original_data.to_csv("test3.csv")
    
#    training_data = preproessing_onehot(train_original_data)
#    training_data.to_csv("training_data.csv")

    
#    high_income = training_data[training_data['Target'] == 1]
#    low_income = training_data[training_data['Target'] == 0]
#    train = pd.concat([high_income.sample(frac=0.8, random_state=1),
#                   low_income.sample(frac=0.8, random_state=1)]) 
#    y_train  =  train['Target']        
#    print(y_train.head())
#    X_train = train[predictors]


