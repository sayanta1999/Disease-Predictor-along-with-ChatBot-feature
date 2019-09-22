# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:03:22 2019

@author: KIIT
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
import random
import webbrowser


def load_data():
    df_train = pd.read_csv(".\Manual-Data\Training.csv")
    df_train = shuffle(df_train)
    df_test = pd.read_csv(".\Manual-Data\Testing.csv")
    df_test = shuffle(df_test)
    return df_train, df_test


def train_test(train, test):
    x = train.drop('prognosis', axis=1)
    y = train['prognosis']
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(x, y)

    x_test = test.drop('prognosis', axis=1)
    y_test = test['prognosis']

    predict = rfc.predict(x_test)

    print(confusion_matrix(predict, y_test))
    print(f"accuracy : {np.mean(predict == y_test)}")

    plt.plot(y_test, predict, color='green')
    plt.show()

    # Testing with some random values
    # x_test1 = np.array(x_test)
    # temp = x_test1[10,:].reshape(1,-1)
    # print(y_test.iloc[10],rfc.predict(temp))

    return rfc


def analyze_data(df):
    # print(df.isnull().sum())

    for col in df.columns[:-1]:
        plt.hist(df[col], label=col)

    plt.legend(loc='best')
    plt.show()

def predict_disease2(model,data):
    print("Bot: Hi, This is your Virtual Assisstant, May i know your name")
    name = input('user: ')
    print("Bot: Could you please help me out to know about your symptoms better")
    print("Bot: Please answer the following symptoms with 'yes' or 'no' only\n Type quit whenever you want to end the chat")

    i = 0
    result = dict(zip(data.columns,[0]*len(data.columns)))
    
    pd.options.mode.chained_assignment = None  # default='warn'

    while(len(data.columns) > 0):
        col = data.columns[random.randint(0,len(data.columns)-1)]
        print(f"\nBot: {name}, Are you having {col}.")
        #ans = input(f"{name}: ")
        #ans = ans.lower()
        
        h = random.randint(0,1)
        if(h==1):
            ans='yes'
        else:
            ans='no'
        print(f"{name}: {ans}")
        
        if(ans=="quit"):
            break
        
        if(ans=="yes"):
            data = data[data[col]==1]
            result[col]=1
        elif(ans=="no"):
            data = data[data[col]==0]
        else:
            while(ans not in ['yes','no','quit']):
                print("Bot: Oops...Wrong Input, Try again")
                ans=input(f"{name}: ")
                ans=ans.lower()
            
            if(ans=="yes"):
                data = data[data[col]==1]
                result[col]=1
            elif(ans=="no"):
                data = data[data[col]==0]
            else:
                break

        data.drop(col,axis=1,inplace=True)
        
        for col in data:
            l=0
            for row in data[col]:
                if(row==0):
                    l=l+1
            if(l==len(data)):
                data.drop(col,axis=1,inplace=True)
        i=i+1
    
    pred_data=np.zeros(len(result))
    j=0
    for i in result:
        if(result[i]==1):
            pred_data[j]=result[i]
        j=j+1

    pred_data = pred_data.reshape(1,-1)
    print("\nProcess Completed")
    return model.predict(pred_data)


def predict_disease(model,col):
    print("Bot: Hi, This is your Virtual Assisstant 2.0, May i know your name")
    name = input('user: ')
    print("Bot: Could you please help me out to know about your symptoms better")
    print(
        "Bot: Please answer the following symptoms with 'yes' or 'no' only\n Type quit whenever you want to end the chat\n")

    result = np.zeros(len(col))
    i=0
    for sym in col:

        print(f"\nBot: {name}, Are you having {sym}")

        # ans = input(f"{name}: ")
        # ans = ans.lower()

        temp = random.randint(0, 1)
        if (temp == 1):
            ans = "yes"
        else:
            ans = "no"

        print(f"{name}: {ans}")
        if (ans.lower() == "quit"):
            break;

        if (ans == "yes"):
            result[i] = 1
        elif (ans == "no"):
            result[i] = 0
        else:
            while (ans not in ['yes','no','quit']):
                print("Bot: Oops...Wrong Input, Try again")
                ans = input(f"{name}: ")
        i=i+1
    result = result.reshape(1, -1)
    return model.predict(result)


def main():
    train, test = load_data()

    #analyze_data(train)

    model = train_test(train,test)

    disease = predict_disease2(model,train.drop('prognosis',axis=1))
    print(f'\nyou may have {disease[0]}')

    print("\nAre you satisfied yes/no?")
    ans = input()
    ans=ans.lower()

    while(ans not in ['yes','no']):
        ans=input("Retry: ")
        ans=ans.lower()

    if(ans=="no"):
        print("\nYou are in Next Level Checking")
        col = train.columns
        col = col[0:len(col)-1]
        disease = predict_disease(model,col)
        print(f"\nyou may have {disease[0]}")

    print("\nThankyou for using our Service")
    
    chrome_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
    webbrowser.register('chrome',None,webbrowser.BackgroundBrowser(chrome_path))
    webbrowser.get('chrome').open_new_tab("https://www.google.com/?#q="+f"treatment for {disease[0]}")
                  

if __name__ == '__main__':
    main()