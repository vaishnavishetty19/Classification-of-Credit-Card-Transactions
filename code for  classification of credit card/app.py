from flask import Flask, render_template, request
import pickle
import numpy as np
with open('rf_pickle_model.pkl','rb') as file:
    model = pickle.load(file)

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def home():
    V1 =  0.000010	
    V2 = -0.000021	
    V3 = -0.000015	
    V4 = -0.000020
    V5 = -0.000006	
    V6 = -0.000028	
    V7 =  0.000005	
    V8 = -0.000004	
    V9 = -0.000004	
    V10=  0.000005	
    V11= -0.000004	
    V12= -0.000022	
    V13=  0.000008	
    V14= -0.000003	
    V15=  0.000013	
    V16=  0.000026
    V17= -0.000016	
    V18=  0.000029
    V19=  0.000009
    V20=  0.000007
    V21= -0.000010	
    V22=  8.675109e-07	
    V23=  0.000001	
    V24=  0.000019	
    V25= -0.000007	
    V26=  0.000008	
    V27=  9.956689e-07	
    V28= -0.000001	
    average= sum([V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28])/28
    time= request.form['D']
    time_difference= request.form['E']
    Amount= request.form['F']
    cond= request.form['G']
    min= request.form['H']
    
    import pandas as pd
    import pickle
    df = pd.read_pickle('C:/Users/Vaishnavi M Shetty/Desktop/code for  classification of credit card/dfmod.pickle')
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    f1 = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','cond','time_difference','average']
    std_x = StandardScaler()
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train[f1] = std_x.fit_transform(X_train[f1])
    arr = [V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,cond,time_difference,average]
    arr1 = std_x.transform([arr]) 

    import json
   # arr1 = np.array([['arr']])
    arr2 = np.array([[time, min]])
    arr = np.concatenate((arr1, arr2), axis=1)

    output = model.predict(arr)
    return render_template('after.html', pred=min)
 


if __name__ == "__main__":
    app.run(debug=True)













