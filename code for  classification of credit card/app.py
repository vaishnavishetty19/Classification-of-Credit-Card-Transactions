from flask import Flask, render_template, request
import pickle
import numpy as np
with open('rf_pickle_model.pkl','rb') as file:
    model = pickle.load(file)

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    V1 = request.form['a']
    V2 = request.form['b']
    V3 = request.form['c']
    V4 = request.form['d']
    V5 = request.form['e']
    V6 = request.form['f']
    V7 = request.form['g']
    V8 = request.form['h']
    V9 = request.form['i']
    V10= request.form['j']
    V11= request.form['k']
    V12= request.form['l']
    V13= request.form['m']
    V14=request.form ['n']
    V15= request.form['o']
    V16= request.form['p']
    V17= request.form['q']
    V18= request.form['r']
    V19= request.form['s']
    V20= request.form['t']
    V21= request.form['u']
    V22= request.form['v']
    V23= request.form['w']
    V24= request.form['x']
    V25= request.form['y']
    V26= request.form['z']
    V27= request.form['A']
    V28= request.form['B']
    average= request.form['C']
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

    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)














