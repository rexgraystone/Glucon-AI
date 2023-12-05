from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("diabetes.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if prediction==0:
        return render_template('diabetes.html',pred="The patient does not have diabetes")
    else:
        return render_template('forest_fire.html',pred="The patient has diabetes")


if __name__ == '__main__':
    app.run(debug=True)
