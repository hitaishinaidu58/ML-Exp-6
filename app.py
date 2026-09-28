from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained Decision Tree model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Input data
    data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    # Prediction
    prediction = model.predict(data)

    return render_template(
        'index.html',
        prediction=prediction[0]
    )


if __name__ == '__main__':
    app.run(debug=True)