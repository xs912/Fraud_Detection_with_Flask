from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # loading the model


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    type = int(request.form["type"]) #"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5
    amount = float(request.form["amount"])
    old_balance = float(request.form["old_balance"])
    new_balance = float(request.form["new_balance"])
    prediction = model.predict([[type, amount, old_balance, new_balance]])
    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'This transaction is suspected to be a {prediction[0]}')

if __name__ == "__main__":
    app.run()