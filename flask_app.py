import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='댓글이 악성일 확률 {}%'.format(output*100))

@app.route('/results', methods=['POST'])
def results():
    print("result 실행 됨------------------")
    print("결과: " + str(request))
    #print("결과: " + str(list(request.get_json(force=True).values())))
    try:
        data = request.get_json(force=True)
    except:
        print("json 파싱 실패")
        return jsonify(-1)
    else:
        print("json 파싱 성공")

    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    print("----------------------")
    return jsonify(output)
    

if __name__ == "__main__":
    app.run(port='5000' ,debug=True)