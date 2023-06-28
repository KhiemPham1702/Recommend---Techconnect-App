from flask import Flask, jsonify, request, g
from flask_cors import CORS
import recommend1
import recommend2
import json

app = Flask(__name__)
app.debug = True
CORS(app)

global user
user = {}


@app.before_request
def before_request():
    g.user = user

@app.route('/sever/send-data', methods=['POST'])
def send_data():
    data = request.json
    # Xử lý dữ liệu ở đây
    global user
    user = data['user']
    g.user = data['user']

    print(data['user'])

    with open("./data/test_pro.json", "w", encoding="utf-8") as outfile:
        json.dump(data['product'], outfile, ensure_ascii=False, indent=1, separators=(", ", ": "))
    with open("./data/test_rating.json", "w", encoding="utf-8") as outfile:
        json.dump(data['rating'], outfile , ensure_ascii=False, indent=1, separators=(", ", ": "))
    result = {'message': 'Data received successfully!'}
    return jsonify(result)

@app.route('/sever/get-data', methods=['GET'])
def get_data():
    # Lấy dữ liệu ở đây
    if g.user['movie'] != '':
        recommendations = recommend1.get_hybrid_recommendations(g.user['id'],g.user['movie'])
        json_recommendations = recommendations.to_json(orient='records')
        data = json_recommendations
    else:
        recommendations = recommend2.get_data_hybrid(g.user['id'])
        data = json.dumps(recommendations)
    # data = recommend.json_recommendations
    return jsonify(data)

if __name__ == '__main__':
    app.run()
