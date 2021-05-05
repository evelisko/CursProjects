# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

import json
import os
import dill
import flask
from logger import Logger
from recomender import Recomender

# import the necessary packages
dill._dill._reverse_typemap['ClassType'] = type

# initialize our Flask application and the model
app = flask.Flask(__name__)

dataframe_path = ""
model_path = ""
loger_patch = ""
config = {}
config_patch = "/app/app/config.json"

with open(config_patch, 'r', encoding='utf8') as f:
    config = json.load(f)
    dataframe_path = config['dataframe_path']
    model_path = config['model_path']
    loger_patch = config['loger_patch']

log = Logger(loger_patch)
log.write('Run Flask Server')
print('Run Flask Server')

recomender = Recomender(model_path, dataframe_path, log)

@app.route("/", methods=["GET"])
def general():
    return """Welcome to fraudelent prediction process. Please use 'http://<address>/netflix_films' to POST"""


@app.route("/netflix_films", methods=["POST"])
def netflix_films():
    data = {"success": False}
    if flask.request.method == "POST":
        try:
            request_json = flask.request.get_json()
            if request_json["method"]:
                method_name = request_json["method"]

                if method_name == "search_films":
                    data["titles"] = recomender.search_films(request_json)
                elif method_name == "film_info":
                    data["description"] = recomender.film_info(request_json)
                elif method_name == "recomendations":
                    data["titles"] = recomender.get_recomendations(request_json)
        except Exception as e:
            log.write(f'Exception: {str(e)}')
            data["success"] = False
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
