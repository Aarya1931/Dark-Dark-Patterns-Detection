from flask import Flask, request
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from flask import jsonify
import os

app = Flask(__name__)

REPO_ID = "pirocheto/phishing-url-detection"
FILENAME = "model.onnx"
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

@app.route("/")
def hello():
    return "Hello"

@app.route("/check-url", methods=["POST"])
def check_url():
    try:
        url = request.json.get("url")
        print("Received URL:", url)
        if not url:
            return "No URL provided in the request.", 400

       
        result = sess.run(None, {"inputs": np.array([url], dtype="str")})[1]

        
        return jsonify({"prediction": f"{result[0][1] * 100:.2f} %"})

    except Exception as e:
        print("Error:", e)
        return "Internal server error.", 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)