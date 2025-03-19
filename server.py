from flask import Flask, request, jsonify, render_template
import torch, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
app = Flask(__name__, template_folder="templates")

print("Loading Qwen model...")

start_time = time.time()
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
load_time = time.time() - start_time

print(f"Model loaded in {load_time:.2f} seconds")

# Single conversation stored in memory
conversation = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

