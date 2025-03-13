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
    data = request.get_json()
    message = data.get("message", "")

    # Append the user's message
    conversation.append({"role": "user", "content": message})

    # Use Qwen's chat template for conversation
    prompt_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1000)
    # Extract newly generated tokens (skip the input length)
    new_tokens = output_ids[0][len(inputs.input_ids[0]):]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Append the assistant's response
    conversation.append({"role": "assistant", "content": response_text})

    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

