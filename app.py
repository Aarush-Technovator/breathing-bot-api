from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This allows all domains to access API. can restrict it later.

model_path = "breathingbrushes_gpt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()

@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.json.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return jsonify({"response": response})

@app.route('/')
def health():
    return 'Breathing Bot API is running!', 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
