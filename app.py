from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load the model and tokenizer (use CPU-friendly config)
model = AutoModelForCausalLM.from_pretrained("breathingbrushes_gpt")
tokenizer = AutoTokenizer.from_pretrained("breathingbrushes_gpt")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('prompt')
    input_text = f"Prompt: {user_input}\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=150,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response_text.split("Response:")[-1].strip()
    return jsonify({'response': answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

