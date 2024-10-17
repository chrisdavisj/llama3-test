from flask import Flask, request, jsonify
from llama import Llama
import time
import uuid
import json
import os

# Initialize Flask app
app = Flask(__name__)

# Load the LLaMA model
ckpt_dir = "/opt/llama3"  # Path to the model checkpoint
tokenizer_path = "/opt/llama3/tokenizer.model"  # Path to the tokenizer
max_seq_len = 512  # Maximum sequence length for LLaMA
max_batch_size = 4  # Maximum batch size for generating responses

# Log file for storing responses
log_file = "responses_log.json"

# Ensure the log file exists
# if not os.path.exists(log_file):
#     with open(log_file, 'w') as f:
#         pass  # Create the file if it doesn't exist

# Build the LLaMA generator
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    # Get the input data from the request
    data = request.get_json()

    # The user will send the entire conversation as a list of messages
    messages = data.get("messages", [])
    
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    # Optional parameters for generation control
    temperature = data.get("temperature", 0.6)  # Default temperature
    top_p = data.get("top_p", 0.9)  # Default top_p
    max_gen_len = data.get("max_gen_len", None)  # Optional max generation length
    
    # Ensure that the format of messages is as expected (system, user, assistant pairs)
    dialogs = [messages]  # Wrapping the message list into a single dialog structure
    
    # Measure the token length of the prompt (conversation context)
    prompt_tokens = sum(len(generator.tokenizer.tokenize(msg['content'])) for msg in messages)

    # Generate the assistant's response using the LLaMA 3 model
    start_time = time.time()  # Track the time when generation starts
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    end_time = time.time()  # Track the time when generation ends

    # Measure the token length of the completion (the assistant's response)
    completion_tokens = len(generator.tokenizer.tokenize(results[0]['generation']['content']))
    
    # Create metadata for the response
    total_tokens = prompt_tokens + completion_tokens
    timestamp = int(start_time)  # Time created
    response_id = str(uuid.uuid4())  # Unique response ID
    
    # Construct the final response object, including the original results
    response = {
        "id": response_id,  # Unique ID for this chat completion
        "created": timestamp,  # Time of generation
        "usage": {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        },
        "time_taken": round(end_time - start_time, 3),  # Time taken for the generation (in seconds),
        "start_time": start_time,
        "results": results  # Full results object from LLaMA 3 model
    }
    print(json.dumps(response))

    # # Store the response in the log file
    # with open(log_file, 'a') as f:
    #     f.write(json.dumps(response) + "\n")  # Write each response as a new line in JSON format

    # Return the response as a JSON object
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    app.run(port=8010)
    #pass
