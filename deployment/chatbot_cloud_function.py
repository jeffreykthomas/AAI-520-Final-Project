import functions_framework
from flask import make_response, jsonify
import requests
import os
import time
import openai
import logging
import tiktoken

logging.basicConfig(level=logging.DEBUG)

MEDIUM_API_URL = "https://api-inference.huggingface.co/models/jeffreykthomas/dialo-medium-ubuntu-generation"
LARGE_API_URL = "https://api-inference.huggingface.co/models/jeffreykthomas/dialo-large-ubuntu-generation"

headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"}
openai.api_key = os.environ['OPENAI_API_KEY']


def trim_conversation(conversation, max_tokens=1000):
    encoding = tiktoken.get_encoding("cl100k_base")
    current_tokens = 0
    trimmed_conversation = []
    for question in reversed(conversation):
        question_tokens = encoding.encode(question['text'])
        if current_tokens + len(question_tokens) < max_tokens:
            trimmed_conversation.append(question)
            current_tokens += len(question_tokens)
        else:
            break

    trimmed_conversation.reverse()
    return trimmed_conversation


def run_dialo_query(conversation, size='medium'):
    logging.debug('running query')
    """
    Queries a fine-tuned Dialo model using a given question.
    Args:
    - question (str): the question to be answered.
    - size (str): the size of the Dialo model to use. Must be either 'medium' or 'large'.

    Returns:
    - answer (dict): the answer from the Dialo model.
    """
    if not conversation:
        raise ValueError("Conversation must not be None or an empty string.")

    if size == 'medium':
        API_URL = MEDIUM_API_URL
    else:
        API_URL = LARGE_API_URL

    # Strip the question of any trailing whitespace
    for question in conversation:
        question['text'].strip()
        question['text'].replace('\n', '')

    # Trim conversation to last 1000 tokens
    trimmed_conversations = trim_conversation(conversation)

    inputs = []
    for question in trimmed_conversations:
        inputs.append(question['text'].strip() + ' <|endoftext|>')

    # join the conversation into a single string
    inputs = ' '.join(inputs)

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    parameters_dict = {
        "max_new_tokens": 100,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1,
        "return_full_text": False,
    }
    model_loaded = False
    while not model_loaded:
        try:
            answer = query({
                "inputs": inputs,
                "parameters": parameters_dict
            })
            if "error" in answer:
                logging.error(f"Error: {answer}")
                time.sleep(5)
            else:
                model_loaded = True
                logging.debug(f"Answer: {answer[0]['generated_text']}")
                return answer
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            time.sleep(5)


def run_gpt_query(conversation):
    # Strip the question of any trailing whitespace and any '/n' characters
    for question in conversation:
        question['text'].strip()
        question['text'].replace('\n', '')

    # Trim conversation to last 2000 tokens
    trimmed_conversation = trim_conversation(conversation, max_tokens=2000)
    messages = [{"role": "system",
                 "content": "You are a factual chatbot that is helpful and an expert in the Ubuntu Operating system."}]
    for message in trimmed_conversation:
        if message['user'] == 'Anon':
            role = 'user'
        else:
            role = 'assistant'

        messages.append({"role": role,
                         "content": message['text']})

    completion = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0613:mentors-more::8AMjJzqE",
        messages=messages,
        temperature=0.5,
        max_tokens=256,
        top_p=1,
        frequency_penalty=1.05,
        presence_penalty=0.5,
        stop=["<|endoftext|>", "User:"]
    )

    return {"message": completion.choices[0]['message']['content']}


ALLOWED_ORIGINS = ['http://localhost:8080', 'http://localhost:9000', 'https://emotiondetection.app',
                   'https://jtdesigns.app']


@functions_framework.http
def ubuntu_chat(request):
    origin = request.headers.get('Origin')
    if origin not in ALLOWED_ORIGINS:
        return 'Unauthorized', 403
    print('Origin okay')

    if request.method == 'OPTIONS':
        response = make_response('', 204)
        response.headers.set('Access-Control-Allow-Origin', origin)
        response.headers.set('Access-Control-Allow-Methods', 'GET, POST')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
        return response
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)

    if request_json and 'conversation' in request_json and 'model' in request_json:
        conversation = request_json['conversation']

        try:
            if request_json['model'] == 'medium':
                # Run the query and gather the output
                output = run_dialo_query(conversation, size='medium')
                logging.debug(f'output: {output}')
            elif request_json['model'] == 'large':
                # Run the query and gather the output
                output = run_dialo_query(conversation, size='large')
            else:
                # Run the query and gather the output
                output = run_gpt_query(conversation)

            response = make_response(output, 200)
            response.headers.set('Content-Type', 'application/json')
        except Exception as e:
            logging.exception("Caught exception while running query.")
            response = make_response(jsonify({"error": str(e)}), 500)
            response.headers.set('Content-Type', 'application/json')

        # Set CORS headers
        response.headers.set('Access-Control-Allow-Origin', origin)
        response.headers.set('Access-Control-Allow-Methods', 'GET, POST')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type')

        return response
    else:
        return jsonify({
            'status': 'error',
            'message': 'Please provide a question in the JSON payload with the key "question".'
        }), 400
