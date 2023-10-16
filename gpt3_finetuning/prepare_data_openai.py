import pandas as pd
import json
from collections import defaultdict

data_folder = 'data/ubuntu/'

# Load the data
train_df = pd.read_csv(data_folder + 'train.csv')

# Keep only examples with label 1
train_df = train_df[train_df['Label'] == 1]

# Random subset the data to 1/100 size to keep cost under $20
train_df = train_df.sample(frac=1/100, random_state=42)
train_df.reset_index(drop=True, inplace=True)


def convert_to_chat_format(df):
    chat_list = []
    num_examples = 0
    for i in range(len(df)):
        chat = {'messages': []}
        full_chat = df['Context'][i] + ' ' + df['Utterance'][i]
        split_chat = full_chat.split('__eot__')
        # remove special characters of '__eot__' and '__eou__'
        split_chat = [s.replace('__eot__', '').replace('__eou__', '') for s in split_chat]
        # remove white spaces
        split_chat = [s.strip() for s in split_chat]
        # add two turns from each conversation, one from user and one from assistant, only if the user turn is a real question, longer than 15 characters
        if split_chat[0] != '' and len(split_chat[0]) > 25:
            num_examples += 1
            chat['messages'].append({'role': 'system',
                                     'content': 'You are a factual chatbot that is helpful and an expert in the Ubuntu Operating system.'
                                     })
            chat['messages'].append({'role': 'user', 'content': split_chat[0]})
            chat['messages'].append({'role': 'assistant', 'content': split_chat[1]})

            chat_list.append(chat)
    return chat_list


conv_list = convert_to_chat_format(train_df)

# Split into train and test sets
train_size = int(len(conv_list) * 0.9)
train_conv_list = conv_list[:train_size]
test_conv_list = conv_list[train_size:]

# convert to jsonl format called dataset
data_path_train = data_folder + 'openai-finetune-ready-data-train.jsonl'
with open(data_path_train, 'w') as f:
    for item in train_conv_list:
        f.write(json.dumps(item) + '\n')

data_path_test = data_folder + 'openai-finetune-ready-data-test.jsonl'
with open(data_path_test, 'w') as f:
    for item in test_conv_list:
        f.write(json.dumps(item) + '\n')

# check the data by combining the train and test sets
with open(data_path_train, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

with open(data_path_test, 'r', encoding='utf-8') as f:
    dataset.extend([json.loads(line) for line in f])

# look for errors in the data, according to https://cookbook.openai.com/examples/chat_finetuning_data_prep

# Format error checks
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name", "function_call") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1

        content = message.get("content", None)
        function_call = message.get("function_call", None)

        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found, data ready for fine-tuning!")