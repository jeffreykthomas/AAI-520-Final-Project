# AAI-520-Final-Project
## Team Members:
- [Shyam Adhikari]()
- [Matthew Sexton](https://github.com/mattwsexton)
- [Jeffrey Thomas](https://github.com/jeffreykthomas/)

## Project Description:
This project aims to build a chatbot using the Ubuntu dataset.

## Project Structure:
```
├── data
│   ├── ubuntu
|   |   ├── train.csv
|   |   ├── test.csv
|   |   ├── valid.csv
|-- gpt2_finetuning
|   ├── gpt2_generator_train.py
|-- gpt3_finetuning
|   ├── gpt3_create_finetune.py
|   ├── gpt3_upload_data.py
|   ├── prepare_data_openai.py
|-- README.md
```

## Data:
The data used for this project is the Ubuntu Dialogue Corpus v2.0. This dataset contains almost 1 million two-person conversations extracted from chat logs from the Ubuntu channels on the Freenode IRC network. The dataset is available on [github](https://github.com/rkadlec/ubuntu-ranking-dataset-creator).

## GPT-2 Finetuning:
The GPT-2 model was finetuned using the [Huggingface](https://huggingface.co/microsoft/DialoGPT-medium?text=Hey+my+name+is+Thomas%21+How+are+you%3F) library. The model was trained for 3 epochs. The training script can be found in the gpt2_finetuning folder.
We ended up with Dialo from Microsoft, an already fine-tuned gpt2 for dialogue generation. 

We trained both the medium and large sizes of the model.

## GPT-3 Finetuning:
The GPT-3 model was finetuned using the [OpenAI](https://beta.openai.com/) API. The model was trained for a single epoch. The training script can be found in the gpt3_finetuning folder.

## Results:
To be determined...