import openai

# get the file id from File create
openai.FineTuningJob.create(
    training_file="file-uc3RROCzcL8T9WdukQM22YuE",
    validation_file="file-5r4JTIIvydxeEz8PNeRQFaDE",
    model="gpt-3.5-turbo-0613",
    hyperparameters={
        "n_epochs": 3,
    }
)