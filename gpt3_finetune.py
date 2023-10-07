import openai
import torch

# Fine-tune the model on the ubuntu dataset

# Load the data
data_folder = 'data/ubuntu/'
train_encodings = torch.load(data_folder + 'train_encodings.pt')
train_targets = torch.load(data_folder + 'train_targets.pt')
val_encodings = torch.load(data_folder + 'val_encodings.pt')
val_targets = torch.load(data_folder + 'val_targets.pt')
test_encodings = torch.load(data_folder + 'test_encodings.pt')
test_targets = torch.load(data_folder + 'test_targets.pt')


train_loader, val_loader, test_loader = load_data(train_encodings, train_targets, val_encodings, val_targets)