import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, get_linear_schedule_with_warmup, \
    BitsAndBytesConfig
from torch.cuda.amp import autocast, GradScaler
import wandb
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torch.nn.functional as F

data_folder = 'data/ubuntu/'

# Add args for model size, epochs, etc.
parser = argparse.ArgumentParser(description='Train a GPT2 model on the Ubuntu Dialog Corpus')
parser.add_argument('--model_size',
                    type=str,
                    default='medium',
                    help='Size of the model to train, "medium" or "large"',
                    choices=['medium', 'large'])
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')

model_size = parser.parse_args().model_size
num_epochs = parser.parse_args().num_epochs

model_name = 'microsoft/DialoGPT-' + model_size
tokenizer_name = 'tokenizer_' + model_size
# Load the tokenized train data
if model_size == 'medium':
    train_encodings = torch.load(data_folder + 'train_encodings.pt')
    val_encodings = torch.load(data_folder + 'val_encodings.pt')
    test_encodings = torch.load(data_folder + 'test_encodings.pt')
else:
    train_encodings = torch.load(data_folder + 'train_encodings_large.pt')
    val_encodings = torch.load(data_folder + 'val_encodings_large.pt')
    test_encodings = torch.load(data_folder + 'test_encodings_large.pt')


batch_size = parser.parse_args().batch_size

# Some detail inspired by
# https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, input_encodings):
        self.input_encodings = input_encodings

    def __getitem__(self, idx):
        input_item = {key: self.input_encodings[key][idx] for key in self.input_encodings}
        return input_item

    def __len__(self):
        return len(self.input_encodings['input_ids'])


class PadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
        input_ids, attn_masks = [], []
        max_len = max(len(seqs['input_ids']) for seqs in batch)  # Find the maximum sequence length in this batch

        for idx, seqs in enumerate(batch):
            pad_len = max_len - len(seqs['input_ids'])  # Calculate how much padding is needed

            input_ids.append(F.pad(torch.LongTensor(seqs['input_ids'].long()), (pad_len, 0), value=self.pad_id))
            attn_masks.append(
                F.pad(torch.LongTensor(seqs['attention_mask'].long()), (pad_len, 0), value=0))

        # Stack the tensors along a new dimension
        input_ids = torch.stack(input_ids)
        attn_masks = torch.stack(attn_masks)

        x_encodings = {'input_ids': input_ids,
                       'attention_mask': attn_masks}

        return x_encodings


def load_data(train_X, val_X, test_X, pad_id):
    ppd = PadCollate(pad_id)
    train_dataset = CustomDataset(train_X)
    val_dataset = CustomDataset(val_X)
    test_dataset = CustomDataset(test_X)

    # Create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ppd.pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ppd.pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=ppd.pad_collate)

    return train_loader, val_loader, test_loader


def remove_pad_tokens(decoded_text, pad_token):
    return decoded_text.replace(pad_token, '')


def run_training():
    # Use wandb to track training
    wandb_project = 'aai-520-final-project'
    wandb_run_name = 'dialo-' + model_size + '-ubuntu-generation'

    wandb.init(project=wandb_project, name=wandb_run_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_folder + tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print('Length of tokenizer:', len(tokenizer))

    # Load the model

    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # Initialize the scaler
    scaler = GradScaler()

    # Load the data
    pad_id = tokenizer.pad_token_id
    train_loader, val_loader, test_loader = load_data(train_encodings, val_encodings,
                                                      test_encodings,
                                                      pad_id)
    no_decay = ["bias", "LayerNorm.weight"]
    num_train_epochs = 3
    accumulation_steps = 4

    num_train_steps = len(train_loader) // accumulation_steps * num_train_epochs

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_train_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            target_ids = batch['input_ids'].to(device)

            with autocast():
                # Forward pass
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=target_ids
                                )
                loss = outputs.loss.mean()

            # Normalize the loss
            loss = loss / accumulation_steps

            # Backward pass and optimization
            scaler.scale(loss).backward()

            # Update only after accumulating gradients for n steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Update step count
                global_step += 1

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Perform optimization step
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients
                optimizer.zero_grad()

                # Update learning rate schedule
                scheduler.step()

                print(
                    f'\rEpoch {epoch + 1}, batch: {batch_idx + 1}/{len(train_loader)}, scaled_loss: {loss.item()}, '
                    f'effective_loss: {loss.item() * accumulation_steps}',
                    end='',
                    flush=True)

            # Validation loop
            if batch_idx % 1000 == 0:
                max_val_batches = 64
                # choose random indices to evaluate on
                random_indices = np.random.randint(0, len(val_loader), max_val_batches)
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_idx, batch in enumerate(val_loader):
                        if val_idx not in random_indices:
                            continue
                        # Move tensors to the device
                        val_input_ids = batch['input_ids'].to(device)
                        val_attention_mask = batch.get('attention_mask', None)
                        if val_attention_mask is not None:
                            val_attention_mask = val_attention_mask.to(device)
                        val_target_ids = batch['input_ids'].to(device)

                        with autocast():
                            # Forward pass
                            outputs = model(input_ids=val_input_ids,
                                            attention_mask=val_attention_mask,
                                            labels=val_target_ids
                                            )

                            val_loss = outputs.loss.mean()
                            val_losses.append(val_loss.item())

                            # print out a sample of the validation set
                            if val_idx == random_indices[0]:
                                print(f'\nContext: {tokenizer.decode(val_input_ids[0], skip_special_tokens=False)}')
                                print(
                                    f'Generated output: {tokenizer.decode(outputs.logits[0].argmax(dim=-1), skip_special_tokens=False)}')
                                labels = val_target_ids.where(val_target_ids != -100, tokenizer.pad_token_id)
                                print(f'Ground truth: {tokenizer.decode(labels[0], skip_special_tokens=False)}')

                combined_val_loss = sum(val_losses) / len(val_losses)
                wandb.log(
                    {'global_step': global_step, 'loss': loss.item() * accumulation_steps,
                     'val_loss': combined_val_loss, 'lr': scheduler.get_last_lr()[0]})
                print(
                    f'\nEpoch {epoch + 1}, batch: {batch_idx + 1}/{len(train_loader)}, loss: {loss.item() * accumulation_steps}, val_loss: {combined_val_loss}')

                if combined_val_loss < best_val_loss:
                    best_val_loss = combined_val_loss
                    # Save the model to HuggingFace
                    if best_val_loss < 1.85:
                        model.module.save_pretrained(wandb_run_name)
                        tokenizer.save_pretrained(wandb_run_name)
                        # Push to Hub
                        model.module.push_to_hub(f'jeffreykthomas/{wandb_run_name}')
                        tokenizer.push_to_hub(f'jeffreykthomas/{wandb_run_name}')
                model.train()


if __name__ == '__main__':
    run_training()
