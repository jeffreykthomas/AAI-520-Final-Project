import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
from torch.cuda.amp import autocast, GradScaler
import wandb
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

data_folder = 'data/ubuntu/'

# Load the tokenized train data
train_encodings = torch.load(data_folder + 'train_encodings.pt')
train_targets = torch.load(data_folder + 'train_targets.pt')

# Load the tokenized validation and test data
val_encodings = torch.load(data_folder + 'val_encodings.pt')
val_targets = torch.load(data_folder + 'val_targets.pt')

test_encodings = torch.load(data_folder + 'test_encodings.pt')
test_targets = torch.load(data_folder + 'test_targets.pt')


class CustomDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __getitem__(self, idx):
        input_item = {key: self.input_encodings[key][idx] for key in self.input_encodings}
        target_item = {key: self.target_encodings[key][idx] for key in self.target_encodings}
        return input_item, target_item

    def __len__(self):
        return len(self.input_encodings['input_ids'])


def load_data(train_X, train_y, val_X, val_y, test_X, test_y):
    train_dataset = CustomDataset(train_X, train_y)
    val_dataset = CustomDataset(val_X, val_y)
    test_dataset = CustomDataset(test_X, test_y)

    # Create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader


def remove_pad_tokens(decoded_text, pad_token):
    return decoded_text.replace(pad_token, '')


def evaluate_bleu(model, val_loader, tokenizer, batch_idx, loss, best_bleu_score, device):
    model.eval()
    bleu_scores = []
    num_samples = 20
    random_indices = np.random.randint(0, len(val_loader), num_samples)
    sample_idx = np.random.choice(random_indices)
    num_samples_evaluated = 0
    print('Length of tokenizer:', len(tokenizer))
    print('Vocab size:', model.module.config.vocab_size)

    # Select random num_samples from the validation set
    with torch.no_grad():
        for val_idx, (val_item, val_target) in enumerate(val_loader):
            # Prepare input and target
            if val_idx in random_indices:
                num_samples_evaluated += 1
                print(f'\rGenerating responses for validation sample number {num_samples_evaluated}/{num_samples}', end='\n', flush=True)
                contexts = val_item['input_ids'].to(device)
                attention_masks = val_item['attention_mask'].to(device)
                ground_truth = val_target['input_ids'].to(device)

                generation_config = GenerationConfig(max_length=612,
                                                     min_new_tokens=32,
                                                     num_return_sequences=1,
                                                     repetition_penalty=1.1,
                                                     do_sample=True,
                                                     pad_token_id=tokenizer.eos_token_id)

                generated_responses = model.module.generate(contexts,
                                                            attention_mask=attention_masks,
                                                            **generation_config.to_dict())
                if val_idx == sample_idx:
                    # print one untruncated generated response
                    print(f'Generated response: {tokenizer.decode(generated_responses[0], skip_special_tokens=False)}')

                # Remove context tokens from generated responses. First, remove the context tokens for each response
                # from the start of the generated responses
                truncated_generated_responses = [generated_response[len(context):] for generated_response, context in zip(generated_responses, contexts)]
                decoded_generated_responses = [tokenizer.decode(generated_response, skip_special_tokens=False) for generated_response in truncated_generated_responses]
                # Remove the padding tokens from the end of the generated responses
                decoded_generated_responses = [remove_pad_tokens(decoded_generated_response, tokenizer.pad_token) for decoded_generated_response in decoded_generated_responses]

                decoded_truth = [tokenizer.decode(t, skip_special_tokens=False) for t in ground_truth]
                # Remove the padding tokens from the end of the ground truth
                decoded_truth = [remove_pad_tokens(decoded_truth_item, tokenizer.pad_token) for decoded_truth_item in decoded_truth]

                # print 1 sample of generated responses and ground truth from the random_indices
                if val_idx == sample_idx:
                    print(f'Context: {remove_pad_tokens(tokenizer.decode(contexts[0], skip_special_tokens=False), tokenizer.pad_token)}')
                    print(f'Generated response: {decoded_generated_responses[0]}')
                    print(f'Ground truth: {decoded_truth[0]}')
                # Calculate BLEU score
                for gen_response, truth in zip(decoded_generated_responses, decoded_truth):
                    bleu_score = sentence_bleu([truth], gen_response)
                    bleu_scores.append(bleu_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    wandb.log({'iteration': batch_idx + 1, 'loss': loss.item(), 'blue_score': avg_bleu_score})
    if avg_bleu_score > best_bleu_score:
        best_bleu_score = avg_bleu_score

    print(f'\rCurrent BLEU score: {avg_bleu_score}, Best BLEU score: {best_bleu_score}')
    return avg_bleu_score, best_bleu_score


def run_training():
    # Use wandb to track training
    wandb_project = 'aai-520-final-project'
    wandb_run_name = 'gpt2-medium-ubuntu-generation-1'

    wandb.init(project=wandb_project, name=wandb_run_name)

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(data_folder)
    print('Length of tokenizer:', len(tokenizer))

    # Load the model
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last n layers
    n = 1
    for i, param in enumerate(model.transformer.h[-n:].parameters()):
        param.requires_grad = True

    # Initialize the scaler
    scaler = GradScaler()

    # Load the data
    train_loader, val_loader, test_loader = load_data(train_encodings, train_targets, val_encodings,
                                                      val_targets, test_encodings,
                                                      test_targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.02, betas=(0.9, 0.95), eps=1e-6)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_bleu_score = 0

    for epoch in range(3):
        model.train()
        for batch_idx, (input_item, target_item) in enumerate(train_loader):
            # Move tensors to the device
            input_ids = input_item['input_ids'].to(device)
            attention_mask = input_item.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            target_ids = target_item['input_ids'].to(device)

            with autocast():
                # Forward pass
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=target_ids
                                )

                loss = outputs.loss.mean()

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            print(f'\rEpoch {epoch + 1}, batch: {batch_idx + 1}/{len(train_loader)}, loss: {loss.item()}', end='',
                  flush=True)

            if (batch_idx + 1) % 1000 == 0:
                # Validation loop
                avg_bleu_score, new_best_bleu_score = evaluate_bleu(model,
                                                                    val_loader,
                                                                    tokenizer,
                                                                    batch_idx,
                                                                    loss,
                                                                    best_bleu_score,
                                                                    device)

                if new_best_bleu_score > best_bleu_score:
                    best_bleu_score = new_best_bleu_score
                    torch.save(model.state_dict(), data_folder + 'gpt2_model.pt')
                model.train()


if __name__ == '__main__':
    run_training()
