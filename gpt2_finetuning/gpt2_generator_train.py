import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import wandb
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torch.nn.functional as F

data_folder = 'data/ubuntu/'

# Load the tokenized train data
train_encodings = torch.load(data_folder + 'train_encodings_large.pt')

# Load the tokenized validation and test data
val_encodings = torch.load(data_folder + 'val_encodings_large.pt')

test_encodings = torch.load(data_folder + 'test_encodings_large.pt')

# Add args for model size, epochs, etc.
model_size = 'large'
num_epochs = 3


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
        # token_type_ids = torch.stack(token_type_ids) if token_type_ids else None
        # labels = torch.stack(labels)

        x_encodings = {'input_ids': input_ids,
                       'attention_mask': attn_masks}

        return x_encodings


def load_data(train_X, val_X, test_X, pad_id):
    ppd = PadCollate(pad_id)
    train_dataset = CustomDataset(train_X)
    val_dataset = CustomDataset(val_X)
    test_dataset = CustomDataset(test_X)

    # Create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=ppd.pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=ppd.pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=ppd.pad_collate)

    return train_loader, val_loader, test_loader


def remove_pad_tokens(decoded_text, pad_token):
    return decoded_text.replace(pad_token, '')


def evaluate_bleu(model, val_loader, tokenizer, device):
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
        for val_idx, batch in enumerate(val_loader):
            # Prepare input and target
            if val_idx in random_indices:
                num_samples_evaluated += 1
                print(f'\rGenerating responses for validation sample number {num_samples_evaluated}/{num_samples}',
                      end='\n', flush=True)
                contexts = batch['input_ids'].to(device)
                attention_masks = batch['attention_mask'].to(device)
                ground_truth = batch['input_ids'].to(device)

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
                truncated_generated_responses = [generated_response[len(context):] for generated_response, context in
                                                 zip(generated_responses, contexts)]
                decoded_generated_responses = [tokenizer.decode(generated_response, skip_special_tokens=False) for
                                               generated_response in truncated_generated_responses]
                # Remove the padding tokens from the end of the generated responses
                decoded_generated_responses = [remove_pad_tokens(decoded_generated_response, tokenizer.pad_token) for
                                               decoded_generated_response in decoded_generated_responses]

                # Remove the input tokens from the ground truth and decode
                ground_truth = [ground_truth_item[len(context):] for ground_truth_item, context in
                                zip(ground_truth, contexts)]
                decoded_truth = [tokenizer.decode(t, skip_special_tokens=False) for t in ground_truth]
                # Remove the padding tokens from the end of the ground truth
                decoded_truth = [remove_pad_tokens(decoded_truth_item, tokenizer.pad_token) for decoded_truth_item in
                                 decoded_truth]

                # print 1 sample of generated responses and ground truth from the random_indices
                if val_idx == sample_idx:
                    print(
                        f'Context: {remove_pad_tokens(tokenizer.decode(contexts[0], skip_special_tokens=False), tokenizer.pad_token)}')
                    print(f'Generated response: {decoded_generated_responses[0]}')
                    print(f'Ground truth: {decoded_truth[0]}')
                # Calculate BLEU score
                for gen_response, truth in zip(decoded_generated_responses, decoded_truth):
                    bleu_score = sentence_bleu([truth], gen_response)
                    bleu_scores.append(bleu_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    print(f'\rCurrent BLEU score: {avg_bleu_score}')


def run_training():
    # Use wandb to track training
    wandb_project = 'aai-520-final-project'
    wandb_run_name = 'dialo-large-ubuntu-generation'

    wandb.init(project=wandb_project, name=wandb_run_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_folder + 'tokenizer_large')
    tokenizer.pad_token = tokenizer.eos_token
    print('Length of tokenizer:', len(tokenizer))

    # Load the model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
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
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    num_train_steps = len(train_loader) // accumulation_steps * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

                print(f'\rEpoch {epoch + 1}, batch: {batch_idx + 1}/{len(train_loader)}, scaled_loss: {loss.item()}, effective_loss: {loss.item() * accumulation_steps}', end='',
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
                    {'global_step': global_step, 'loss': loss.item() * accumulation_steps, 'val_loss': combined_val_loss, 'lr': scheduler.get_last_lr()[0]})
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

    model.eval()
    print('Evaluating BLEU score on test set...')
    evaluate_bleu(model,
                  test_loader,
                  tokenizer,
                  device)


if __name__ == '__main__':
    run_training()
