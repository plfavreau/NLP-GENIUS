import os

import torch
from classes import LyricsDataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from training import train

from preprocessing import import_data_transformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_lyrics_from_title(title, device, max_length, num_variations):
    model, tokenizer = load_model("trained_model", device)
    generated_lyrics = generate_lyrics(model, tokenizer, title, max_length, num_variations)
    return generated_lyrics

def generate_lyrics(model, tokenizer, title, max_length, num_variations):
    input_ids = tokenizer.encode(title, return_tensors='pt').to(model.device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_variations,
        no_repeat_ngram_size=2,
        early_stopping=False,
        num_beams=5,
        do_sample=True,  # Enable sampling
        top_k=50,  # Sample from top 50 tokens
        top_p=0.95,  # Sample from top 95% probability mass
        temperature=0.7  # Adjust temperature for sampling
    )
    
    generated_lyrics = []
    for sequence in output:
        generated_lyrics.append(tokenizer.decode(sequence, skip_special_tokens=True))
    return generated_lyrics

def getPerplexityFromPretrained():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model("transformers/trained_model", device)
    VALIDATION_ROWS = 200000
    _ , validation_set = import_data_transformer("dataset/song_lyrics.csv", VALIDATION_ROWS * 5) # validation set is 20% of the data
    print(f"Validation set size: {len(validation_set)}")
    # reset_index(drop=True) is used to reset the index of the DataFrame
    validation_set = validation_set.reset_index(drop=True)
    validation_dataset = LyricsDataset(validation_set, tokenizer, max_length=768)
    print(f"Validation dataset size: {len(validation_dataset)}")
    validation_dataloader = DataLoader(validation_dataset, batch_size=4)
    print(f"Validation dataloader size: {len(validation_dataloader)}")
    perplexity = calculate_perplexity(model, validation_dataloader, device)
    return f"Perplexity on validation set: {perplexity:.2f}"

def calculate_perplexity(model, dataloader, device):
    model.eval()
    perplexity = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs[0]
            perplexity += torch.exp(loss).item()
    perplexity /= len(dataloader)
    return perplexity

def post_process_lyrics(lyrics):
    post_processed_lyrics = []
    for line in lyrics:
        post_processed_lyrics += [x if x in ['\n', ' '] else x.capitalize() for x in line.split('\n')]
    return post_processed_lyrics

def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_model(model_path, device):
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def tf_entry_point(isTraining, file_path, nbRowsForTraining, device):
    
    if isTraining:
        train_set, validation_set = import_data_transformer(file_path, nbRowsForTraining)

        # Reset the index of the DataFrames
        train_set = train_set.reset_index(drop=True)
        validation_set = validation_set.reset_index(drop=True)
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        
        # Create datasets and dataloaders
        train_dataset = LyricsDataset(train_set, tokenizer, max_length=768)
        validation_dataset = LyricsDataset(validation_set, tokenizer, max_length=768) 
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=4)
        
        # Set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Adjusts lr every epoch
    
        epochs = 3
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train(model, train_dataloader, optimizer, device, epoch)
            perplexity = calculate_perplexity(model, validation_dataloader, device)
            scheduler.step()
            print(f"Perplexity on validation set for epoch {epoch} : {perplexity:.2f}")
        
        # Save the trained model
        save_model(model, tokenizer, "trained_model")
    else:
        # Load the trained model
        model, tokenizer = load_model("trained_model", device)

    perplexity = calculate_perplexity(model, validation_dataloader, device)
    print(f"Perplexity on validation set: {perplexity:.2f}")    

def prettyPrintLyrics(lyrics):
    for e in lyrics:
        print(e.capitalize())
    
        
if __name__ == "__main__":
    # count the words sum of all the columns in the first 1 million rows :
    # cat song_lyrics.csv | head -n 1000000 | awk -F ',' '{print $1}' | wc -w

    exit()
    isTraining = os.environ.get('TSF_TRAINING', 'False').lower() == 'true'
    title = os.environ.get('TSF_TITLE', 'Life is about raining.')
    rows = os.environ.get('TSF_ROWS', '100')

    print(f"Training enabled: {isTraining}")
    if isTraining:
        print(f"Training model with {rows} rows")
    else:
        print(f"Generating lyrics for the title '{title}'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = "song_lyrics.csv"
    
    if isTraining:
        tf_entry_point(isTraining, file_path, rows, device)
    else:
        # lyrics = generate_lyrics_from_title(title, device, max_length=1024, num_variations=1)
        print(getPerplexityFromPretrained())
        # print(f"Generated Lyrics for the title '{title}':")
        # post_processed_lyrics = post_process_lyrics(lyrics)
        # prettyPrintLyrics(post_processed_lyrics)
        
