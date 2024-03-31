import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from preprocessing import import_data_transformer
import random

class LyricsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.lyrics = dataframe['lyrics']
        self.titles = dataframe['title'] 
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        title = self.titles[idx]
        
        # Concatenate title and lyrics
        text = title + ' ' + lyric
        
        # Tokenize and encode text
        encoding = self.tokenizer(text, 
                                  truncation=True, 
                                  padding='max_length', 
                                  max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask'])
        }

def train(model, dataloader, optimizer, device, epoch):
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

def generate_lyrics(model, tokenizer, title, max_length, num_return_sequences):
    input_ids = tokenizer.encode(title, return_tensors='pt').to(model.device)
    output = model.generate(
        input_ids, 
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    lyrics = tokenizer.decode(output[0], skip_special_tokens=True)
    return lyrics

def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_model(model_path, device):
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def tf_entry_point(isTraining, train_set, validation_set):
    # Reset the index of the DataFrames
    train_set = train_set.reset_index(drop=True)
    validation_set = validation_set.reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    
    # Create datasets and dataloaders
    train_dataset = LyricsDataset(train_set, tokenizer, max_length=768)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    if isTraining:
        epochs = 1
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train(model, train_dataloader, optimizer, device, epoch)
        
        # Save the trained model
        save_model(model, tokenizer, "trained_model")
    else:
        # Load the trained model
        model, tokenizer = load_model("trained_model", device)
        
        # Generate lyrics for a random title
        random_title = random.choice(validation_set['title'].tolist())
        print(f"Generating lyrics for the title: '{random_title}'")
        generated_lyrics = generate_lyrics(model, tokenizer, random_title, max_length=500, num_return_sequences=1)
        print("Generated Lyrics:")
        print(generated_lyrics)
        
if __name__ == "__main__":
    file_path = "song_lyrics.csv"  
    train_set, validation_set = import_data_transformer(file_path, 300)
    isTraining = False
    tf_entry_point(isTraining, train_set, validation_set)
