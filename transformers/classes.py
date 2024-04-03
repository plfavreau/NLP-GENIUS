from torch.utils.data import Dataset
import torch

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