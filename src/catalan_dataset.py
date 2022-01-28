from torch.utils.data import Dataset
import pandas as pd

class CatalanDataset(Dataset):
    def __process_item__(self, item):
        return self.tokenizer(str(item), 
                              return_tensors='pt', 
                              truncation=True, 
                              padding='max_length',
                              max_length = 1024)

    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        
        df = pd.read_csv(path, sep = ';')
        self.items = df.text
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.__process_item__(self.items[i])