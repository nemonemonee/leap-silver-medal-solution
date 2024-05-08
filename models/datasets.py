from torch.utils.data import Dataset

class LEAPDataset(Dataset):
    def __init__(self, src, label):
        self.src = src
        self.label = label

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return self.src[idx], self.label[idx]
    
class LEAPTestDataset(Dataset):
    def __init__(self, src):
        self.src = src
        
    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return self.src[idx]

class MixDataset(Dataset):
    def __init__(self, src, slr, label):
        self.src = src
        self.slr = slr
        self.label = label

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return self.src[idx], self.slr[idx], self.label[idx]
    
class MixTestDataset(Dataset):
    def __init__(self, src, slr):
        self.src = src
        self.slr = slr
        
    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return self.src[idx], self.slr[idx]