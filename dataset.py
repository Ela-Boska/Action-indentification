import torch
from torch.utils.data import Dataset
from utils import loadcsv, aggregate, to_DFS

class my_dataset(Dataset):
    def __init__(self, length=5, train=True):
        super(my_dataset, self).__init__()
        self.database = []
        self.length = length
        self.train = train
        if self.train:
            self.size = (60-self.length)*2
        else:
            self.size = (15-self.length)*2
        s = ['j','k','s','y']
        for s1 in range(4):
            for s2 in range(4):
                data = loadcsv('szy_double/%s%s.csv' % (s[s1],s[s2]), [2,3,8,13,17,18], begining=10, ending=85)
                _,_,data = to_DFS(data, 100, 100, cut=[-6,6])
                data = aggregate(data,[2,3,8,13,17,18])
                data = torch.tensor(data, dtype=torch.float32)
                self.database.append((data,s1*10+s2))
    
    def __getitem__(self,id:int):
        if self.train:
            return (self.database[id//self.size][0][id%self.size:id%self.size+self.length*2],self.database[id//self.size][1])
        else:
            return (self.database[id//self.size][0][id%self.size+120:id%self.size+self.length*2+120],self.database[id//self.size][1])

    def __len__(self):
        if self.train:
            return 16*(60-self.length)*2
        else:
            return 16*(15-self.length)*2
    
class dataset2(Dataset):
    def __init__(self, length=5, train=True):
        super(dataset2, self).__init__()
        self.database = []
        self.length = length
        self.train = train
        if self.train:
            self.size = (60-self.length)*2
        else:
            self.size = (15-self.length)*2
        s = ['j','k','s','y']
        for s1 in range(4):
            for s2 in range(4):
                data = loadcsv('szy_double/%s%s.csv' % (s[s1],s[s2]), [4,9,14], begining=10, ending=85)
                _,_,data = to_DFS(data, 100, 100, cut=[-15,15])
                data = aggregate(data,[4,9,14])
                data = torch.tensor(data, dtype=torch.float32)
                self.database.append((data,s1*10+s2))
    
    def __getitem__(self,id:int):
        if self.train:
            return (self.database[id//self.size][0][id%self.size:id%self.size+self.length*2],self.database[id//self.size][1])
        else:
            return (self.database[id//self.size][0][id%self.size+120:id%self.size+self.length*2+120],self.database[id//self.size][1])

    def __len__(self):
        if self.train:
            return 16*(60-self.length)*2
        else:
            return 16*(15-self.length)*2