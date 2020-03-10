# input 10 x 30 x 3 = 360
import torch
from torch import nn
from time import time
p_dropout = 0
class WIFI(nn.Module):
    def __init__(self):
        super(WIFI, self).__init__()
        self.linear1 = nn.Linear(900,1024)
        self.batch1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024,1024)
        self.batch2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(1024,8)
    
    def forward(self, input):
        # (N,10,39,3) -> (N,360)
        output = input.view(input.shape[0],-1)
        output = self.linear1(output)
        output = self.batch1(output)
        output = nn.functional.dropout(output, p_dropout, self.training)
        output = nn.functional.relu(output)
        output = self.linear2(output)
        output = self.batch2(output)
        output = nn.functional.dropout(output, p_dropout, self.training)
        output = nn.functional.relu(output)
        output = self.linear3(output)
        return output

def train(input:torch.Tensor, target:torch.Tensor, model, optimizer, criterion):
    # input: (batch,3)
    # target: (batch)
    optimizer.zero_grad()
    output:torch.Tensor = model(input)
    loss = criterion(output[:,:4], target//10) + criterion(output[:,4:], target%10)
    loss.backward()
    optimizer.step()
    return loss.item()

def TRAIN(model, dataloader, optimizer, writer, max_epoch):
    model.train()
    plot_loss = []
    count = 0
    t1 = time()
    for epoch in range(max_epoch):
        loss_total = 0
        for input,target in dataloader:
            count += 1
            target = target.view(-1)
            loss = train(input, target, model, optimizer, nn.CrossEntropyLoss())
            plot_loss.append(loss)
            writer.add_scalar('loss',loss,count)
            loss_total += loss
        print('loss={}'.format(loss_total))
    print("finished")
    model.eval()

def evaluate(model:nn.Module, dataloader):
    count1 = 0
    count2 = 0
    count3 = 0
    length = 0
    for input,target in dataloader:
        length += len(input)
        output = model(input)
        count1 += torch.sum(output[:,:4].argmax(1)==target//10).item()
        count2 += torch.sum(output[:,4:].argmax(1)==target%10).item()
        count3 += torch.sum((output[:,:4].argmax(1)==target//10)*(output[:,4:].argmax(1)==target%10)).item()
    return count1/length, count2/length, count3/length