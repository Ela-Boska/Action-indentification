# input 10 x 30 x 3 = 900
import torch
from torch import nn
from time import time
p_dropout = 0
class WIFI(nn.Module):
    def __init__(self):
        super(WIFI, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3,3], stride=[1,1],padding=[1,1])   # 10x30x16
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[4,4], stride=[2,2],padding=[1,1])  # 5x15x32
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[5,15], stride=[1,1],padding=[0,0]) # 1x1x64
        self.batch3 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(64,8)                                                                          # 8 
    
    def forward(self, input:torch.Tensor):
        # (N,10,30,3) -> (N,720)
        output = input.transpose(2,3)
        output = output.transpose(1,2)
        output = self.conv1(output)
        output = self.batch1(output)
        output = nn.functional.dropout2d(output, p_dropout, self.training)
        output = nn.functional.relu(output)
        #print(output.shape)
        output = self.conv2(output)
        output = self.batch2(output)
        output = nn.functional.dropout(output, p_dropout, self.training)
        output = nn.functional.relu(output)
        #print(output.shape)
        output = self.conv3(output)
        output = self.batch3(output)
        output = nn.functional.dropout2d(output, p_dropout, self.training)
        output = nn.functional.relu(output)
        #print(output.shape)
        output = self.linear1(output.view(-1,64))
        #print(output.shape)
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
    
    loss_total = 0
    count = 0
    iter_total = max_epoch*90
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