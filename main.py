# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import Config
from model import DPCNN
from data import TextDataset
import argparse

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out_channel', type=int, default=2)
parser.add_argument('--label_num', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

# Create the configuration
config = Config(sentence_max_size=50,
                batch_size=args.batch_size,
                word_num=11000,
                label_num=args.label_num,
                learning_rate=args.lr,
                cuda=args.gpu,
                epoch=args.epoch,
                out_channel=args.out_channel)

training_set = TextDataset(path='data/train')

training_iter = data.DataLoader(dataset=training_set,
                                batch_size=config.batch_size,
                                num_workers=2)


model = DPCNN(config)
embeds = nn.Embedding(config.word_num, config.word_embedding_dimension)

if torch.cuda.is_available():
    model.cuda()
    embeds = embeds.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr)

count = 0
loss_sum = 0
# Train the model
for epoch in range(config.epoch):
    for data, label in training_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            labels = label.cuda()

        input_data = embeds(autograd.Variable(data))
        out = model(input_data)
        loss = criterion(out, autograd.Variable(label.float()))

        loss_sum += loss.data[0]
        count += 1

        if count % 100 == 0:
            print("epoch", epoch, end='  ')
            print("The loss is: %.5f" % (loss_sum/100))

            loss_sum = 0
            count = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # save the model in every epoch
    model.save('checkpoints/epoch{}.ckpt'.format(epoch))

