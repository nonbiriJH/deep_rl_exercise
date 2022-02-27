from operator import mod
from torch.utils.data import DataLoader
from torch import nn  # model, loss
from torch import optim  # optimiser
import torch

from _1_dataset import training_data, test_data
from _2_nn_model import NeuralNetwork
from _3_process_train import train_loop
from _4_process_test import test_loop
from _5_model_save_load import model_save_load

epochs = 1
learning_rate = 1e-3
batch_size = 64
model_dir = "model_archive/model_weights"

# Data Loader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#model, loss, optimiser
model = NeuralNetwork()
# mean negative log softmax on output node corresponding to label
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# load model if exists
model_save_load(model=model, mode="load", model_dir=model_dir)

# run train and test for each epoch
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# save model
model_save_load(model=model, mode="save", model_dir=model_dir)
