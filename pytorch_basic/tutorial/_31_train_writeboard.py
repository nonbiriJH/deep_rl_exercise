from _4_process_test import test_loop

def train_writeboard(epochs, train_dataloader, test_dataloader, model, loss_fn, optimizer, writer):
    # run train and test for each epoch
    running_loss = 0.0
    log_freq = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        #train
        size = len(train_dataloader.dataset)
        for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

        # Backpropagation
            optimizer.zero_grad() #reset grad on each weight to 0, otherwise accumulate on batch
            loss.backward() #compute loss grad wrt weights
            optimizer.step() #adjust weights by grad and learning rate
            running_loss += loss.item()
            if batch % log_freq == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") #
                writer.add_scalar('training loss',
                            running_loss / log_freq,
                            epochs * size + batch)
                running_loss = 0.0

        test_loop(test_dataloader, model, loss_fn)
    print("Done!")