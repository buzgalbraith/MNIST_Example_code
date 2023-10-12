## this file is used to train the model and save it to a file
from example_cnn import *
from load_mnist import *
import time

## load data
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
    batch_size=32,
)
train_loader, test_loader = mnist_dataloader.load_data()
## instantiate the conv net

## we can also use the nn.DataParallel module to train on multiple GPUs. if you have the time try running with this line muted and then unmuted
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
con_net_instance = conv_net( 
    output_size=10, hidden_size=16 ## try shifting this to 16, 
).to(device)  ## here the input image size is [28X28] and the classes are the digits [0,9]

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
## train the model
loss_fn = nn.CrossEntropyLoss()  ## defining the loss function
max_epochs = 10
optimizer = torch.optim.Adam(
    con_net_instance.parameters(), lr=0.0001
)  ## defining the optimizer
for epoch in range(max_epochs):
    start_time = time.time()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device) ## send the data to the GPU if available
        y_hat = con_net_instance(x)
        loss = loss_fn(y_hat, y.long())
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(
        f"Finished epoch: {epoch+1} in {time.time() - start_time} seconds with average loss: {running_loss/len(train_loader)}"
    )
    ## typically we would also want to check performance on a validation set here, but we will skip that for brevity

## assuming this trains well, which it may not we can save the model
con_net_instance.save_model(file_path="example_conv_net.pt")
