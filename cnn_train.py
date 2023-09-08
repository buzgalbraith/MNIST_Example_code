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
    batch_size=5,
)
train_loader, test_loader = mnist_dataloader.load_data()
## instantiate the conv net
con_net_instance = conv_net(
    input_size=28, output_size=10
)  ## here the input image size is [28X28] and the classes are the digits [0,9]
## train the model
loss_fn = nn.CrossEntropyLoss()  ## defining the loss function
optimizer = torch.optim.Adam(
    con_net_instance.parameters(), lr=0.0001
)  ## defining the optimizer
for epoch in range(2):
    start_time = time.time()
    running_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat = con_net_instance(x)
        loss = loss_fn(y_hat, y.long())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(
        f"Finished epoch: {epoch} in {time.time() - start_time} seconds with average loss: {running_loss/len(train_loader)}"
    )
    ## typically we would also want to check performance on a validation set here, but we will skip that for brevity

## assuming this trains well, which it may not we can save the model
con_net_instance.save_model(file_path="example_conv_net.pt")
