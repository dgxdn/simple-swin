import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from models.swin import SwinTransformer

train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
 download=True, transform=ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
 download=True, transform=ToTensor())
batch_size = 64
train_Dataloader = DataLoader(train_data, batch_size=batch_size)
test_Dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in test_Dataloader:
    print(X.shape)
    print(y.shape)
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_availabe()
    else "cpu"
)
print(f"using {device} device")

model = SwinTransformer( img_size=32, patch_size=2, in_chans=3, num_classes=10,
                 embed_dim=48, depths=[2, 6, 2], num_heads=[3, 6, 12],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d} / {size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(f"size: {size}  num_batches : {num_batches}")
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print("")
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 20
print(len(test_Dataloader.dataset))
for t in range(epochs):
    print(f"epoch {t+1}")
    train(model, train_Dataloader, loss_fn, optimizer)
    test(test_Dataloader, model,loss_fn)
print("Done!")