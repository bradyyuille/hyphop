import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from KFAttention import KFAttention
from KFLayer import KFLayer
from KFPooling import KFPooling

from hflayers.activation import HopfieldCore

model_params = {
    "input_dim": 784,
    "hidden_dim": 128,
    "num_classes": 10,
    "beta": None,
    "num_memories": 64
}

class ModelWrapper(nn.Module):
    def __init__(self, mode, input_dim, hidden_dim, num_classes, beta, num_memories=64):
        super().__init__()
        self.mode = mode
        self.embedder = nn.Linear(input_dim, hidden_dim)
        
        # Karcher Flow Models
        if mode == "kf_attention":
            self.core = KFAttention(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
        elif mode == "kf_pooling":
            self.core = KFPooling(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
            self.static_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        # elif mode == "kf_layer":
        #     self.core = KFLayer(num_memories=num_memories, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
            
        # HNIAYN Models
        elif mode == "hf_attention":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1)
        elif mode == "hf_pooling":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1, query_as_static=True)
            self.static_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        elif mode == "hf_layer":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1, key_as_static=True, value_as_static=True)
            self.static_key = nn.Parameter(torch.randn(num_memories, 1, hidden_dim) * 0.02)
            self.static_value = nn.Parameter(torch.randn(num_memories, 1, hidden_dim) * 0.02)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        embeds = self.embedder(x) # (B, hidden_dim)
        batch_size = embeds.size(0) 

        hniayn_embeds = embeds.unsqueeze(0) # expects (SeqLen, B, hidden_dim)

        if self.mode == "kf_attention":
            z = self.core(embeds, embeds, embeds)
        elif self.mode == "kf_pooling":
            q = self.static_query
            z = self.core(q, embeds, embeds)
        elif self.mode == "kf_layer":
            z = self.core(embeds)
        elif self.mode == "hf_attention":
            z, *_ = self.core(hniayn_embeds, hniayn_embeds, hniayn_embeds)
            z = z.squeeze(0)
        elif self.mode == "hf_pooling":
            q = self.static_query.expand(-1, batch_size, -1)
            z, *_ = self.core(q, hniayn_embeds, hniayn_embeds)
            z = z.squeeze(0)
        elif self.mode == "hf_layer":
            k = self.static_key.expand(-1, batch_size, -1)
            v = self.static_value.expand(-1, batch_size, -1)
            z, *_ = self.core(hniayn_embeds, k, v)
            z = z.squeeze(0)

            
        return self.classifier(z)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='3on3 MNIST')
    parser.add_argument('--model', type=str, default='kf_attention', 
                        choices=['kf_attention', 'kf_layer', 'kf_pooling', 
                                 'hf_attention', 'hf_layer', 'hf_pooling'])
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ModelWrapper(mode=args.model, **model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # weight_decay=1e-4) # added weight decay

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_3on3.pt")

if __name__ == '__main__':
    main()