import os
from packaging import version

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import datasets, transforms


import horovod
import horovod.torch as hvd

import hydra
from omegaconf import DictConfig


from example.init.optimizer import init_optimizer_from_config

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

@hydra.main(config_name='config', version_base=None, config_path='conf')
def main(cfg: DictConfig, ):
    def train_mixed_precision(epoch, scaler):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if cfg.training.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.nll_loss(output, target)

            scaler.scale(loss).backward()
            # Make sure all async allreduces are done
            optimizer.synchronize()
            # In-place unscaling of all gradients before weights update
            scaler.unscale_(optimizer)
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
            # Update scaler in case of overflow/underflow
            scaler.update()

            if batch_idx % cfg.training.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss Scale: {}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                           100. * batch_idx / len(train_loader), loss.item(), scaler.get_scale()))
    
    def train_epoch(epoch):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if cfg.training.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % cfg.training.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                           100. * batch_idx / len(train_loader), loss.item()))

    def metric_average(val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def test():
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in test_loader:
            if cfg.training.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(cfg.training.seed)

    if cfg.training.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(cfg.training.seed)
    else:
        if cfg.training.use_mixed_precision:
            raise ValueError("Mixed precision is only supported with cuda enabled.")

    if (cfg.training.use_mixed_precision and version.parse(torch.__version__)
            < version.parse('1.6.0')):
        raise ValueError("""Mixed precision is using torch.cuda.amp.autocast(),
                            which requires torch >= 1.6.0""")

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.training.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    data_dir = cfg.dataset.data_dir or './data'
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = \
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = \
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size_test,
                                              sampler=test_sampler, **kwargs)

    model = Net()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not cfg.training.use_adasum else 1

    if cfg.training.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if cfg.training.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    # optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
    #                       momentum=args.momentum)

    opt_cls, kwargs = init_optimizer_from_config(
            cfg.optimizer, model.forward_features.parameters()
        )

    
    optimizer = opt_cls([kwargs])
    
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if cfg.training.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if cfg.training.use_adasum else hvd.Average,
                                         gradient_predivide_factor=cfg.training.gradient_predivide_factor)

    if cfg.training.use_mixed_precision:
        # Initialize scaler in global scale
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, cfg.training.epochs + 1):
        if cfg.training.use_mixed_precision:
            train_mixed_precision(epoch, scaler)
        else:
            train_epoch(epoch)
        # Keep test in full precision since computation is relatively light.
        test()

    
