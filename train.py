"""Train Real NVP on MNIST.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""


import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
import sys # for my favorite debugging method: print(..) ; sys.exit()

from models import RealNVP, RealNVPLoss
from tqdm import tqdm

from torchsummary import summary


IN_CHANNELS = 1 # Standard: IN_CHANNELS = 3 (for CIFAR10)
MID_CHANNELS = 64 # Standard
NUM_BLOCKS = 3 # Standard: NUM_BLOCKS = 8 // # Corresponds to NR of resnet blocks in scale and translation networks
NUM_SCALES = 1 # Standard: NUM_SCALES=2 ; # with NUM_SCALES = 1 we just have 4 coupling layers with checkerboard masking and w/o channel wise masking
NUM_EPOCHS = 10
NUM_SAMPLES_TRAIN = 2e3 # number samples per epoch in train time. There are 60k images in MNIST
NUM_SAMPLES_TEST = 5000 # number samples per epoch in test time evaluation # 10k test samples in total
BATCH_SIZE = 64 # Standard: BATCH_SIZE = 64
MODEL_PATH = 'model_checkpoints/model_test.pth.tar'
RESOLUTION = [28, 28]

def main(args):
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(RESOLUTION),
        transforms.ToTensor()    
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    
    # The original code used CIFAR10
    # trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # The original code used CIFAR10
    # testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # 10k samples in testloader


    # Model
    print('Building model..')
    net = RealNVP(num_scales=NUM_SCALES, in_channels=IN_CHANNELS, mid_channels=MID_CHANNELS, num_blocks=NUM_BLOCKS)
    print(net)


    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    for epoch in tqdm(range(start_epoch, start_epoch + args.num_epochs)):
        tqdm.write(f"Epoch {epoch}")
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
        test(epoch, net, testloader, device, loss_fn, args.num_samples)


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    
    # print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    # print('Train loader data set summary',trainloader.dataset)
    print('Nr of samples in data set',len(trainloader.dataset))

    # with tqdm(total=len(trainloader.dataset)) as progress_bar:

    with tqdm(total=NUM_SAMPLES_TRAIN) as progress_bar:
        print(' --- TRAIN --- ')
        for idx, (x, _) in enumerate(trainloader):

            if (idx+1)*BATCH_SIZE >= NUM_SAMPLES_TRAIN: 
                break
            
            x = x.to(device)

            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))
        
        # save model after each epoch
        torch.save(net.state_dict(), MODEL_PATH)

def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    # for CIFAR10
    #z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    # for MNIST
    z = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


def test(epoch, net, testloader, device, loss_fn, num_samples):
    
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()

    with torch.no_grad():
        print(' --- TEST --- ')
        with tqdm(total=NUM_SAMPLES_TEST) as progress_bar:

        # commented out for shorter 
        # with tqdm(total=len(testloader.dataset)) as progress_bar:
            for idx, (x, _) in enumerate(testloader):

                if (idx+1)*BATCH_SIZE >= NUM_SAMPLES_TEST: 
                    break

                x = x.to(device)
                z, sldj = net(x, reverse=False)         
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on MNIST')

    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=NUM_EPOCHS, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = 0

    main(parser.parse_args())
