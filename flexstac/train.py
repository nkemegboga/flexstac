from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd512 import build_ssd
import numpy as np
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = (v1, v2)[args.version == 'v2']


if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

#os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2'
train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
ssd_dim = 512  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = 21
num_loc = 4
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = args.iterations
weight_decay = args.weight_decay
stepvalues = (int(0.67*args.iterations), int(0.83*args.iterations), args.iterations)
gamma = 0.1
momentum = 0.9

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', ssd_dim, num_classes=num_classes, num_loc=num_loc)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net, device_ids=[0, 1])
    

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
        
def get_smaller_set(dataset, max_cat_thresh=20, min_cat_thresh=15, categories_count=20):

    keep_indices = []
    labels = list(range(0,categories_count))
    labels_dict = {l: 0 for l in labels}

    for ix, (i, j) in enumerate(dataset):
        keep = True
        for val in j[:, -1]:
            if labels_dict[(int(val))] >= max_cat_thresh:
                keep = False
                break
        if keep:
            for val in j[:, -1]:
                labels_dict[(int(val))] += 1
            keep_indices.append(ix)

        if min(labels_dict.values()) >= min_cat_thresh:
            break
            
    return keep_indices, labels_dict

def labels_check(dataset, categories_count=20):
    labels = list(range(0,categories_count))
    labels_dict = {l: 0 for l in labels}

    for ix, (i, j) in enumerate(dataset):
        for val in j[:, -1]:
            labels_dict[(int(val))] += 1
        
    return labels_dict

if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())
    dataset = data.Subset(dataset, get_smaller_set(dataset, max_cat_thresh=20, min_cat_thresh=15)[0])
    print(labels_check(dataset, categories_count=20))

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.dataset.name)
    print("Dataset size:", len(dataset))
    print("Epoch size:", epoch_size)
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,)
    start = time.time()
    for iteration in range(args.start_iter, max_iter):
        
        batch = 0
        batch_iterator = iter(data_loader)
        t0 = time.time()
        for images, targets in batch_iterator:
            batch += 1
            if batch % 100 == 0:
                print("{} batches processed ...".format(batch))
            
            if iteration in stepvalues:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda()) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno) for anno in targets]
            # forward
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss = loc_loss + loss_l.data.item()
            conf_loss = conf_loss + loss_c.item()
        
        t1 = time.time()
        print('iter: {} || Time taken: {} s || Loss: {}'.format(iteration, round(t1-t0, 4), round(loss.item(), 4)))
        if iteration % 50 == 0:
            torch.save(ssd_net.state_dict(), '{}ssd512_0712_iter_{}.pth'.format(args.save_folder, iteration))

    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')
    print("Training time: {} mins".format((time.time() - start)/60))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
