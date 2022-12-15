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
from utils.augmentations import SSDAugmentation, SSDAugmentation_soft, SSDAugmentation_hard
from utils.flexSTAC import prep_unsupervised_data, labels_check, get_smaller_set
from layers.modules import MultiBoxLoss
from ssd512 import build_ssd
import numpy as np
import time, random, logging

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

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
num_iterations_before_relabel = 20
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
    print('Resuming training, loading {}...'.format(args.resume), flush=True)
    ssd_net.load_weights(args.resume)
    
else:
    vgg_weights = torch.load(os.path.join(args.save_folder, args.basenet))
    print('Loading base network...', flush=True)
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
        
def partition_data(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def update_teacher(teacher_model, student_model, alpha_teacher=0.5):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data[:] = alpha_teacher * teacher_param[:].data[:] + (1 - alpha_teacher) * student_param[:].data[:]
    return teacher_model

if not args.resume:
    print('Initializing weights...', flush=True)
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    teacher_model = build_ssd('train', ssd_dim, num_classes=num_classes, num_loc=num_loc)
    teacher_model.load_weights(args.resume)
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...', flush=True)

    # Labeled data
    dataset_supervised = VOCDetection(args.voc_root, [('2007', 'trainval'), ('2012', 'trainval')],
                                      SSDAugmentation(ssd_dim, means), AnnotationTransform())
    train_ds = data.Subset(dataset_supervised,
                           get_smaller_set(dataset_supervised, max_cat_thresh=20, min_cat_thresh=15)[0])
    print(labels_check(train_ds, categories_count=20), flush=True)
    data_loader_supervised = data.DataLoader(train_ds, 1000, num_workers=2,
                                             shuffle=True, collate_fn=detection_collate,)
    img_sup, label_sup = list(iter(data_loader_supervised))[0]
    
    step_index = 0
    start = time.time()
    for iteration in range(args.start_iter, max_iter):
        if iteration % num_iterations_before_relabel == 0:
            ssod_images, ssod_labels = None, None
            s_time = time.time()
            if iteration == 0:
                use_dynamic_thresh = False
                model_path = args.resume
            else:
                model_path = os.path.join(args.save_folder, "tmp.pth")
                torch.save(ssd_net.state_dict(), model_path)
                use_dynamic_thresh = True
                
            # Unlabeled data
            dataset_unlabeled = VOCDetection(args.voc_root, [('2007', 'trainval'), ('2012', 'trainval')],
                                             SSDAugmentation_soft(ssd_dim, means), AnnotationTransform())
            data_loader_unlabeled = data.DataLoader(dataset_unlabeled, 16, num_workers=2,
                                                    shuffle=True, collate_fn=detection_collate,)
            batch_iterator_unlabeled = iter(data_loader_unlabeled)
            ssod_images, ssod_labels = prep_unsupervised_data(
                voc_root=args.voc_root,
                batch_iterator=batch_iterator_unlabeled,
                model_path=model_path,
                ssd_dim=ssd_dim,
                means=means,
                num_classes=num_classes,
                detection_thresh=0.9,
                min_detection_thresh=0.7,
                use_dynamic_thresh=use_dynamic_thresh,
            )

            # Merge labeled and unlabeled data
            batch_iterator_unlabeled, data_loader_unlabeled = None, None
            ssod_labels.extend(label_sup)
            ssod_images.extend([img_sup_.unsqueeze(0) for img_sup_ in img_sup])

            print("Number of labels (after aug):", len(ssod_labels), flush=True)
            print("Number of images (after aug):", len(ssod_images), flush=True)
            print("Time taken to generate updated training data: {} mins".format((time.time()-s_time)/60), flush=True)

            epoch_size = len(ssod_labels) // args.batch_size
            print("# of batches per epoch:", epoch_size, flush=True)
        
        batch = 0
        dataset_length = len(ssod_labels)
        partitions = partition_data(list(range(dataset_length)), dataset_length//args.batch_size)
        t0 = time.time()
        for partition in partitions:
            images = torch.cat([ssod_images[part_] for part_ in partition], dim=0)
            targets = [ssod_labels[part_] for part_ in partition]
            batch += 1
            if batch % 100 == 0:
                print("{} batches of {} processed ...".format(batch, epoch_size), flush=True)
            
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
            
            teacher_model = update_teacher(teacher_model, net, alpha_teacher=0.5)
        
        t1 = time.time()
        print('iter: {} || Time taken: {} s || Loss: {}'.format(iteration, round(t1-t0, 4), round(loss.item(), 4)), flush=True)
        if iteration % 5 == 0:
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, 'ssd512_0712_iter_{}.pth'.format(iteration)))
            torch.save(teacher_model.state_dict(), os.path.join(args.save_folder, 'ssd512_0712_iter_{}_teacher.pth'.format(iteration)))

    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, "{}.pth".format(args.version)))
    torch.save(teacher_model.state_dict(), os.path.join(args.save_folder, "{}_teacher.pth".format(args.version)))
    print("Training time: {} mins".format((time.time() - start)/60), flush=True)


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
