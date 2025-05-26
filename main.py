import sys

import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from resnet import ResNet18CIFAR
from file_output_duplicator import FileOutputDuplicator
from solver import Solver
from utils import *
import arguments

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cifar_test_transform():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408, ],
                                 std=[0.2675, 0.2565, 0.2761]),
        ])

def main(args):
    sys.stdout = FileOutputDuplicator(sys.stdout, 'stdout.txt', 'w')
    sys.stderr = FileOutputDuplicator(sys.stderr, 'stderr.txt', 'w')

    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_test_transform(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)

        args.num_images = 50000
        args.num_val = 5000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_test_transform(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_val = 0
        args.num_images = 50000
        args.budget = 2000
        args.initial_budget = 2000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))
    val_indices = []
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    seed_everything(42)
    initial_indices = np.random.choice(all_indices, args.initial_budget, replace=False)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)
            
    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    splits = list(range(2000, 20001, 2000))

    current_indices = list(initial_indices)

    # need to retrain all the models on the new images
    # re initialize and retrain the models
    # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
    seed_everything(42)
    task_model = ResNet18CIFAR(num_classes=args.num_classes)
    vae = model.VAE(args.latent_dim)
    discriminator = model.Discriminator(args.latent_dim)

    state_dicts = {}
    state_dicts['vae'] = vae.state_dict()
    state_dicts['discriminator'] = discriminator.state_dict()
    state_dicts['task_model'] = task_model.state_dict()

    seed_everything(42)
    accuracies = []
    for i, split in enumerate(splits):
        task_model.load_state_dict(state_dicts['task_model'])
        vae.load_state_dict(state_dicts['vae'])
        discriminator.load_state_dict(state_dicts['discriminator'])

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        print("=" * 80)
        print("AL Round nr. ", i)
        print("Number of Train labels: ", len(current_indices))
        print("Number of Unlabeled pools: ", len(unlabeled_indices))
        print("=" * 80)
        # train the models on the current data
        acc, vae, discriminator = solver.train(querry_dataloader,
                                               val_dataloader,
                                               task_model, 
                                               vae, 
                                               discriminator,
                                               unlabeled_dataloader)


        print('Final test accuracy with {} number of training samples is: {:.2f}'.format(split, acc))
        accuracies.append(acc)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))
    print("FINAL TEST Accuracies: ", accuracies)

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

