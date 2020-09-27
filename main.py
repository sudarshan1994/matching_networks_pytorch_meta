import torch
import math
import os
import time
import json
import logging
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchmeta.utils.data import BatchMetaDataLoader


from metamodel import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from torchmeta.datasets import Omniglot, MiniImagenet,TieredImagenet
from torchvision.transforms import ToTensor, Resize, Compose
from matching_network import MatchingNetTrainer
from matching_model import MatchingNetwork
import  torch.nn.functional as F

from pdb import set_trace

def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))

        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))

        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))



    if args.dataset == 'omniglot':
        dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shots,
                                      num_test_per_class=args.num_shots_test)
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])


        meta_train_dataset = Omniglot(args.folder,
                                      transform=transform,
                                      target_transform=Categorical(args.num_ways),
                                      num_classes_per_task=args.num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)    

        meta_val_dataset = Omniglot(args.folder,
                                    transform=transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)

        meta_test_dataset = Omniglot(args.folder,
                                     transform=transform,
                                     target_transform=Categorical(args.num_ways),
                                     num_classes_per_task=args.num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

        #model = ModelConvOmniglot(args.num_ways, hidden_size=64)
        model = MatchingNetwork(args.num_shots, args.num_ways, args.num_shots_test, fce = True, num_input_channels =1, lstm_layers =1, lstm_input_size = 64, unrolling_steps =2, device = device )
        loss_fn = F.nll_loss


    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    best_value = None
    
    matching_net_trainer = MatchingNetTrainer(args,model, 
                                meta_optimizer, 
                                num_adaptation_steps = args.num_steps,
                                step_size = args.step_size,
                                loss_fn = loss_fn)
    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    for epoch in range(args.num_epochs):
        matching_net_trainer.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False)

        results = matching_net_trainer.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet','tieredimagenet'], default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=4,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=5,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=600,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.01,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
