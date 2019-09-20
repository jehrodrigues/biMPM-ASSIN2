import argparse
import logging

import numpy as np
import random
import torch
import torch.optim as O

from datasets import get_dataset, get_dataset_configurations
from models import get_model
from runners import Runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentence similarity model')
    parser.add_argument('--model', default='bimpm', choices=['bimpm'], help='Model to use')
    parser.add_argument('--dataset', default='assin', choices=['assin'], help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--regularization', type=float, default=3e-4, help='Regularization')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducibility')
    parser.add_argument('--device', type=int, default=0, help='Device, -1 for CPU')
    parser.add_argument('--log-interval', type=int, default=50, help='Device, -1 for CPU')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    dataset_cls, train_loader, dev_loader, test_loader, embedding = get_dataset(args)
    model = get_model(args, dataset_cls, embedding)

    total_params = 0
    for param in model.parameters():
        size = [s for s in param.size()]
        total_params += np.prod(size)
    logger.info('Total number of parameters: %s', total_params)

    loss_fn, metrics, y_to_score, resolved_pred_to_score = get_dataset_configurations(args)

    optimizer = O.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.regularization)
    runner = Runner(model, loss_fn, metrics, optimizer, y_to_score, resolved_pred_to_score, args.device, None)
    runner.run(args.epochs, train_loader, dev_loader, test_loader, args.log_interval)
