import torch
from torch.autograd import Variable
import torch.nn as nn

from datasets.assin import ASSIN

from metrics.pearson_correlation import PearsonCorrelation
from metrics.spearman_correlation import SpearmanCorrelation


def get_dataset(args):
    if args.dataset == 'assin':
        train_loader, dev_loader, test_loader = ASSIN.iters(batch_size=args.batch_size, device=args.device, shuffle=True)

        embedding_dim = ASSIN.TEXT.vocab.vectors.size()
        embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
        embedding.weight = nn.Parameter(ASSIN.TEXT.vocab.vectors)
        embedding.weight.requires_grad = False

        return ASSIN, train_loader, dev_loader, test_loader, embedding


def get_testset(args):
    if args.dataset == 'assin':
        test_evaluation = ASSIN.iters_testset(batch_size=1000, device=args.device, shuffle=True)

        #embedding_dim = ASSIN.TEXT.vocab.vectors.size()
        #embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
        #embedding.weight = nn.Parameter(ASSIN.TEXT.vocab.vectors)
        #embedding.weight.requires_grad = False

        return test_evaluation


def get_dataset_configurations(args):
    if args.dataset == 'assin':
        loss_fn = nn.KLDivLoss()
        metrics = {
            'pearson': PearsonCorrelation(),
            'spearman': SpearmanCorrelation()
        }

        def y_to_score(y, batch):
            num_classes = batch.dataset.num_classes
            predict_classes = Variable(torch.arange(1, num_classes + 1).expand(len(batch.id), num_classes))
            if y.is_cuda:
                with torch.cuda.device(y.get_device()):
                    predict_classes = predict_classes.cuda()

            return (predict_classes * y).sum(dim=1)

        def resolved_pred_to_score(y, batch):
            num_classes = batch.dataset.num_classes
            predict_classes = Variable(torch.arange(1, num_classes + 1).expand(len(batch.id), num_classes))
            if y.is_cuda:
                with torch.cuda.device(y.get_device()):
                    predict_classes = predict_classes.cuda()

            return (predict_classes * y.exp()).sum(dim=1)

        #resolved_pred_to_score = (lambda y, batch: y) if args.unsupervised else resolved_pred_to_score
        resolved_pred_to_score = resolved_pred_to_score

        return loss_fn, metrics, y_to_score, resolved_pred_to_score
