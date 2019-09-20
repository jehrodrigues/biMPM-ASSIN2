import torch

from models.bimpm import BiMPM


def get_model(args, dataset_cls, embedding):
    if args.model == 'bimpm':
        model = BiMPM(embedding, 300, 50, 20, 100, dataset_cls.num_classes, 0.1)
    else:
        raise ValueError(f'Unrecognized dataset: {args.model}')

    if args.device != -1:
        with torch.cuda.device(args.device):
            model = model.cuda()

    return model
