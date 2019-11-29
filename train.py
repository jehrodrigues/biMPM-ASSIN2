# -*- coding: utf-8 -*-

from ignite.engine.engine import Engine
from xml.dom import minidom
from datasets import get_testset
import torch


def _prepare_batch(batch):
    x, y = batch, batch.relatedness_score
    return x, y


def create_supervised_trainer(model, optimizer, loss_fn, cuda=False):
    """
    Factory function for creating a trainer for supervised models
    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: a trainer engine with supervised update function
    """
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.data.cpu()[0]

    return Engine(_update)


def create_supervised_evaluator(model, metrics=None, y_to_score=None, pred_to_score=None, cuda=False):
    """
    Factory function for creating an evaluator for supervised models
    Args:
        model (torch.nn.Module): the model to train
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    def _inference(engine, batch):
        model.eval()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        if y_to_score is not None:
            y = y_to_score(y, batch)

        if pred_to_score is not None:
            y_pred = pred_to_score(y_pred, batch)

        return batch.id, y_pred, y

    engine = Engine(_inference)

    if metrics is None:
        metrics = {}

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_evaluator_test(model, metrics=None, y_to_score=None, pred_to_score=None, cuda=False):
    """
    Factory function for creating an evaluator for supervised models
    Args:
        model (torch.nn.Module): the model to train
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    def _write_xml(filename, pred):
        """Docstring."""
        with open(filename, encoding='utf8') as fp:
            xml = minidom.parse(fp)
            print('Iniciou XML')
        pairs = xml.getElementsByTagName('pair')
        for pair in pairs:
            #print('pred: ', pred)
            sim = str(pred[pairs.index(pair)]).split(',')
            similarity = sim[0].replace('tensor(', '')
            #print('similarity: ', similarity)
            #print('pairs.index: ', pairs.index(pair))
            #print('pair: ', str(pred[pairs.index(pair)]))
            pair.setAttribute('similarity', similarity)
        with open(filename, 'w', encoding='utf8') as fp:
            fp.write(xml.toxml())
            print('XML escrito')

    def _inference(engine, batch):
        model.eval()
        #print(len(batch.relatedness_score))
        x, y = _prepare_batch(batch)
        #x = batch
        y_pred = model(x)
        if y_to_score is not None:
            y = y_to_score(y, batch)

        if pred_to_score is not None:
            y_pred = pred_to_score(y_pred, batch)

        #_write_xml('/home/jessica/Projects/ASSIN/biMPM-ASSIN2/data/assin/assin-ptbr-test.xml', y_pred)
        _write_xml('/home/jessica/teste-bimpm/data/assin/output.xml', y_pred)
        return batch.id, y_pred, y

    engine = Engine(_inference)

    if metrics is None:
        metrics = {}

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
