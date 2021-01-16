import argparse
import json
import os
import shutil
import sys

import configs.config_template as CONFIG  # Allows use of autocomplete, this is overwritten by cmd line argument
import torch
import torch.nn as nn

from cqa_dataloader import build_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--expt_name', required=True, type=str)
parser.add_argument('--data_root', default='data', type=str)

args = parser.parse_args()

EXPT_DIR = os.path.join(args.data_root, 'experiments', args.expt_name)

if args.evaluate or args.resume:
    if os.path.exists(EXPT_DIR):
        shutil.copy(os.path.join(EXPT_DIR, f'config_{args.expt_name}.py'), f'configs/config_{args.expt_name}.py')
    else:
        sys.exit("Experiment Folder does not exist")

exec(f'import configs.config_{args.expt_name} as CONFIG')
CONFIG.root = args.data_root


def inline_print(text):
    """
    A simple helper to print text inline. Helpful for displaying training progress among other things.
    Args:
        text: Text to print inline
    """
    sys.stdout.write('\r' + text)
    sys.stdout.flush()


def fit(net, dataloader, criterion, optimizer, epoch, config):
    """
    Train 1 epoch on the given dataloader and model

    Args:
        net: Model instance to train
        dataloader: Dataloader to use
        criterion: Training objective
        optimizer: Optimizer to use
        epoch: Current Epoch
        config: config
    """

    net.train()
    correct = 0
    total = 0
    total_loss = 0
    for q, a, i, qid, ql in dataloader:
        q = q.cuda()
        i = i.cuda()
        ql = ql.cuda()
        a = a.cuda()
        p = net(i, q, ql)
        loss = criterion(p, a)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
        optimizer.step()
        if config.dataset == 'FigureQA':
            p_scale = torch.sigmoid(p)
            pred_class = p_scale >= 0.5
            c = float(torch.sum(pred_class.float() == a))
        else:
            _, idx = p.max(dim=1)
            c = torch.sum(idx == a).item()
        correct += c
        total += len(ql)
        total_loss += loss * len(ql)
        inline_print(
            f'Running {dataloader.dataset.split}, Processed {total} of {len(dataloader) * dataloader.batch_size} '
            f'Accuracy: {correct / total}, Loss: {total_loss / total}')

    print(f'\nTrain Accuracy for Epoch {epoch + 1}: {correct / total}')


def predict(net, dataloaders, epoch, config):
    """
    Evaluate 1 epoch on the given list of dataloaders and model, prints accuracy and saves predictions

    Args:
        net: Model instance to train
        dataloaders: List of dataloaders to use
        epoch: Current Epoch
        config: config
    """
    net.eval()
    for data in dataloaders:
        correct = 0
        total = 0
        results = dict()
        with torch.no_grad():
            for q, a, i, qid, ql in data:
                q = q.cuda()
                i = i.cuda()
                ql = ql.cuda()
                a = a.cuda()
                p = net(i, q, ql)
                _, idx = p.max(dim=1)
                if config.dataset == 'FigureQA':
                    p_scale = torch.sigmoid(p)
                    pred_class = p_scale >= 0.5
                    c = float(torch.sum(pred_class.float() == a))
                    for qqid, curr_pred_class in zip(qid, pred_class):
                        qqid = int(qqid.item())
                        if qqid not in results:
                            results[qqid] = int(curr_pred_class)
                else:
                    c = torch.sum(idx == a).item()
                    for qqid, pred in zip(qid, idx):
                        qqid = int(qqid.item())
                        if qqid not in results:
                            results[qqid] = int(pred.item())

                correct += c
                total += len(ql)
                print_str = f'Running {data.dataset.split}, Processed {total} of {len(data) * data.batch_size} '
                if 'test' not in data.dataset.split:
                    inline_print(print_str + f'Accuracy: {correct / total}')
                else:
                    inline_print(print_str)

        result_file = os.path.join(EXPT_DIR, f'results_{data.dataset.split}_{epoch + 1}.json')
        json.dump(results, open(result_file, 'w'))
        print(f"Saved {result_file}")
        if 'test' not in data.dataset.split:
            print(f'\n{data.dataset.split} Accuracy for Epoch {epoch + 1}: {correct / total}')


def make_experiment_directory(config):
    if not args.evaluate and not args.resume and not config.overwrite_expt_dir:
        if os.path.exists(EXPT_DIR):
            raise RuntimeError(f'Experiment directory {EXPT_DIR} already exists, '
                               f'and the config is set to do not overwrite')

    if not os.path.exists(EXPT_DIR):
        os.makedirs(EXPT_DIR)


def update_learning_rate(epoch, optimizer, config):
    if epoch < len(config.lr_warmup_steps):
        optimizer.param_groups[0]['lr'] = config.lr_warmup_steps[epoch]
    elif epoch in config.lr_decay_epochs:
        optimizer.param_groups[0]['lr'] *= config.lr_decay_rate


def training_loop(config, net, train_loader, val_loaders, test_loaders, optimizer, criterion, start_epoch=0):
    for epoch in range(start_epoch, config.max_epochs):
        update_learning_rate(epoch, optimizer, config)
        fit(net, train_loader, criterion, optimizer, epoch, config)
        curr_epoch_path = os.path.join(EXPT_DIR, str(epoch + 1) + '.pth')
        latest_path = os.path.join(EXPT_DIR, 'latest.pth')
        data = {'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']}
        torch.save(data, curr_epoch_path)
        torch.save(data, latest_path)

        if epoch % config.test_interval == 0 or epoch >= config.test_every_epoch_after:
            predict(net, val_loaders, epoch, config)
            predict(net, test_loaders, epoch, config)


def evaluate_saved(net, dataloader, config):
    weights_path = os.path.join(EXPT_DIR, 'latest.pth')
    saved = torch.load(weights_path)
    net.eval()
    net.load_state_dict(saved['model_state_dict'])
    predict(net, dataloader, saved['epoch'], config)


# %%
def main():
    make_experiment_directory(CONFIG)
    print('Building Dataloaders according to configuration')

    if args.evaluate or args.resume:
        CONFIG.lut_location = os.path.join(EXPT_DIR, 'LUT.json')
        train_data, val_data, test_data, n1, n2 = build_dataloaders(CONFIG)
    else:
        train_data, val_data, test_data, n1, n2 = build_dataloaders(CONFIG)
        lut_dict = {'ans2idx': train_data.dataset.ans2idx,
                    'ques2idx': train_data.dataset.ques2idx,
                    'maxlen': train_data.dataset.maxlen}
        json.dump(lut_dict, open(os.path.join(EXPT_DIR, 'LUT.json'), 'w'))
        shutil.copy(f'configs/config_{args.expt_name}.py',
                    os.path.join(EXPT_DIR, 'config_' + args.expt_name + '.py'))

    print('Building model to train: ')
    if CONFIG.dataset == 'FigureQA':
        net = CONFIG.use_model(n1, 1, CONFIG)
    else:
        net = CONFIG.use_model(n1, n2, CONFIG)

    print("Model Overview: ")
    print(net)
    net.cuda()
    start_epoch = 0
    if not args.evaluate:
        print('Training...')
        optimizer = CONFIG.optimizer(net.parameters(), lr=CONFIG.lr)
        criterion = torch.nn.CrossEntropyLoss()
        if CONFIG.dataset == 'FigureQA':
            criterion = nn.BCEWithLogitsLoss()

        if args.resume:
            resumed_data = torch.load(os.path.join(EXPT_DIR, 'latest.pth'))
            print(f"Resuming from epoch {resumed_data['epoch'] + 1}")
            net.load_state_dict(resumed_data['model_state_dict'])
            optimizer = CONFIG.optimizer(net.parameters(), lr=resumed_data['lr'])
            optimizer.load_state_dict(resumed_data['optim_state_dict'])
            start_epoch = resumed_data['epoch']
        training_loop(CONFIG, net, train_data, val_data, test_data, optimizer, criterion, start_epoch)

    else:
        print('Evaluating...')
        evaluate_saved(net, test_data, CONFIG)
        evaluate_saved(net, val_data, CONFIG)


if __name__ == "__main__":
    main()
