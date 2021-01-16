import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import configs.config_template as CONFIG
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', required=True, type=str)
parser.add_argument('--data_root', default='data', type=str)
parser.add_argument('--epoch', default=-1, type=int)  # -1 will evaluate latest epoch

args = parser.parse_args()

EXPT_DIR = os.path.join(args.data_root, 'experiments', args.expt_name)

if os.path.exists(EXPT_DIR):
    shutil.copy(os.path.join(EXPT_DIR, f'config_{args.expt_name}.py'), f'configs/config_{args.expt_name}.py')
else:
    sys.exit("Experiment Folder does not exist")

exec(f'import configs.config_{args.expt_name} as CONFIG')
CONFIG.root = args.data_root


def get_latest_epoch():
    weights_path = os.path.join(EXPT_DIR, 'latest.pth')
    saved = torch.load(weights_path)
    return saved['epoch'] + 1


def main():
    lut = json.load(open(os.path.join(EXPT_DIR, 'LUT.json')))
    idx2ans = {v: k for k, v in lut['ans2idx'].items()}

    for split, fname in CONFIG.val_filenames.items():
        epoch = args.epoch
        if epoch == -1:
            epoch = get_latest_epoch()

        preds = json.load(open(os.path.join(EXPT_DIR,
                                            f'results_{split}_{epoch}.json')))
        gt = json.load(open(os.path.join(CONFIG.root, CONFIG.dataset, 'qa', fname)))
        acc = {'overall': list(),
               'by_qtype': defaultdict(list),
               'by_imgtype': defaultdict(list)
               }

        for gt_qa in gt:
            pred_idx = preds[str(gt_qa['question_index'])]
            gt_ans = gt_qa['answer']
            gt_qtype = gt_qa['question_type']
            gt_imgtype = gt_qa['image_type']
            if pred_idx == len(idx2ans):
                acc['overall'].append(0)
                acc['by_imgtype'][gt_imgtype].append(0)
                acc['by_qtype'][gt_qtype].append(0)
            else:
                pred_ans = idx2ans[pred_idx]
                if pred_ans == gt_ans:
                    acc['overall'].append(1)
                    acc['by_imgtype'][gt_imgtype].append(1)
                    acc['by_qtype'][gt_qtype].append(1)
                else:
                    acc['overall'].append(0)
                    acc['by_imgtype'][gt_imgtype].append(0)
                    acc['by_qtype'][gt_qtype].append(0)

        print(f'Metrics for {split} split')
        for group in acc:
            print(f'\tCategory: {group}:')
            if isinstance(acc[group], dict):
                for sub_group in acc[group]:
                    print(
                        f'\t\tAccuracy for {sub_group} is: {100 * sum(acc[group][sub_group]) / len(acc[group][sub_group]):.6}')
            else:
                print(f'\t\tOverall accuracy is: {100 * sum(acc[group]) / len(acc[group]):.6}')


if __name__ == '__main__':
    main()
