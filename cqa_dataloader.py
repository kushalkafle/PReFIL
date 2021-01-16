import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader


class CQADataset(Dataset):
    def __init__(self, cqa_data, ans2idx, ques2idx, maxlen, split, config):
        self.cqa_data = cqa_data
        self.ques2idx = ques2idx
        self.ans2idx = ans2idx
        self.maxlen = maxlen
        self.split = split
        self.config = config
        if self.split == 'train':
            self.prep = config.train_transform
        else:
            self.prep = config.test_transform

    def __len__(self):
        return len(self.cqa_data)

    def __getitem__(self, index):
        ques, ques_len = encode_questions(self.cqa_data[index]['question'], self.ques2idx,
                                          self.maxlen)
        if 'test' not in self.split:
            ans = encode_answers(self.cqa_data[index]['answer'], self.ans2idx, self.config)
        else:
            ans = torch.zeros((1,))
        ques_id = self.cqa_data[index]['question_index']
        img_path = os.path.join(self.config.root, self.config.dataset, 'images', self.split,
                                self.cqa_data[index]['image_filename'])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.prep(img)
        return ques, ans, img_tensor, ques_id, ques_len


def encode_questions(question, ques2idx, maxlen):
    ques_vec = torch.zeros(maxlen).long()
    ques_words = word_tokenize(question.lower())
    ques_len = len(ques_words)
    for i, word in enumerate(ques_words):
        ques_vec[i] = ques2idx.get(word, len(ques2idx))  # last idx is reserved for <UNK>, needed for real OCRs
    return ques_vec, ques_len


def encode_answers(answer, ans2idx, config):
    if config.dataset == 'FigureQA':
        a = torch.zeros((1,))
        if answer == '1':
            a[0] = 0.0
        else:
            a[0] = 1.0
        return a
    else:
        return ans2idx.get(answer, len(ans2idx))


def collate_batch(data_batch):
    data_batch.sort(key=lambda x: x[-1], reverse=True)
    return torch.utils.data.dataloader.default_collate(data_batch)


def tokenize(q):
    return word_tokenize(q['question'].lower())


def build_lut(cqa_train_data):
    print("Building lookup table for question and answer tokens")

    pool = ProcessPoolExecutor(max_workers=8)
    questions = list(pool.map(tokenize, cqa_train_data, chunksize=1000))
    pool.shutdown()
    print("Finished")

    maxlen = max([len(q) for q in questions])
    unique_tokens = set([t for q in questions for t in q])
    ques2idx = {word: idx + 1 for idx, word in enumerate(unique_tokens)}  # save 0 for padding
    answers = set([q['answer'] for q in cqa_train_data])
    ans2idx = {ans: idx for idx, ans in enumerate(answers)}
    return ans2idx, ques2idx, maxlen


# %%
def build_dataloaders(config):
    cqa_train_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.train_filename)))
    if config.lut_location == '':
        ans2idx, ques2idx, maxlen = build_lut(cqa_train_data)
    else:
        lut = json.load(open(config.lut_location, 'r'))
        ans2idx = lut['ans2idx']
        ques2idx = lut['ques2idx']
        maxlen = lut['maxlen']

    n = int(config.data_subset * len(cqa_train_data))
    np.random.seed(666)
    np.random.shuffle(cqa_train_data)
    cqa_train_data = cqa_train_data[:n]
    train_dataset = CQADataset(cqa_train_data, ans2idx, ques2idx, maxlen, 'train', config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch,
                                  num_workers=8)

    val_datasets = []
    for split in config.val_filenames:
        cqa_val_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.val_filenames[split])))
        val_datasets.append(CQADataset(cqa_val_data, ans2idx, ques2idx, maxlen, split, config))

    val_dataloaders = []
    for vds in val_datasets:
        val_dataloaders.append(DataLoader(vds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch,
                                          num_workers=8))

    test_datasets = []
    for split in config.test_filenames:
        cqa_test_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.test_filenames[split])))
        n = int(config.data_subset * len(cqa_test_data))
        cqa_test_data = cqa_test_data[:n]
        test_datasets.append(CQADataset(cqa_test_data, ans2idx, ques2idx, maxlen, split, config))

    test_dataloaders = []
    for tds in test_datasets:
        test_dataloaders.append(DataLoader(tds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch,
                                           num_workers=8))

    return train_dataloader, val_dataloaders, test_dataloaders, len(ques2idx) + 1, len(ans2idx) + 1


def main():
    pass


if __name__ == '__main___':
    main()
