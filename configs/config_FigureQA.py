import torch
from torchvision import transforms

import model

# Dataset Store Definitions
train_file = dict()
train_file['DVQA'] = 'DVQA_train_qa_oracle.json'
train_file['FigureQA'] = 'FigureQA_train_qa.json'

val_files = dict()
val_files['DVQA'] = {'val_easy': 'DVQA_val_easy_qa_oracle.json',
                     'val_hard': 'DVQA_val_hard_qa_oracle.json'}

val_files['FigureQA'] = {'val1': 'FigureQA_val1_qa.json',
                         'val2': 'FigureQA_val2_qa.json'}

test_files = dict()
test_files['FigureQA'] = {'test1': 'FigureQA_test1_qa.json',
                          'test2': 'FigureQA_test2_qa.json'}
test_files['DVQA'] = {}

transform_combo_train = dict()
transform_combo_test = dict()

transform_combo_train['FigureQA'] = transforms.Compose([
    transforms.Resize((224, 320)),
    transforms.RandomCrop(size=(224, 320), padding=8),
    transforms.RandomRotation(2.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.9365, 0.9303, 0.9295],
                         std=[1, 1, 1])
])

transform_combo_train['DVQA'] = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(size=(256, 256), padding=8),
    transforms.RandomRotation(2.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8744, 0.8792, 0.8836],
                         std=[1, 1, 1])

])

transform_combo_test['FigureQA'] = transforms.Compose([
    transforms.Resize((224, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.9365, 0.9303, 0.9295],
                         std=[1, 1, 1])

])

transform_combo_test['DVQA'] = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8744, 0.8792, 0.8836],
                         std=[1, 1, 1])

])

# Data and Preprocessing

root = 'data'  # This will be overwritten by command line argument
dataset = 'FigureQA'  # Should be defined above in the datastore section
data_subset = 1.0  # Random Fraction of data to use for training
batch_size = 64

train_filename = train_file[dataset]
val_filenames = val_files[dataset]
test_filenames = test_files[dataset]

train_transform = transform_combo_train[dataset]
test_transform = transform_combo_test[dataset]

lut_location = ''  # When training, LUT for question and answer token to idx is computed from scratch if left empty, or
# if your data specification has not changed, you can copy previously computed LUT.json and point to it to save time
# When resuming or evaluating, this is ignored and LUT computed for that experiment will be used

# Model Details

use_model = model.PReFIL
word_emb_dim = 32
ques_lstm_out = 256
num_hidden_act = 1024
num_rf_out = 256
num_bimodal_units = 256

image_encoder = 'dense'

if image_encoder == 'dense':
    densenet_config = (6, 6, 6)
    densenet_dim = [128, 160, 352] # Might be nice to compute according to densenet_config

# Training/Optimization

optimizer = torch.optim.Adamax
test_interval = 5  # In epochs
test_every_epoch_after = 20
max_epochs = 100
overwrite_expt_dir = False  # For convenience, set to True while debugging
grad_clip = 50

# Parameters for learning rate schedule

lr = 7e-4
lr_decay_step = 2  # Decay every this many epochs
lr_decay_rate = .7
lr_decay_epochs = range(15, 25, lr_decay_step)
lr_warmup_steps = [0.5 * lr, 1.0 * lr, 1.0 * lr, 1.5 * lr, 2.0 * lr]
dropout_classifier = True
