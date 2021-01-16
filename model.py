import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from blockwise_densenet import DenseNet


class QuestionEncoder(nn.Module):
    def __init__(self, num_tokens, config):
        super(QuestionEncoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, config.word_emb_dim)
        self.lstm = nn.LSTM(input_size=config.word_emb_dim,
                            hidden_size=config.ques_lstm_out,
                            num_layers=1)

    def forward(self, q, q_len):
        q_embed = self.embedding(q)
        packed = pack_padded_sequence(q_embed, q_len, batch_first=True)
        o, (h, c) = self.lstm(packed)
        return c.squeeze(0)


class DenseNetEncoder(nn.Module):
    def __init__(self, densenet_config):
        super(DenseNetEncoder, self).__init__()
        self.densenet = DenseNet(block_config=densenet_config).cuda()

    def forward(self, img):
        _, dense, final = self.densenet(img)
        return dense[0], dense[1], final


class BimodalEmbedding(nn.Module):
    def __init__(self, num_mmc_units, ques_dim, img_dim, num_mmc_layers=4):
        super(BimodalEmbedding, self).__init__()
        self.bn = nn.BatchNorm2d(ques_dim + img_dim)
        self.transform_convs = []
        self.num_mmc_layers = num_mmc_layers
        self.transform_convs.append(nn.Conv2d(ques_dim + img_dim, num_mmc_units, kernel_size=1))
        self.transform_convs.append(nn.ReLU())
        for i in range(num_mmc_layers - 1):
            self.transform_convs.append(nn.Conv2d(num_mmc_units, num_mmc_units, kernel_size=1))
            self.transform_convs.append(nn.ReLU())
        self.transform_convs = nn.Sequential(*self.transform_convs)

    def forward(self, img_feat, ques_feat):
        # Tile ques_vector, concatenate
        _, _, nw, nh = img_feat.shape
        _, qdim = ques_feat.shape
        ques_feat = ques_feat.unsqueeze(2)
        ques_tile = ques_feat.repeat(1, 1, nw * nh)
        ques_tile = ques_tile.view(-1, qdim, nw, nh)
        combine_feat = self.bn(torch.cat([img_feat, ques_tile], dim=1))
        bimodal_emb = self.transform_convs(combine_feat)
        return bimodal_emb


class Classifier(nn.Module):
    def __init__(self, num_classes, feat_in, config):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(feat_in, config.num_hidden_act)
        self.classifier = nn.Linear(config.num_hidden_act, num_classes)
        self.drop = nn.Dropout()
        self.use_drop = config.dropout_classifier

    def forward(self, bimodal_emb):
        # Tile ques_vector, concatenate
        projection = self.relu(self.lin1(bimodal_emb))
        if self.use_drop:
            projection = self.drop(projection)
        preds = self.classifier(projection)
        return preds


class RecurrentFusion(nn.Module):
    def __init__(self, num_bigru_units, feat_in):
        super(RecurrentFusion, self).__init__()
        self.bigru = nn.GRU(input_size=feat_in,
                            hidden_size=num_bigru_units,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, mmc_feat):
        _, fs, nw, nh = mmc_feat.shape
        mmc_feat = mmc_feat.view(-1, fs, nw * nh)
        mmc_feat = torch.transpose(mmc_feat, 1, 2)
        output, h = self.bigru(mmc_feat)
        return torch.flatten(torch.transpose(h, 0, 1), start_dim=1)


class BasePReFIL(nn.Module):
    def __init__(self, num_tokens, config):
        super(BasePReFIL, self).__init__()
        self.config = config
        self.rnn = QuestionEncoder(num_tokens, config)
        self.cnn = DenseNetEncoder(config.densenet_config)
        img_dims = config.densenet_dim
        self.bimodal_low = BimodalEmbedding(config.num_bimodal_units, config.ques_lstm_out, img_dims[0])
        self.bimodal_high = BimodalEmbedding(config.num_bimodal_units, config.ques_lstm_out, img_dims[2])
        self.maxpool_low = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.config = config

    @staticmethod
    def flatten_to_2d(mmc_feat):
        return mmc_feat.reshape(-1, mmc_feat.shape[1] * mmc_feat.shape[2] * mmc_feat.shape[3])

    def forward(self, img, ques, q_len):
        ques_feat = self.rnn(ques, q_len)
        feat_low, feat_mid, feat_high = self.cnn(img)
        feat_low = self.maxpool_low(feat_low)
        bimodal_feat_low = self.bimodal_low(feat_low, ques_feat)
        bimodal_feat_high = self.bimodal_high(feat_high, ques_feat)
        return bimodal_feat_low, bimodal_feat_high


class PReFIL(BasePReFIL):
    def __init__(self, num_tokens, num_ans_classes, config):
        super(PReFIL, self).__init__(num_tokens, config)
        self.rf_low = RecurrentFusion(config.num_rf_out, config.num_bimodal_units)
        self.rf_high = RecurrentFusion(config.num_rf_out, config.num_bimodal_units)
        self.classifier = Classifier(num_ans_classes, config.num_rf_out * 4, config)

    def forward(self, img, ques, q_len):
        bimodal_feat_low, bimodal_feat_high = super(PReFIL, self).forward(img, ques, q_len)
        rf_feat_low = self.rf_low(bimodal_feat_low)
        rf_feat_high = self.rf_high(bimodal_feat_high)
        final_feat = torch.cat([rf_feat_low, rf_feat_high], dim=1)
        answer = self.classifier(final_feat)
        return answer


def main():
    pass


if __name__ == '__main___':
    main()
