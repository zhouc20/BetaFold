'''
Author: lin-yh20@mails.tsinghua.edu.cn
Date: 2023-04-24
LastEditTime: 2023-04-24
LastEditors: Yuhuan Lin
Description: the training file
             see networkTool.py to set up the parameters
             will generate training log file loss.log and checkpoint in folder 'expName'
FilePath: /model/Transformer.py
'''
import math
import torch.nn as nn
from networkTool import *
from Attention_Module import *

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder1 = nn.Embedding(20, embed1)
        #self.encoder2 = nn.Embedding(, 6)
        #self.encoder3 = nn.Embedding(9, 4)

        self.ninp = ninp
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(ninp, ninp)
        self.decoder1 = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data)
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data = nn.init.xavier_normal_(self.decoder0.weight.data)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data = nn.init.xavier_normal_(self.decoder1.weight.data)

    def forward(self, src , dataFeat):

        aci_len = src.shape[1]

        Aci = src[:, :, 0]
        Dihedral_Angle = src[ :, :, 1]
        Site_pair = src[:, :, 2]

        aAci = self.encoder(Aci.long())  # a[bptt,batchsize,FeatDim(levels),EmbeddingDim]
        aDihedral_Angle = self.encoder1(Dihedral_Angle.long())
        aSite_pair = self.encoder2(Site_pair.long())

        a = torch.cat((aAci, aDihedral_Angle, aSite_pair), 3)

        src = a.reshape((aci_len, -1))

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder1(self.act(self.decoder0(output)))



        return output