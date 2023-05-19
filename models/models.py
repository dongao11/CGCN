import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import gnn_iclr


class Embedding(nn.Module):

    def __init__(self, args, emb_size,inputsize):
        super(Embedding, self).__init__()
        self.emb_size = emb_size
        self.inputsize=inputsize
        self.ndf = 64
        self.args = args

        # Input 84x84x3
        self.conv1 = nn.Conv2d(self.inputsize, 210, kernel_size=3, stride=1, padding=(3-1)//2, bias=False)
        self.bn1 = nn.BatchNorm2d(210)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(210, 220, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(220)

        # Input 20x20x96
        self.conv3 = nn.Conv2d(220, 230, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(230)
        self.drop_3 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2d(230, 256, kernel_size=3, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop_4 = nn.Dropout2d(0.5)

        # Input 5x5x256
        self.fc1 = nn.Linear(self.ndf*4*5*5, self.emb_size, bias=True)
        # self.fc1 = nn.Linear(self.ndf * 4 * 4 * 4, self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

        self.convv = nn.Conv2d(in_channels=self.inputsize, out_channels=256, kernel_size=1, padding=0, stride=2,
                          bias=False)

    def forward(self, input):
        e1 = self.bn1(self.conv1(input))
        x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = self.bn2(self.conv2(x))
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = self.bn3(self.conv3(x))
        x = F.leaky_relu(e3, 0.2, inplace=True)
        x = self.drop_3(x)
        e4 = self.bn4(self.conv4(x))
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        cinput=input
        rinput = self.convv(cinput)
        x=x+rinput

        x = x.view(-1, self.ndf*4*5*5)#
        # x = x.view(-1, self.ndf * 4 * 4 * 4)

        output = self.bn_fc(self.fc1(x))

        return [e1, e2, e3, e4, None, output]


class MetricNN(nn.Module):#
    def __init__(self, args, emb_size):
        super(MetricNN, self).__init__()

        self.metric_network = args.metric_network
        self.emb_size = emb_size
        self.args = args

        if self.metric_network == 'gnn_iclr_nl':#这里是使用gnn
            assert(self.args.train_N_way == self.args.test_N_way)#
            num_inputs = self.emb_size + self.args.train_N_way
            if self.args.dataset == 'IP':
                self.gnn_obj= gnn_iclr.GNN_nl(args, num_inputs, nf=96, J=1)
            elif 'UP' in self.args.dataset:
                self.gnn_obj= gnn_iclr.GNN_nl(args, num_inputs, nf=96, J=1)
            elif 'salinas' in self.args.dataset:
                self.gnn_obj= gnn_iclr.GNN_nl(args, num_inputs, nf=96, J=1)
            elif 'PC' in self.args.dataset:
                self.gnn_obj= gnn_iclr.GNN_nl(args, num_inputs, nf=96, J=1)
            elif 'Houston' in self.args.dataset:
                self.gnn_obj = gnn_iclr.GNN_nl(args, num_inputs, nf=96, J=1)
        else:
            raise NotImplementedError

    def gnn_iclr_forward(self, z, zi_s, labels_yi):
        # Creating WW matrix
        for iqur in range(self.args.query_set):
            zero_pad = Variable(torch.zeros(labels_yi[0].size()))
            if self.args.cuda:
                zero_pad = zero_pad.cuda()
            labels_yi = [zero_pad] + labels_yi


        if self.args.cuda:
            zero_pad = zero_pad.cuda()


        zi_s = z + zi_s

        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]#torch.cat([zi, label_yi], 1)

        nodes = [node.unsqueeze(1) for node in nodes]

        nodes = torch.cat(nodes, 1)


        logits,crossobj = self.gnn_obj(nodes)
        logits=logits.squeeze(-1)

        outputs = F.sigmoid(logits)

        return crossobj, logits


    def forward(self, inputs):
        '''input: [batch_x, [batches_xi], [labels_yi]]'''
        [z, zi_s, labels_yi, oracles_yi, hidden_labels] = inputs

        if 'gnn_iclr' in self.metric_network:
            return self.gnn_iclr_forward(z, zi_s, labels_yi)
        else:
            raise NotImplementedError


class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs)
        else:
            raise(NotImplementedError)


def load_model(model_name, args, io):
    try:
        model = torch.load('checkpoints/%s/models/%s.t7' % (args.exp_name, model_name))
        io.cprint('Loading Parameters from the last trained %s Model' % model_name)
        return model
    except:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None


def create_models(args):
    print (args.dataset)

    if 'UP' == args.dataset:
        enc_nn = Embedding(args, 128,103)
    elif 'IP' == args.dataset:
        enc_nn = Embedding(args, 128,200)
    elif 'salinas' == args.dataset:
        enc_nn = Embedding(args, 128,204)
    elif 'PC' == args.dataset:
        enc_nn = Embedding(args, 128,102)
    elif 'Houston' == args.dataset:
        enc_nn = Embedding(args, 128, 144)
    else:
        raise NameError('Dataset ' + args.dataset + ' not knows')
    return enc_nn, MetricNN(args, emb_size=enc_nn.emb_size)

###############


