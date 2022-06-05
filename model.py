from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import torch


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(torch.nn.Module):
    def __init__(self,num_feature,num_hidden, output_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(num_feature, num_hidden)
        self.conv2 = GraphConvolution(num_hidden, output_size)
        
    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.conv2(x, adj)
        return x
        
class LinkPredictor(torch.nn.Module):
    def __init__(self, num_hidden):
        super(LinkPredictor, self).__init__()
        self.linear = torch.nn.Linear(num_hidden, num_hidden)
        self.adjust = torch.nn.Linear(num_hidden, 1)
        self.drop   = torch.nn.Dropout(0.3)
#        self.temp   = nn.Parameter(torch.ones(1))
    def forward(self, x):
        x = torch.matmul(x, torch.transpose(self.linear(x), 0, 1))
        return F.sigmoid(x)

class LP_Generator(torch.nn.Module):
    def __init__(self, num_feature,num_hidden,output_size):
        super(LP_Generator, self).__init__()
        self.conv1 = GraphConvolution(num_feature, num_hidden)
        self.conv2 = GraphConvolution(num_hidden, output_size)
        self.linear = torch.nn.Linear(output_size, output_size)
        
    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.conv2(x, adj)
        x = torch.matmul(x, torch.transpose(self.linear(x), 0, 1))
        return F.sigmoid(x)
    
class Adaptive_LP(torch.nn.Module):
    def __init__(self, num_feats, num_hidden):
        super(Adaptive_LP, self).__init__()
        self.nodevec1 = torch.nn.Parameter(torch.randn(num_feats, num_hidden), requires_grad=True)
        self.nodevec2 = torch.nn.Parameter(torch.randn(num_hidden, num_feats), requires_grad=True)
    def forward(self):
        adp = F.sigmoid(torch.mm(self.nodevec1, self.nodevec2))
        return adp






