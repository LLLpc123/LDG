import torch
import torch.nn as nn
import numpy as np

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        # print('Orthogonal pretrainer loss: %.2e' % loss)
        pass
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation

        self.reset_parameters()

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

    def reset_parameters(self):
        W = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.data.copy_(torch.from_numpy(W))

        b = np.zeros(self.hidden_size, dtype=np.float32)
        self.linear.bias.data.copy_(torch.from_numpy(b))
class biaffine_layer(nn.Module):
    def __init__(self,cfg, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(biaffine_layer, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                  bias=False)
        
        self.activ = cfg.activ_in_biaffine
    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        if self.activ is not None:
            biaffine = self.activ(biaffine)
        return biaffine

class biaffine(nn.Module):
    def __init__(self, cfg):
        super(biaffine, self).__init__()
        self.cfg = cfg
        self.mlp_arc_dep = NonLinear(
            cfg.hidden_dim, cfg.hidden_dim, activation=nn.GELU())
        self.mlp_arc_head = NonLinear(
            cfg.hidden_dim, cfg.hidden_dim, activation=nn.GELU())
        self.mlp_rel_dep = NonLinear(
            cfg.hidden_dim, cfg.hidden_dim, activation=nn.GELU())
        self.mlp_rel_head = NonLinear(
            cfg.hidden_dim, cfg.hidden_dim, activation=nn.GELU())
        self.arc_biaffine = biaffine_layer(cfg,
            cfg.hidden_dim, cfg.hidden_dim, 1, bias=(True, False))
        self.rel_biaffine = biaffine_layer(cfg,
            cfg.hidden_dim, cfg.hidden_dim, 45, bias=(True, True))
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, hidden_states):

        x_arc_dep = self.mlp_arc_dep(hidden_states)
        x_arc_head = self.mlp_arc_head(hidden_states)
        x_rel_dep = self.mlp_rel_dep(hidden_states)
        x_rel_head = self.mlp_rel_head(hidden_states)

        arc_score = self.arc_biaffine(
            self.drop(x_arc_dep), self.drop(x_arc_head)).squeeze(-1)
        rel_score = self.rel_biaffine(
            self.drop(x_rel_dep), self.drop(x_rel_head))
        
        return arc_score,rel_score

    def get_rel(self,arc_score,rel_score):
        arc_pred = torch.argmax(arc_score,dim = -1)
        rel_score = rel_score
        return torch.gather(rel_score,2,arc_pred.unsqueeze(2).unsqueeze(3).expand(-1,-1,-1,45))