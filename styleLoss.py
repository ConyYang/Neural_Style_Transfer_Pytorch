import torch.nn as nn
import torch

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        # 2d
        feature = input.view(a*b, c*d)
        # gram = feature * transpose
        gram = torch.mm(feature, feature.t())
        # return avg score
        gram /= (a*b*c*d)
        return gram

class Style_Loss(nn.Module):
    def __init__(self, graph, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.graph = graph.detach()*self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, content):
        G = self.gram(content) * self.weight
        self.loss = self.criterion(G, self.graph)
        content_out = content.clone()
        return content_out

    def backward(self, retain_variables = True):
        self.loss.backward(retain_graph = retain_variables)
        return self.loss
