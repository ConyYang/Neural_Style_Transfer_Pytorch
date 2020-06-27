import torch.nn as nn

class Content_Loss(nn.Module):
    def __init__(self, graph, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight

        self.graph = graph.detach()*self.weight
        self.criterion = nn.MSELoss()

    def forward(self, content):
        self.loss = self.criterion(content*self.weight, self.graph)
        content_out = content.clone()
        return content_out

    def backward(self, retain_variables = True):
        self.loss.backward(retain_graph = retain_variables)
        return self.loss


# content loss test
