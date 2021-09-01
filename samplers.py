import torch 
from torch import nn 
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops as arsl
from torch_sparse import SparseTensor 

import torch_cluster # Required import for the below method to work
random_walk = torch.ops.torch_cluster.random_walk

class SampleConv(nn.Module):
    '''
    Only performs message passing between a random selection of 1-hop
    neighbors. Otherwise equivilant to a GCN conv assuming aggr='mean'
    '''
    def __init__(self, in_feats, out_feats, n_samples=5, aggr='mean', always_sample=False):
        super(SampleConv, self).__init__()

        self.lin = nn.Linear(in_feats, out_feats)
        self.mp = MessagePassing(aggr=aggr)

        self.n = n_samples 
        self.always_sample = always_sample
        self.warnings = set()


    def forward(self, x, ei, edge_weight=None):
        '''
        TODO impliment edge weight; for now, just ignore
        '''
        x = self.lin(x)

        # Only sample during training. Use all edges for evaluation
        if self.training or self.always_sample:
            ei = self.sample(ei)
        
        # Need to invert for propagation if sparse
        if type(ei) == SparseTensor:
            ei = ei.t()

        return self.mp.propagate(ei, x=x, size=None)


    def sample(self, ei, n_samples=None):
        n_samples = self.n if n_samples is None else n_samples 
        batch = torch.tensor(list(range(ei.max())))
        loops = batch.clone()

        # Assumes self loops already in 
        if type(ei) == SparseTensor:
            row, col, _ = ei.csr() 
        
        else:
            ei = arsl(ei)[0]
            row, col, _ = SparseTensor.from_edge_index(ei).csr()
            self.warn('sp', "It is recommended to input edge indices as a torch_sparse.SparseTensor type")

        batch = batch.repeat(n_samples)
        samples = random_walk(row, col, batch, 1, 1., 1.)[0][:, 1]
        
        # Add explicit self-loops to make sure result is 1/|N+1| (n + avg(N(n)))
        batch = torch.cat([batch, loops])
        samples = torch.cat([samples, loops])

        return torch.stack([batch, samples])


    def warn(self, key, msg):
        if key not in self.warnings:
            print(msg)
            self.warnings.add(key)


class SampleMean(SampleConv):
    def __init__(self, n_samples=5, aggr='mean', always_sample=False):
        super().__init__(1,1, n_samples=n_samples, aggr=aggr, always_sample=always_sample)
        
        # Just remove the linear layer and have it take the mean
        # of a random sample of node neighbors
        self.lin = nn.Identity()
        