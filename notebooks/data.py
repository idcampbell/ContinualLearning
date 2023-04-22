from os.path import join
import torch
from torch.utils.data import Dataset
import utils

class SpatialDataset(Dataset):
    def __init__(self, data_type='standard', n_contexts=100, data_path = '../data/',
                 device=utils.set_torch_device()):
        inputs = torch.load(join(data_path,'spatial_inputs.pt'),map_location=device)
        if data_type=='standard':
            outputs = torch.load(join(data_path,'spatial_outputs.pt'),map_location=device)[:, :n_contexts]
        elif data_type=='narrow':
            outputs = torch.load(join(data_path,'spatial_outputs_narrow.pt'),map_location=device)[:, :n_contexts]
        else:
            raise Exception(f'Unsupported data type: {data_type}. Must be "standard" or "narrow".')
    
        n_locs = inputs.shape[0] # number of (x,y) positions.
        n_contexts = outputs.shape[1]
        context_onehots = torch.eye(n_contexts,device=device).float()
        self.outputs = outputs.T.ravel().reshape(-1,1).float()
        self.inputs = inputs.repeat_interleave(n_contexts, dim=0)
        self.contexts = context_onehots.repeat_interleave(n_locs, dim=0)


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, idx):
        return self.inputs[idx], self.contexts[idx], self.outputs[idx]