import plotly.express as px
from plotly.subplots import make_subplots
import torch

def set_torch_device(use_gpu=True):
    if not use_gpu:
        return torch.device('cpu')
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def plot_predictions(train_y, model_predictions, n_contexts):
    n_rows = max(1,n_contexts//2)
    fig = make_subplots(rows=n_rows,cols=4,row_heights=[1]*n_rows,column_widths=[1,1,1,1],
                        subplot_titles=('True','Predicted','True','Predicted'))
    for context_idx in range(n_contexts):
        fig.add_trace(px.imshow(train_y[context_idx]).data[0],row=context_idx//2+1,col=1+2*(context_idx%2))
        fig.add_trace(px.imshow(model_predictions[context_idx]).data[0],row=context_idx//2+1,col=2+2*(context_idx%2))
    fig.update_layout(height=300*n_rows,width=300*4)
    fig.show()

def plot_contexts(context_data):
    n_rows = max(1,context_data.shape[0]//4+ int((context_data.shape[0]%4)>0))
    fig = make_subplots(rows=n_rows,cols=4)
    for context_idx in range(context_data.shape[0]):
        fig.add_trace(px.imshow(context_data[context_idx]).data[0],row=context_idx//4+1,col=1+context_idx%4)
    fig.update_layout(height=300*n_rows,width=300*4)
    fig.show()