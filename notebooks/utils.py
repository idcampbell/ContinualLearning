import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
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
    n_rows = max(1,n_contexts//2+n_contexts%2)
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


def plot_representation_pca(reps):
    resolution = int(np.sqrt(reps.shape[0]))
    x,y = np.meshgrid(np.arange(resolution),np.arange(resolution))
    rep_colors = np.zeros((3,resolution,resolution))
    rep_colors[0,:,:] = 255*x/resolution
    rep_colors[1,:,:] = 255*y/resolution
    rep_colors = rep_colors.reshape((3,resolution*resolution)).T.astype(int)
    colors = [f'rgb({str(c[0])},{str(c[1])},{str(c[2])})' for c in rep_colors]

    mds_reps = PCA(2).fit_transform(reps)
    
    fig = go.Figure(
        go.Scatter(
            x=mds_reps[:,0],
            y=mds_reps[:,1],
            mode='markers',
            marker_color=colors
        )
    )
    fig.show()

def plot_receptive_fields(reps):
    resolution, n_channels = reps.shape[1], reps.shape[2]
    n_cols = 16
    n_rows = n_channels//n_cols + int((n_channels%n_cols)>0)

    plot_img = np.empty(((resolution+1)*n_rows,(resolution+1)*n_cols))
    plot_img[:] = np.nan
    for channel_idx in range(n_channels):
        row_idx = channel_idx//n_cols
        col_idx = channel_idx%n_cols
        plot_img[row_idx*(resolution+1):(row_idx+1)*(resolution+1)-1,
                 col_idx*(resolution+1):(col_idx+1)*(resolution+1)-1] = reps[:,:,channel_idx]
    
    fig = px.imshow(plot_img)
    fig.show()