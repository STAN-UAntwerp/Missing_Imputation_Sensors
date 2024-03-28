import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn 
import torch.utils.data
import torch.optim as optim

def data_preprocessing(df_missings,df_ground_truth):
    
    pivot_df_ms = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    pivot_df_gt = df_ground_truth.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    pivot_df_ms = pivot_df_ms.droplevel(level='latitude', axis=1).reset_index(drop=True)
    pivot_df_gt = pivot_df_gt.droplevel(level='latitude', axis=1).reset_index(drop=True)
    pivot_df_ms.columns = [str(col) for col in pivot_df_ms.columns]
    pivot_df_gt.columns = [str(col) for col in pivot_df_gt.columns]
    scaler_ms = MinMaxScaler(feature_range=(0, 1))
    df_ms = scaler_ms.fit_transform(pivot_df_ms)
    df_gt = scaler_ms.fit_transform(pivot_df_gt)
    shape = df_ms.shape
    x = pd.DataFrame(np.vstack(df_ms))
    x.fillna(0, inplace=True)
    df_ms = x.to_numpy()
    df_ms = df_ms.reshape(shape[0], shape[1])
    
    return df_ms,df_gt,scaler_ms

def MIDA_model(df_ms, df_gt,use_cuda = False, num_epochs = 500, batch_size = 1, theta = 7):
    missed_data = torch.from_numpy(df_ms).float()
    train_data = torch.from_numpy(df_gt).float()
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    class Autoencoder(nn.Module):
        def __init__(self, dim):
            super(Autoencoder, self).__init__()
            self.dim = dim
            self.drop_out = nn.Dropout(p=0.5)
            self.encoder = nn.Sequential(
                nn.Linear(dim+theta*0, dim+theta*1),
                nn.Tanh(),
                nn.Linear(dim+theta*1, dim+theta*2),
                nn.Tanh(),
                nn.Linear(dim+theta*2, dim+theta*3)
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim+theta*3, dim+theta*2),
                nn.Tanh(),
                nn.Linear(dim+theta*2, dim+theta*1),
                nn.Tanh(),
                nn.Linear(dim+theta*1, dim+theta*0)
            )
        def forward(self, x):
            x = x.view(-1, self.dim)
            x_missed = self.drop_out(x)
            z = self.encoder(x_missed)
            out = self.decoder(z)
            out = out.view(-1, self.dim)
            return out
        
    model = Autoencoder(dim=df_gt.shape[1]).to(device)
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=0.01, nesterov=True)
    cost_list = []
    early_stop = False
    for epoch in range(num_epochs):
        total_batch = len(train_data) // batch_size
        for i, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            reconst_data = model(batch_data)
            cost = loss(reconst_data, batch_data)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            cost_list.append(cost.item())
        if early_stop :
            break
    model.eval()
    imputed_data = model(missed_data.to(device))
    imputed_data = imputed_data.cpu().detach().numpy()
    return imputed_data,model
    
def MIDA_imputation(df_missings,df_ground_truth,df_gt, use_cuda = False, **kwargs):
    num_epochs = kwargs.get('num_epochs', 100)
    batch_size = kwargs.get('batch_size', 1)
    theta = kwargs.get('theta', 7)
    df_ms,df_gt_all,scaler_ms = data_preprocessing(df_missings,df_gt)
    imputed,model = MIDA_model(df_ms, df_gt_all, use_cuda, num_epochs, batch_size, theta)
    imputed = scaler_ms.inverse_transform(imputed)
    imputed_df = pd.DataFrame(imputed)
    pivot_df_ms = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    imputed_df.columns = pivot_df_ms.columns
    imputed_df.index = pivot_df_ms.index
    unpivoted_df = imputed_df.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')

    return unpivoted_df