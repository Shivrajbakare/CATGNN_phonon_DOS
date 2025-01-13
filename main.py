import os
import yaml
import numpy as np

import pandas as pd
from tqdm import tqdm
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

from jarvis.core.atoms import Atoms

#from process_data import *
from process_data import process_data_and_split, get_train_valid_test_dataloaders


f = open('model_params.yaml')
model_params = yaml.safe_load(f)
f.close()
data_dir=os.path.join(os.getcwd(), 'data')
dataset, idx_train, idx_valid, idx_test=process_data_and_split(filename='id_prop.csv', data_dir=data_dir, structure_format=model_params['structure_format'],
									atom_feats_type = model_params['atom_feats_type'],
									atom_attrs_type = model_params['atom_attrs_type'],
									valid_size = model_params['validSplit'], test_size = model_params['testSplit'], plot_train_valid_test_split=False,
									cutoff_r=model_params['max_radius'], 
									edge_lmax=model_params['lmax'], 
									edge_number_of_basis=model_params['number_of_basis'], 
									edge_basis_type=model_params['basis_type'],
									edge_cutoff_basis=model_params['cutoff_basis'],
									)
if model_params['atom_feats_type'] == 'cgcnn' and model_params['atom_attrs_type'] == 'cgcnn':
	model_params['in_dim']=92

def get_neighbors(df, idx):
	n = []
	for entry in df.iloc[idx].itertuples():
		N = entry.data.pos.shape[0]
		for i in range(N):
			n.append(len((entry.data.edge_index[0] == i).nonzero()))
	return np.array(n)

df=pd.DataFrame.from_dict(dataset)
n_train = get_neighbors(df, idx_train)
n_valid = get_neighbors(df, idx_valid)
n_test = get_neighbors(df, idx_test)

model_params['num_neighbors']=n_train.mean()
	
train_dataloader, valid_dataloader, test_dataloader = get_train_valid_test_dataloaders(dataset, idx_train, idx_valid, idx_test, batch_size=model_params['batch_size'])
	
################## processing data completed ################## 

################## defining model ################## 

from CATGNN.model import PeriodicNetwork
import torch
import torch_geometric
#from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

import torch.nn as nn
#from torch_geometric.nn.conv  import MessagePassing
import torch.nn.functional as F
#import torch_scatter
	
#from torch_geometric.nn.inits import glorot, zeros
#from torch_geometric.utils import softmax
	
#from e3nn import o3
from typing import Union, Optional, Dict

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


model = PeriodicNetwork(
	in_dim=model_params['in_dim'], 
	em_dim=model_params['em_dim'], 
	out_dim=model_params['out_dim'], 
	edge_dim=model_params['number_of_basis'],
	target_dim=model_params['target_dim'],
	n_GAT_layers=model_params['n_GAT_layers'],
	nonlinear_post_scatter=False,
	
#	irreps_in=model_params['irreps_in'],
	irreps_in=str(model_params['em_dim'])+"x0e",
#	irreps_out=model_params['irreps_out'],
	irreps_out=str(model_params['em_dim'])+"x0e",
#	irreps_node_attr=model_params['irreps_node_attr'],
	irreps_node_attr=str(model_params['em_dim'])+"x0e",
	layers=model_params['n_conv_layers'],
	number_of_basis = model_params['number_of_basis'],
	basis_type=model_params['basis_type'],
	mul=model_params['mul'],
	lmax=model_params['lmax'],
	max_radius=model_params['max_radius'],
	num_neighbors=model_params['num_neighbors'],
#	reduce_output=False
	)

device =torch.device('cude' if torch.cuda.is_available() else 'cpu')
net=model.to(device)

################## model training ################## 

def output_test_loader_results(test_loader, net, device, filename='test_set_props.csv'):
	ids=np.array([])
	labels, preds = torch.tensor([]).to(device), torch.tensor([]).to(device)

	#print(model(dataset[0]['data']))
	for i, mat in enumerate(test_loader):
		mat=mat.to(device)
		#print(mat)
		#print(torch.tensor(mat.mat_id))
		ids=np.concatenate([ids, mat.mat_id], axis=0)
		net.eval()
		with torch.no_grad():
			pred=net(mat).float()
		
		labels=torch.cat([labels, mat.prop.float()], dim=0)
		preds=torch.cat([preds, pred], dim=0)
		#print(mat)
		#print(model(mat))

	mae_loss = torch.nn.L1Loss()
	mse_loss = torch.nn.MSELoss()
	mae_loss_test=mae_loss(labels, preds)
	mse_loss_test=mse_loss(labels, preds)

	print('mae_loss:', mae_loss_test.detach().item())
	print('mse_loss:', mse_loss_test.detach().item())
	ids = ids.tolist()
	labels=labels.cpu().tolist()
	preds = preds.cpu().tolist()

	df_test_set=pd.DataFrame.from_dict({'id': ids, 'true props': labels, 'pred props': preds})
	df_test_set.to_csv(filename, index=False)


import time
from Trainer import train, train_val_loss_plot
from plot_predictions import plot_predictions

output_dir=os.path.join(os.getcwd(), 'results')

optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-5)
import torch.optim.lr_scheduler as lr_scheduler
scheduler = lr_scheduler.LinearLR(optimizer , start_factor=1.0, end_factor=0.05, total_iters=100)
#scheduler=None

loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()

run_name = 'model_' + time.strftime("%y%m%d", time.localtime())
train(model, optimizer, train_dataloader, valid_dataloader, loss_fn, loss_fn_mae, run_name, max_iter=model_params['num_epochs'], 
	scheduler=scheduler, device=device, output_dir=output_dir, patience=int(model_params['patience']))

train_val_loss_plot(output_dir, device, run_name, filename='history.jpg', dpi=1000)

new_model=model
new_model.load_state_dict(torch.load(os.path.join(output_dir, run_name+'_best_model.pth')))
output_test_loader_results(test_loader=test_dataloader, net=new_model, device=device, filename=os.path.join(output_dir, 'test_set_props.csv'))


dataloader = DataLoader(df['data'].values, batch_size=256)
df['mse'] = 0.
df['pred'] = np.empty((len(df), 0)).tolist()

new_model.to(device)
new_model.eval()
with torch.no_grad():
	i0 = 0
	for i, d in tqdm(enumerate(dataloader), total=len(dataloader), bar_format=bar_format):
		d.to(device)
		output = new_model(d)
		loss = F.mse_loss(output, d.prop, reduction='none').mean(dim=-1).cpu().numpy()
		df.loc[i0:i0 + len(d.prop) - 1, 'pred'] = [[k] for k in output.cpu().numpy()]
		df.loc[i0:i0 + len(d.prop) - 1, 'mse'] = loss
		i0 += len(d.prop)
        
df['pred'] = df['pred'].map(lambda x: x[0].tolist())
df.to_csv('data_preds.csv', index=False)
plot_predictions(df, idx_test, 'Testing')
