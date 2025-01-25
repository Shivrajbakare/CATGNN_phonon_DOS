import numpy as np

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

import torch.nn as nn

import torch.nn.functional as F
import torch_scatter
	
#from torch_geometric.nn.inits import glorot, zeros
#from torch_geometric.utils import softmax
	
#from e3nn import o3
from typing import Union, Optional, Dict


from CATGNN.conv_layer import Network
from CATGNN.CAT_layers import Elements_Attention, MHA_CAT, find_activation

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

			
class PeriodicNetwork(Network):
	def __init__(self, 
		in_dim, 
		em_dim, 
		out_dim, 
		edge_dim, 
		target_dim, 
		n_GAT_layers: int=3, 
		nonlinear_post_scatter: bool=True, 
		**kwargs
		):            
            
		super().__init__(**kwargs)

		# embed the mass-weighted one-hot encoding
		self.em = nn.Linear(in_dim, em_dim)
		self.emx = nn.Linear(in_dim, em_dim)
		self.emz = nn.Linear(in_dim, em_dim)
			
		# embed the gaussian basis of bond length
		self.edge_em = nn.Linear(edge_dim, out_dim)
		#self.edge_em = nn.Linear(edge_dim, edge_dim)
			
		self._activation = find_activation('softplus') #torch.nn.SiLU()

		self.GAT=nn.ModuleList([MHA_CAT(input_dim=out_dim, output_dim=out_dim, edge_dim=out_dim, heads=8, GAT_implement=False) for i in range(n_GAT_layers)])
		self.batch_norm = nn.ModuleList([nn.BatchNorm1d(out_dim) for i in range(n_GAT_layers)])

			
		self.E_R_Atten = Elements_Attention(out_dim)
			
		self.nonlinear_post_scatter=nonlinear_post_scatter
			
		self.linear1 = nn.Linear(out_dim, out_dim)
		self.batch_norm1=nn.BatchNorm1d(out_dim)
		self.linear2 = nn.Linear(out_dim, out_dim)
		self.batch_norm2=nn.BatchNorm1d(out_dim)
		self.out=nn.Linear(out_dim, target_dim)

	def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
		num_graphs=data["ptr"].numel() - 1
		cgcnn_feats=data.x

		data.x = F.relu(self.emx(data.x))
		data.z = F.relu(self.emz(data.z))

		output = super().forward(data)
		
		output = torch.relu(output)
		
		edge_length_embedded = self._activation(self.edge_em(data.edge_length_embedded))
		pre_output=output
		for a_idx in range(len(self.GAT)):
			output=self.GAT[a_idx](data.edge_index, output, edge_length_embedded = edge_length_embedded)
			output = self.batch_norm[a_idx](output)
			#output = F.softplus(output)
			output = F.relu(output)
			output = torch.add(output, pre_output)
			pre_output=output
		
		ag = self.E_R_Atten(output, data.batch, cgcnn_feats)

		output = torch.add(output, (output)*ag)
		y = torch_scatter.scatter_mean(src=output, index=data.batch, dim=0,)# dim_size=num_graphs)
			
		if self.nonlinear_post_scatter:
			#y = self._activation(y)
			y=torch.relu(self.batch_norm1(self.linear1(y)))
			#y = self._activation(y)
			y=torch.relu(self.batch_norm2(self.linear2(y)))
			#y = self._activation(y)
		maxima, _ = torch.max(y, dim=1)
		y = y.div(maxima.unsqueeze(1))

		return y

