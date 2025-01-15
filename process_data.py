import os
import numpy as np
import math
import itertools
from typing import Union

import pandas as pd 
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

# jarvis tools and ase
from jarvis.core.atoms import Atoms, ase_to_atoms #as j_atoms
from jarvis.core.specie import Specie
#from ase import Atom, Atoms
from ase import Atoms as ase_atoms 
from ase.io import read # Atoms

# torch 
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

# torch e3nn 
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
#from e3nn.nn.models.gate_points_2101 import smooth_cutoff


# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# colors with dataset types
palette = ['#285fb2', '#f3b557', '#67c791', '#c85c46']
dataset_types = ['train', 'valid', 'test']
colors = dict(zip(dataset_types, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])


def smooth_cutoff(x):
	u = 2 * (x - 1)
	y = (math.pi * u).cos().neg().add(1).div(2)
	y[u > 0] = 0
	y[u < -1] = 1
	return y

# outputs one hot encoding atomic mass, one hot encoding atomic number, "type_encoding" dictionary of atomic numbers
def obtain_features():
	type_encoding={}
	an_type=[]
	element_mass=[]
	with open('elements.txt') as f:
		text=f.read().splitlines()
	for i, element in enumerate(text):
		type_encoding[element]=Specie(element).Z-1
		an_type.append(Specie(element).Z-1)
		element_mass.append(Specie(element).atomic_mass)
	#print(an_type)
	atom_type_onehot = torch.eye(len(an_type))
	#specie_onehot = torch.eye(len(type_encoding))
	mass_onehot = torch.diag(torch.tensor(element_mass))
	
	return mass_onehot, atom_type_onehot, type_encoding

# outputs one	
class composition_feats_encoder:
	def __init__(self):

		with open('elements.txt') as f:
			text=f.read().splitlines()
		self.elements=list(text)
		self.e_arr = np.array(self.elements)
		
	def encode(self, composition_dict):
		encoded_feats   = [0]*len(self.elements)
		counts   = [j for j in composition_dict.values()]
		total    = sum(counts)

		for idx in range(len(composition_dict.keys())):
			element = list(composition_dict.keys())[idx]
			ratio  = counts[idx]/total
			idx_e  = self.elements.index(element)
			encoded_feats[idx_e] = ratio
		return torch.tensor(encoded_feats).float().view(1,-1)
		
# gets neighbors for each atom, outputs neighbors, edges sources and destination, edges distances, and edges shifts	
def get_all_neighbors(
			atoms, 
			r=5, 
			bond_tol=0.0
			):
	"""
	Get neighbors for each atom in the unit cell, out to a distance r.

	Contains [index_i, index_j, distance, image] array.
	Adapted from pymatgen.
	"""
	recp_len = np.array(atoms.lattice.reciprocal_lattice().abc)
	maxr = np.ceil((r + bond_tol) * recp_len / (2 * math.pi))
	nmin = np.floor(np.min(atoms.frac_coords, axis=0)) - maxr
	nmax = np.ceil(np.max(atoms.frac_coords, axis=0)) + maxr
	all_ranges = [np.arange(x, y) for x, y in zip(nmin, nmax)]
	matrix = atoms.lattice_mat
	neighbors = [list() for _ in range(len(atoms.cart_coords))]
	# all_fcoords = np.mod(self.frac_coords, 1)
	coords_in_cell = atoms.cart_coords  # np.dot(all_fcoords, matrix)
	site_coords = atoms.cart_coords
	indices = np.arange(len(site_coords))
	
	src_edges=[]; dst_edges=[]; distance_edges=[]; shift_edge=[]
	for image in itertools.product(*all_ranges):
		coords = np.dot(image, matrix) + coords_in_cell
		z = (coords[:, None, :] - site_coords[None, :, :]) ** 2
		all_dists = np.sum(z, axis=-1) ** 0.5
		all_within_r = np.bitwise_and(all_dists <= r, all_dists >= 0.0)
		for j, d, within_r in zip(indices, all_dists, all_within_r):
			for i in indices[within_r]:
				if d[i] >= bond_tol:
					# if d[i] > bond_tol and i!=j:
					neighbors[i].append([i, j, d[i], image])
					src_edges.append(i)
					dst_edges.append(j)
					distance_edges.append(d[i])
					shift_edge.append(image)
					
					#if i==2: print(i, j, d[i], image)
	#print(type(neighbors[0][:][:][0]))
	return np.asarray(neighbors, dtype="object"), src_edges, dst_edges, distance_edges, shift_edge
	

# split the materials indices based on elements according to validation and testing ratio sizes
def split_elements_indices(
				df, 
				valid_size, 
				test_size, 
				seed
				):
	# initialize output arrays
	idx_train, idx_valid, idx_test = [], [], []
    
	# remove empty examples
	df = df[df['data'].str.len()>0]
    
	# sort df in order of fewest to most examples
	df = df.sort_values('count')
    
	for _, entry in tqdm(df.iterrows(), total=len(df), bar_format=bar_format):
		df_specie = entry.to_frame().T.explode('data')

		try:
			idx_train_s, idx_valid_test_s = train_test_split(df_specie['data'].values, test_size=test_size+valid_size,
															random_state=seed)
			idx_valid_s, idx_test_s = train_test_split(idx_valid_test_s, test_size=test_size/(valid_size+test_size),
														random_state=seed)
															
		except:
			# too few examples to perform split - these examples will be assigned based on other constituent elements
			# (assuming not elemental examples)
			print('too few')
			pass

		else:
			# add new indices that do not exist in previous lists
			idx_train += [k for k in idx_train_s if k not in idx_train + idx_valid + idx_test]
			idx_valid += [k for k in idx_valid_s if k not in idx_train + idx_valid + idx_test]
			idx_test += [k for k in idx_test_s if k not in idx_train + idx_valid + idx_test]
			
	return idx_train, idx_valid, idx_test


# get fraction of number of materials samples containing a given element from the entire dataset
def elements_representation(
				all_element_idx, 
				partial_element_idx
				):
	
	return len([idx for idx in all_element_idx if idx in partial_element_idx])/len(all_element_idx)
	

# create dictionary indexed by element names storing index of samples containing given element	
def elements_stats(
			dataset, 
			species
			):
	
	species_dict = {specie: [] for specie in species}
	for i, entry in enumerate(dataset):
		for specie in entry['species']:
			species_dict[specie].append(i)
	#print(species_dict)
		
	# create a dataframe of element statistics
	stats = pd.DataFrame({'symbol': species})
	stats['data'] = stats['symbol'].astype('object')
	#print(stats)
		
	for specie in species:
		stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
	stats['count'] = stats['data'].apply(len)
	#print(stats)
	return stats


# plot elements representation
def split_subplot(
		ax, 
		df, 
		species, 
		dataset, 
		bottom=0., 
		legend=False
		):    
	
	width = 0.4
	color = [int(colors[dataset].lstrip('#')[i:i+2], 16)/255. for i in (0,2,4)]
	bx = np.arange(len(species))
        
	ax.bar(bx, df[dataset], width, fc=color+[0.7], ec=color, lw=1.5, bottom=bottom, label=dataset)
        
	ax.set_xticks(bx)
	ax.set_xticklabels(species)
	ax.tick_params(direction='in', length=0, width=1)
	ax.set_ylim(top=1.18)
	if legend: ax.legend(frameon=False, ncol=3, loc='upper left')


# perform an element-balanced train/valid/test split	
def train_valid_test_element_balanced_split(
						dataset, 
						species, 
						valid_size, 
						test_size, 
						seed=123, 
						plot_train_valid_test_split=False
						):
										
	print('split train/valid/test datasets ...')
	stats = elements_stats(dataset, species)
	idx_train, idx_valid, idx_test = split_elements_indices(stats, valid_size, test_size, seed)
		
	df=pd.DataFrame.from_dict(dataset)
	idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

	print('number of materials in training dataset:', len(idx_train))
	print('number of materials in validation dataset:', len(idx_valid))
	print('number of materials in testing dataset:', len(idx_test))
	print('total number of materials:', len(idx_train + idx_valid + idx_test))
	assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0

	if plot_train_valid_test_split:
		# plot element representation in each dataset
		stats['train'] = stats['data'].map(lambda x: elements_representation(x, sorted(idx_train)))
		stats['valid'] = stats['data'].map(lambda x: elements_representation(x, sorted(idx_valid)))
		stats['test'] = stats['data'].map(lambda x: elements_representation(x, sorted(idx_test)))
		stats = stats.sort_values('symbol')

		fig, ax = plt.subplots(2,1, figsize=(14,7))
		b0, b1 = 0., 0.
		for i, dataset_type in enumerate(dataset_types):
			split_subplot(ax[0], stats[:len(stats)//2], species[:len(stats)//2], dataset_type, bottom=b0, legend=True)
			split_subplot(ax[1], stats[len(stats)//2:], species[len(stats)//2:], dataset_type, bottom=b1)

			b0 += stats.iloc[:len(stats)//2][dataset_type].values
			b1 += stats.iloc[len(stats)//2:][dataset_type].values

		fig.tight_layout()
		fig.subplots_adjust(hspace=0.1)
		fig.savefig('train_valid_test_split_plot.jpg', dpi=500)

	return idx_train, idx_valid, idx_test


# get torch geometric (tg) data through Data class
def get_tg_Data(
		jarvis_atoms, 
		mass_onehot,
		atom_type_onehot,
		type_encoding,
		comp_feats,
		freq,
		prop,
		atom_feats_type: Union[str, None],
		atom_attrs_type: Union[str, None],
		cutoff_r=4,
		edge_lmax=1, 
		edge_number_of_basis=31, 
		edge_basis_type='gaussian',
		edge_cutoff_basis: bool=False,
		):
	
	nei, src_edges, dst_edges, distance_edges, shift_edge=get_all_neighbors(jarvis_atoms, r=cutoff_r)
	symbols=list(jarvis_atoms.elements).copy()
	positions=torch.from_numpy(jarvis_atoms.cart_coords.copy())
	lattice=torch.from_numpy(jarvis_atoms.lattice_mat.copy()).unsqueeze(0)
	edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(np.asarray(src_edges))]
	edge_vec = (positions[torch.from_numpy(np.asarray(dst_edges))]
				- positions[torch.from_numpy(np.asarray(src_edges))]
				+ torch.einsum('ni,nij->nj', torch.tensor(shift_edge, dtype=default_dtype), lattice[edge_batch]))

	irreps_edge_attr = o3.Irreps.spherical_harmonics(edge_lmax)
	edge_sh = o3.spherical_harmonics(irreps_edge_attr, edge_vec, True, normalization='component')
	edge_length = edge_vec.norm(dim=1)
	edge_length_embedded = soft_one_hot_linspace(
			x=edge_length,
			start=0.0,
			end=cutoff_r,
			number=edge_number_of_basis,
			basis=edge_basis_type,
			cutoff=edge_cutoff_basis,
			).mul(edge_number_of_basis**0.5)
	edge_attrs = smooth_cutoff(edge_length / cutoff_r)[:, None] * edge_sh


	if atom_feats_type=='cgcnn':
		from jarvis.core.specie import get_node_attributes
		sps_features = []
		for ii, s in enumerate(symbols):
			feat = list(get_node_attributes(s, atom_features='cgcnn'))
			if atom_attrs_type != None and atom_attrs_type != 'cgcnn':
				feat.extend(11*[0]); 
			sps_features.append(feat)
		sps_features = np.array(sps_features)
		#exit()
		x = torch.Tensor(sps_features).type(default_dtype)
	elif atom_feats_type=='mass':
		x = mass_onehot[[type_encoding[specie] for specie in symbols]]
	elif atom_feats_type=='z':
		x = atom_type_onehot[[type_encoding[specie] for specie in symbols]]
	elif atom_feats_type==None:
		print('you selected None for atom features which cannot happen. Therefore you will be forced to use one hot encoding atomic mass')
		x = mass_onehot[[type_encoding[specie] for specie in symbols]]
	else:
		raise ValueError('you can select one of the following: "cgcnn", "mass", "z" or None')

	if atom_attrs_type=='cgcnn':
		sps_features = []
		for ii, s in enumerate(symbols):
			feat = list(get_node_attributes(s, atom_features='cgcnn'))
			if atom_feats_type != 'cgcnn':
				feat.extend(11*[0]); 
			sps_features.append(feat)
		sps_features = np.array(sps_features)
		z = torch.Tensor(sps_features).type(default_dtype)
	elif atom_attrs_type=='mass':
		z = mass_onehot[[type_encoding[specie] for specie in symbols]]
	elif atom_attrs_type=='z':
		z = atom_type_onehot[[type_encoding[specie] for specie in symbols]]
	elif atom_attrs_type==None:
		z=None
	else:
		raise ValueError('you can only select one of the following: "cgcnn", "mass", "z" or None')


	data = Data(
		pos=positions, lattice=lattice, symbol=symbols,
		z = z, #atom_type_onehot[[type_encoding[specie] for specie in symbols]],
		x = x, #mass_onehot[[type_encoding[specie] for specie in symbols]], 
		comp_feats=comp_feats, 
		edge_index=torch.stack([torch.LongTensor(src_edges), torch.LongTensor(dst_edges)], dim=0),
		edge_shift=torch.tensor(shift_edge, dtype=default_dtype),
		edge_vec=edge_vec,
		edge_attrs=edge_attrs,
		edge_length_embedded=edge_length_embedded,
		freq=torch.tensor(freq, dtype=default_dtype).unsqueeze(0),
		prop=torch.tensor(prop, dtype=default_dtype).unsqueeze(0),
		)
	
	return data


# reads the csv file, processes data, splits data into training, validation, and testing sets
def process_data_and_split(
				filename='id_prop.csv', 
				data_dir=os.path.join(os.getcwd(), 'data'), 
				structure_format='cif',
				valid_size = 0.15, 
				test_size = 0.1, 
				output_split_indices: bool = False,

				atom_feats_type = 'mass',
				atom_attrs_type = 'z',

				plot_train_valid_test_split: bool = False,
				cutoff_r=4, 
				edge_lmax=1, 
				edge_number_of_basis=31, 
				edge_basis_type='gaussian',
				edge_cutoff_basis: bool=False,
				):

	mass_onehot, atom_type_onehot, type_encoding=obtain_features()
	encoder_composition = composition_feats_encoder()

	df=pd.read_csv(filename)
	df['id']=df.id.apply(str)
	data_dir=data_dir
	
	dataset_dict = []
	species = []
	for index, entry in tqdm(df.iterrows(), total=len(df), bar_format=bar_format):
		info={}
		
		if structure_format.lower() == 'cif':
			#jarvis_atoms=Atoms.from_cif(os.path.join(data_dir, str(entry.id)+'.cif'), get_primitive_atoms=False, use_cif2cell=False)
			ase_atoms = read(os.path.join(data_dir, str(entry.id)+'.cif'))
			jarvis_atoms = ase_to_atoms(ase_atoms)
		elif structure_format.lower() == 'poscar' or structure_format.lower() == 'vasp':
			jarvis_atoms=Atoms.from_poscar(os.path.join(data_dir, str(entry.id)+'.POSCAR'))
		else:
			raise ValueError("please select one of the following options 'poscar', 'vasp' or 'cif'")

		#print(jarvis_atoms)

		info['id']=str(entry.id)
			
		info['structure']=jarvis_atoms.to_dict()
		info['formula']=jarvis_atoms.composition.formula
		info['species']=list(set(jarvis_atoms.elements))
		species.extend(list(set(jarvis_atoms.elements)))

		info['freq']=eval(entry.freq)
		info['prop']=eval(entry.prop)
		
		composition_features = encoder_composition.encode(jarvis_atoms.composition._content)
		
		tg_data=get_tg_Data(jarvis_atoms,
				mass_onehot=mass_onehot,
				atom_type_onehot=atom_type_onehot,
				type_encoding=type_encoding,
				comp_feats=composition_features,
				freq=eval(entry.freq),
				prop=eval(entry.prop), 

				atom_feats_type = atom_feats_type,
				atom_attrs_type = atom_attrs_type,

				cutoff_r=cutoff_r,
				edge_lmax=edge_lmax, 
				edge_number_of_basis=edge_number_of_basis, 
				edge_basis_type=edge_basis_type,
				edge_cutoff_basis=edge_cutoff_basis,)
		tg_data.mat_id=str(entry.id)
		
		info['data']=tg_data

		dataset_dict.append(info)
			
	species=sorted(list(set(species)))
	#print(len(set(species)), sorted(list(set(species))))
	idx_train, idx_valid, idx_test=train_valid_test_element_balanced_split(dataset=dataset_dict, species=species, valid_size = valid_size, test_size = test_size, plot_train_valid_test_split=plot_train_valid_test_split)
	
	return dataset_dict, idx_train, idx_valid, idx_test


# creates dataloaders for training, validation, and testing datasets 
def get_train_valid_test_dataloaders(
								dataset, 
								idx_train, 
								idx_valid, 
								idx_test, 
								batch_size=32
								):
	
	df=pd.DataFrame.from_dict(dataset)
	train_dataloader=DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
	valid_dataloader=DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size, shuffle=True)
	test_dataloader=DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size, shuffle=False)
	
	return train_dataloader, valid_dataloader, test_dataloader

