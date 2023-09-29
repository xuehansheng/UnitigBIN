# !/usr/bin/env python
# -*- coding: utf8 -*-

import ast
import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import scipy.sparse as sp
from itertools import product,tee

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

def set_device():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	return device

def save_dict_to_csv(data, filename):
	with open(filename+'.csv', 'w') as file:
		for key, value in data.items():
			file.write(f"{key},{value}\n")

def normalize_adj(adj, self_loop=True):
	"""Symmetrically normalize adjacency matrix."""
	if self_loop:
		adj = adj + sp.eye(adj.shape[0])
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
	"""Convert sparse matrix to tuple representation."""
	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)
	return sparse_mx

def preprocess(graph):
	adj = normalize_adj(graph)
	adj = sparse_to_tuple(adj)
	return adj

def normalise(M):
	d = np.array(M.sum(1))
	di = np.power(d, -1).flatten()
	di[np.isinf(di)] = 0.
	DI = sp.diags(di)	# D inverse i.e. D^{-1}
	return DI.dot(M)

def generate_feature_mapping(kmer_len):
	BASE_COMPLEMENT = {"A":"T","T":"A","G":"C","C":"G"}
	kmer_hash = {}
	counter = 0
	for kmer in product("ATGC",repeat=kmer_len):
		if kmer not in kmer_hash:
			kmer_hash[kmer] = counter
			rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
			kmer_hash[rev_compl] = counter
			counter += 1
	return kmer_hash, counter

def window(seq,n):
	els = tee(seq,n)
	for i,el in enumerate(els):
		for _ in range(i):
			next(el, None)
	return zip(*els)

def compute_composition(seq, kmer_len=4):
	feature_mapping, nr_features = generate_feature_mapping(kmer_len)
	# Create a list containing all kmers, translated to integers
	kmers = [feature_mapping[kmer_tuple] for kmer_tuple in window(str(seq).upper(),kmer_len) if kmer_tuple in feature_mapping]
	kmers.append(nr_features - 1)
	composition_v = np.bincount(np.array(kmers))
	composition_v[-1] -= 1
	# Adding pseudo counts before storing in dict
	composition = composition_v + np.ones(nr_features)
	# Normalize kmer frequencies to remove effect of contig length log(p_ij) = log[(X_ij +1) / rowSum(X_ij+1)]
	# composition = np.log(composition.divide(composition.sum(axis=1),axis=0))
	composition = composition / composition.sum()
	return composition

def read_marker_gene_sets(lineage_file):
	with open(lineage_file, "r") as f:
		lines = f.readlines()
	sets = lines[1].strip().split("\t")[-1]
	sets = ast.literal_eval(sets)
	return sets

def read_contig_genes(contig_markers):
	contigs = {}
	with open(contig_markers, "r") as f:
		for line in f:
			values = line.strip().split("\t")
			contig_name = values[0]
			contig_name = "_".join(contig_name.split("_")[:2])
			contigs[contig_name] = {}
			mappings = ast.literal_eval(values[1])
			for contig in mappings:
				for gene in mappings[contig]:
					if gene not in contigs[contig_name]:
						contigs[contig_name][gene] = 0
					contigs[contig_name][gene] += 1
					if len(mappings[contig][gene]) > 1:
						breakpoint()
	return contigs

def get_markers_to_contigs(marker_sets, contigs):
	marker2contigs = {}
	for marker_set in marker_sets:
		for gene in marker_set:
			marker2contigs[gene] = []
			for contig in contigs:
				if gene in contigs[contig]:
					marker2contigs[gene].append(contig)
	return marker2contigs

if __name__ == '__main__':
	compute_composition('AAAA')
	