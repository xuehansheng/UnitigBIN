# !/usr/bin/env python
# -*- coding: utf8 -*-

import torch,sys
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import set_device,sparse_to_tuple,normalize_adj
from modules import MultiVGAE,ConstraintMatching,ConstraintBinning,ConstraintProcessing
import scipy.sparse as sp
from collections import Counter,defaultdict
import gc
gc.collect()
torch.cuda.empty_cache()

class UnitigBIN:
	def __init__(self, ipt_dim, args):
		self.args = args
		self.ipt_dim = ipt_dim
		self.device = set_device()
		self.model = MultiVGAE(ipt_dim, args.hid_dim*2, args.hid_dim, args, self.device)
		self.model.to(self.device)

	def PPRDiffusionConv_Torch(self, A, alpha, eps):
		# print('--- PPR-based Diffusion Convolution (Torch) ---')
		N = A.shape[0]
		A_loop = torch.tensor(torch.eye(N)+A,dtype=torch.float).to(self.device) ## Self-loops
		D_loop_vec = torch.sum(A_loop, 0)
		D_loop_vec_invsqrt = 1 / torch.sqrt(D_loop_vec)
		D_loop_invsqrt = torch.diag(D_loop_vec_invsqrt)
		DA = torch.matmul(D_loop_invsqrt, A_loop)
		T_sym = torch.matmul(DA, D_loop_invsqrt)
		I = torch.eye(N).to(self.device)
		S = alpha * torch.linalg.inv(I - (1 - alpha) * T_sym)
		S_tilde = torch.multiply(S, torch.ge(S, eps))
		D_tilde_vec = torch.sum(S_tilde, 0)
		T_S = S_tilde / D_tilde_vec
		return T_S.to_sparse()

	def learning(self, data):
		# print("# Params:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
		V = torch.tensor(data.edge_index[0]).to(self.device)
		E = torch.tensor(data.edge_index[1]).to(self.device)
		R = torch.tensor(data.batchReads[0]).to(self.device)
		I = torch.tensor(data.batchReads[1]).to(self.device)
		triplets = data.triplets
		batches = data.batches #[readList,readAdj,readFeat]
		pairs = torch.tensor(data.readsPair).to(self.device)

		adj_norm,adj_labels,features = [],[],[]
		adj_convs = []
		norms,weight_tensors = [],[]
		for idx,batch in batches.items(): #batch:[readList,readAdj,readFeat,readAdjFull]
			diffConv = self.PPRDiffusionConv_Torch(batch[1].toarray(), self.args.alpha, self.args.eps)
			adj_norm.append(diffConv)   #diffusion
			adj = sparse_to_tuple(batch[1])#sparse_to_tuple(sp.csr_matrix(batch[1]))
			adjTensor = torch.sparse.FloatTensor(torch.LongTensor(adj[0].T),torch.FloatTensor(adj[1]),torch.Size(adj[2])).to(self.device)
			adj_labels.append(adjTensor) #adjacency
			feat = sparse_to_tuple(sp.csr_matrix(np.concatenate((batch[1].toarray(),batch[2].toarray()),axis=1)))
			featTensor = torch.sparse.FloatTensor(torch.LongTensor(feat[0].T),torch.FloatTensor(feat[1]),torch.Size(feat[2])).to(self.device)
			features.append(featTensor) #features

			pos_weight = float(batch[1].shape[0]*batch[1].shape[0]-batch[1].sum())/batch[1].sum()
			weight_mask = adjTensor.to_dense().view(-1) == 1
			weight_tensor = torch.ones(weight_mask.size(0)).to(self.device)
			weight_tensor[weight_mask] = pos_weight
			weight_tensors.append(weight_tensor)
			norm = batch[1].shape[0]*batch[1].shape[0]/float((batch[1].shape[0]*batch[1].shape[0]-batch[1].sum())*2)
			norms.append(norm)

		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		print("\n### Learning: Representing Unitig-level Assembly Graph with Constraints.")
		cnt_wait,best = 0,1e9
		for epoch in range(self.args.epochs):
			# Training loop
			self.model.train()
			optimizer.zero_grad()

			adj_recons,means,logstds,meansAll,logstdsAll,meansRead,logstdsRead = self.model.forward(adj_norm, features, (R,I))

			rec_loss,kl_div = 0,0
			for i in range(self.args.nbatchGraph):
				rec_loss += norm * F.binary_cross_entropy(adj_recons[i].view(-1), adj_labels[i].to_dense().view(-1), weight=weight_tensors[i])
				kl_div += -0.5/adj_recons[i].size(0)*(1+2*logstds[i]-means[i]**2-torch.exp(logstds[i])**2).sum(1).mean()
			rec_loss = rec_loss/self.args.nbatchGraph
			kl_div = kl_div/self.args.nbatchGraph

			rnd = np.unique(np.random.randint(0, triplets.shape[0], self.args.nbatchConst))
			Tensor_triplets = torch.tensor(triplets[rnd]).to(self.device)
			con_loss = 10*self.model.triplet_loss(meansRead,F.elu(logstdsRead)+1+1e-14,Tensor_triplets)

			loss = rec_loss + kl_div
			loss += self.args.lambda1*con_loss

			if self.args.nbatchGraph != 1:
				bat_loss = 10*self.model.batch_loss(meansAll,F.elu(logstdsAll)+1+1e-14,pairs)
				loss += bat_loss
			else:
				bat_loss = torch.tensor(0.0)

			if (epoch+1)%20==0 or epoch == 0:
				print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.5f}".format(loss.item()), "Rec_loss=", "{:.5f}".format(rec_loss.item()), 
					"KL_Div=", "{:.5f}".format(kl_div.item()), "Batch_loss=", "{:.5f}".format(bat_loss.item()), "Con_loss=", "{:.5f}".format(con_loss.item()))

			if loss < best:
				cnt_wait,best = 0,loss
				torch.save(self.model.state_dict(), 'best_learning_'+str(self.args.runs)+'.pkl')
			else:
				cnt_wait += 1
			if cnt_wait == self.args.patience:
				print('Early stopping!')
				break

			loss.backward()
			optimizer.step()
	
		self.model.load_state_dict(torch.load('best_learning_'+str(self.args.runs)+'.pkl'))
		self.model.eval()
		_,_,_,_,_,meansRead,logstdsRead = self.model.forward(adj_norm, features, (R,I))
		_embed = torch_scatter.scatter(meansRead[..., V, :], E, dim=-2, reduce='mean')
		embed = _embed.cpu().detach().numpy()
		composition_contig = data.featContig

		print("\n### Matching: an adapted matching algorithm to initialize markered contigs.")
		constraints = data.constraints
		self.Matching = ConstraintMatching(self.args)
		_embed = {i:emb for i,emb in enumerate(composition_contig)}
		initBins = self.Matching.Estimation(constraints, _embed, THRESHOLD=self.args.threshold)
		pred_bins_matching = dict()
		for binid,contigs in initBins.items():
			for contig in contigs:
				pred_bins_matching[contig] = binid

		self.Processing = ConstraintProcessing(data.constraints,data.neg_contigs,data.contigMap,self.args)
		contigsBins_spliting,binsContigs_spliting = self.Processing.Spliting(pred_bins_matching)
		contigsBins_merging,binsContigs_merging = self.Processing.Merging(binsContigs_spliting)
		pred_bins_matching = contigsBins_merging #contigsBins_merging

		print("\n### Propagating: annotate unmarked contigs while satisfying constraints.")
		init_labels_contigs = pred_bins_matching
		n_initBins = len(init_labels_contigs)
		initalBins = {key:val for key,val in init_labels_contigs.items()} #contig-level
		inits = [val for key,val in init_labels_contigs.items()]
		# mask (contig-level)
		idxs = [idx for idx,val in initalBins.items()]
		mask = np.array([True if idx in idxs else False for idx in range(data.n_contig)])
		init_labels = [initalBins[idx] if idx in idxs else 0 for idx in range(data.n_contig)]
		init_labels = torch.LongTensor(init_labels).to(self.device)

		mask = torch.LongTensor(mask).to(self.device)
		mask = mask.float()
		mask = mask / mask.mean()

		adj = sparse_to_tuple(normalize_adj(data.readADJ))
		adj_conv = torch.sparse.FloatTensor(torch.LongTensor(adj[0].T),torch.FloatTensor(adj[1]),torch.Size(adj[2])).to(self.device)

		self.binner = ConstraintBinning(self.args.hid_dim,self.args.hid_dim,n_initBins,self.args).to(self.device)
		# print(self.binner)
		# print("# Params:", sum(p.numel() for p in self.binner.parameters() if p.requires_grad))
		conflict = torch.Tensor(np.ones(n_initBins)-np.eye(n_initBins)).to(self.device)
		neg_contigs = torch.LongTensor(data.neg_contigs).to(self.device) #negative = data.neg_contigs
		opt = torch.optim.Adam(self.binner.parameters(), lr=0.001, weight_decay=self.args.weight_decay)

		cnt_wait,best,loss_last = 0,1e9,1e9
		min_violating = 1e9
		min_preds = {}
		patience = self.args.patience
		for epoch in range(self.args.epochs):
			self.binner.train()
			opt.zero_grad()

			out,_ = self.binner(adj_conv, meansRead, (V,E))
			loss = F.cross_entropy(out, init_labels, reduction='none')
			loss *= mask
			loss = loss.mean()

			loss_n = 10*self.binner.constraintloss(out, neg_contigs, conflict)
			loss += self.args.lambda2*loss_n

			if (epoch+1)%100==0 or epoch == 0:
				print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.5f}".format(loss.item()), "loss_c=", "{:.5f}".format(loss_n.item()))

			if loss < best:
				cnt_wait,best = 0,loss
				torch.save(self.binner.state_dict(), 'best_binner_LP_'+str(self.args.runs)+'.pkl')
			else:
				cnt_wait += 1
			if cnt_wait == patience:
				print('Early stopping!')
				break

			loss.backward(retain_graph=True)
			opt.step()

		self.binner.load_state_dict(torch.load('best_binner_LP_'+str(self.args.runs)+'.pkl'))
		self.binner.eval()
		out,_emb = self.binner(adj_conv, meansRead, (V,E))
		pred = out.argmax(dim=1)
		pred_labels = {i:j.item() for i,j in enumerate(pred)} #contig

		print("\n### Refining: fine-tune contigs binning assignments.")
		contigsBins_spliting,binsContigs_spliting = self.Processing.Spliting(pred_labels)
		contigsBins_merging,binsContigs_merging = self.Processing.Merging(binsContigs_spliting)
		pred_labels = contigsBins_merging

		return pred_labels
