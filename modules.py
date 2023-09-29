# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch,torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from utils import glorot_init,set_device
import numpy as np
from collections import defaultdict,Counter


class GraphConv(nn.Module):
	def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
		super(GraphConv, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.activation = activation

	def forward(self, inputs):
		adj, x = inputs
		x = torch.matmul(x, self.weight)
		x = torch.matmul(adj, x)
		outputs = self.activation(x)
		return outputs

class VGAE(nn.Module):
	def __init__(self, input_dim, hidden1_dim, hidden2_dim, args, device):
		super(VGAE, self).__init__()
		self.input_dim = input_dim
		self.hidden1_dim = hidden1_dim
		self.hidden2_dim = hidden2_dim
		self.dropout = args.dropout
		self.Base_GCN1 = GraphConv(input_dim, hidden1_dim)
		self.GCN_mean = GraphConv(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.GCN_logstddev = GraphConv(hidden1_dim, hidden2_dim, activation=lambda x:x)
		self.device = device

	def embedd(self, adj, feats):
		hidden = self.Base_GCN1((adj, feats))
		mean = self.GCN_mean((adj, hidden))
		return hidden, mean

	def encoder(self, adj, feats):
		hidden, mean = self.embedd(adj, feats)
		logstd = self.GCN_logstddev((adj, hidden)) #+1#+1e-14
		gaussian_noise = torch.randn(adj.size(0), self.hidden2_dim).to(self.device)
		sampled_z = gaussian_noise*torch.exp(logstd) + mean
		return sampled_z, mean, logstd

	def decoder(self, z):
		z = F.dropout(z,self.dropout,training=self.training)
		x_hat = torch.sigmoid(torch.matmul(z, z.t()))
		return x_hat

	def forward(self, adj, feats):
		z, mean, logstd = self.encoder(adj,feats)
		x_reconst = self.decoder(z)
		return x_reconst, mean, logstd

# MultiVGAE model
class MultiVGAE(nn.Module):
	def __init__(self, input_dim, hidden1_dim, hidden2_dim, args, device):
		super(MultiVGAE, self).__init__()
		self.args = args
		self.n_model = args.nbatchGraph
		self.models = nn.ModuleList()
		for i in range(self.n_model):
			self.models.append(VGAE(input_dim[i], hidden1_dim, hidden2_dim, args, device))

	def forward(self, adj, feats, batchRI):
		recs,means,logstds = [],[],[]
		R,I = batchRI
		for i in range(self.n_model):
			rec,mean,logstd = self.models[i](adj[i],feats[i])
			recs.append(rec)
			means.append(mean)
			logstds.append(logstd)
		if self.n_model == 5:
			meansAll = torch.cat((means[0],means[1],means[2],means[3],means[4]), 0)
			logstdsAll = torch.cat((logstds[0],logstds[1],logstds[2],logstds[3],logstds[4]), 0)
		elif self.n_model == 4:
			meansAll = torch.cat((means[0],means[1],means[2],means[3]), 0)
			logstdsAll = torch.cat((logstds[0],logstds[1],logstds[2],logstds[3]), 0)
		elif self.n_model == 3:
			meansAll = torch.cat((means[0], means[1], means[2]), 0)
			logstdsAll = torch.cat((logstds[0], logstds[1], logstds[2]), 0)
		elif self.n_model == 2:
			meansAll = torch.cat((means[0], means[1]), 0)
			logstdsAll = torch.cat((logstds[0], logstds[1]), 0)
		elif self.n_model == 1:
			meansAll = means[0]
			logstdsAll = logstds[0]
		meansRead = torch_scatter.scatter(meansAll[..., I, :], R, dim=-2, reduce='mean')
		logstdsRead = torch_scatter.scatter(logstdsAll[..., I, :], R, dim=-2, reduce='mean')
		return recs,means,logstds,meansAll,logstdsAll,meansRead,logstdsRead


	def batch_loss(self, mean, variance, pairs):
		energy = self.energy_kl(mean, variance, pairs)
		loss = energy**2
		return loss.mean()

	def triplet_loss(self, mean, variance, triplets):
		pos_pairs = torch.stack([triplets[:, 0], triplets[:, 1]], 1)
		neg_pairs = torch.stack([triplets[:, 0], triplets[:, 2]], 1)
		energy_pos = self.energy_kl(mean, variance, pos_pairs)
		energy_neg = self.energy_kl(mean, variance, neg_pairs)
		energy = energy_pos**2 + torch.exp(-energy_neg)
		loss = energy.mean()
		return loss

	def energy_kl(self, mean, variance, pairs):
		'''Computes the energy of a set of node pairs as the KL divergence between their respective Gaussian embeddings.'''
		i_mu = torch.index_select(mean, 0, pairs[:,0])
		j_mu = torch.index_select(mean, 0, pairs[:,1])
		i_sigma = torch.index_select(variance, 0, pairs[:,0])
		j_sigma = torch.index_select(variance, 0, pairs[:,1])
		sigma_ratio = j_sigma / i_sigma
		trace_fac = sigma_ratio.sum(1)
		log_det = (torch.log(sigma_ratio+1e-14)).sum(1)
		mu_diff_sq = ((i_mu-j_mu)**2 / i_sigma).sum(1)
		return 0.5 * (trace_fac + mu_diff_sq - self.args.hid_dim - log_det)


### Constraint-based Binning model
class ConstraintBinning(nn.Module):
	def __init__(self, ipt_dim, hid_dim, nbin, args):
		super(ConstraintBinning, self).__init__()
		self.gcn = GraphConv(ipt_dim, hid_dim)
		self.clf = nn.Sequential(
			nn.Linear(in_features=hid_dim, out_features=hid_dim),nn.ReLU(inplace=True),
			nn.Linear(in_features=hid_dim, out_features=nbin))
		self.softmax = nn.Softmax()

	def forward(self, adj, feats, VE):
		V,E = VE
		hid = self.gcn((adj, feats))
		hid_c = torch_scatter.scatter(hid[...,V,:],E,dim=-2,reduce='mean')
		lbl_f = self.clf(hid_c)
		return lbl_f,hid_c

	def constraintloss(self, preds, negative, conflict):
		preds = self.softmax(preds)
		left,right = negative[:,0],negative[:,1]
		phi_left,phi_right = preds[left,:],preds[right,:]

		conflicts_relation_loss = torch.sum(torch.matmul(phi_left, conflict)*phi_right, axis=1)
		relation_loss = -torch.log(conflicts_relation_loss)
		relation_loss = torch.sum(relation_loss, axis=0)	
		loss_conflict = relation_loss/len(negative)
		return loss_conflict

class ConstraintOptimizing(nn.Module):
	def __init__(self, ipt_dim, hid_dim, nbin, args):
		super(ConstraintOptimizing, self).__init__()
		self.gcn = GraphConv(ipt_dim, hid_dim)
		self.clf = nn.Sequential(
			nn.Linear(in_features=hid_dim, out_features=hid_dim),nn.ReLU(inplace=True),
			nn.Linear(in_features=hid_dim, out_features=nbin))
		self.softmax = nn.Softmax()

	def forward(self, adj, feats, VE):
		V,E = VE
		hid = self.gcn((adj, feats))
		hid_c = torch_scatter.scatter(hid[...,V,:],E,dim=-2,reduce='mean')
		lbl_f = self.clf(hid_c)
		return lbl_f

	def constraintloss(self, preds, negative, conflict):
		preds = self.softmax(preds)
		left,right = negative[:,0],negative[:,1]
		phi_left,phi_right = preds[left,:],preds[right,:] #torch.Size([6469,21]),torch.Size([6469,21])

		conflicts_relation_loss = torch.sum(torch.matmul(phi_left, conflict)*phi_right, axis=1) #torch.Size([6469])
		relation_loss = -torch.log(conflicts_relation_loss)
		relation_loss = torch.sum(relation_loss, axis=0)	
		loss_conflict = relation_loss/len(negative)
		return loss_conflict


class ConstraintMatching(nn.Module):
	def __init__(self, args):
		super(ConstraintMatching, self).__init__()
		self.args = args

	def Estimation(self, binSets, embeds, THRESHOLD):
		lenlist = [len(line) for line in binSets]
		n_Inits,firstset = max(lenlist),lenlist.index(max(lenlist))
		# print("### Number for bins (Initial):{:d}.".format(n_Inits))
		initBins = defaultdict(set)
		for i,val in enumerate(binSets[firstset]):
			initBins[i].add(val)
		candidates = [i for i in range(len(binSets)) if i!=firstset]

		for cand in candidates:
			binset = binSets[cand]
			###CALCULATE
			MAT = np.zeros((len(initBins),len(binset)))
			for i in range(len(initBins)):
				bins = initBins[i]
				for j in range(len(binset)):
					embJ = embeds[binset[j]]
					dists = []
					for _bin_ in bins:
						_emb_ = embeds[_bin_]
						dists.append(np.linalg.norm(embJ-_emb_))
					dist = sum(dists)/len(dists)
					MAT[i,j] = dist
			### ASSIGNMENT
			minVal,minIdx = [],[]
			flagBIN,flagCAND = [],[]
			CNT = 0
			while CNT < len(binset):
				idx = [list(val)[0] for val in np.where(MAT==np.max(MAT.min()))]
				if idx[0] not in flagBIN and idx[1] not in flagCAND:
					minIdx.append(idx)
					minVal.append(MAT.min())
					flagBIN.append(idx[0])
					flagCAND.append(idx[1])
					MAT[idx[0],idx[1]] = np.inf
					CNT += 1
				else:
					MAT[idx[0],idx[1]] = np.inf
			FLAG = [True]
			for i in range(1,len(minVal)):
				diff = minVal[i]-minVal[i-1]
				if diff > THRESHOLD:
					FLAG.append(False)
				else:
					FLAG.append(True)
			for i,val in enumerate(FLAG):
				binid,condid = minIdx[i]
				if val == True:
					initBins[binid].add(binset[condid])
				else:
					initBins[len(initBins)].add(binset[condid])
		print("### Number for bins (Matching:{:d}, Initial:{:d}).".format(len(initBins),n_Inits))
		return initBins

class ConstraintProcessing(nn.Module):
	def __init__(self, markers,negative,contigMap,args):
		super(ConstraintProcessing, self).__init__()
		self.args = args
		self.markers = markers
		self.negative = negative
		self.contigMap = contigMap
		self.N_MARKERS = len(markers)

	def Spliting(self, preds):
		### POST-PROCESSING
		binsContigs = defaultdict(list)
		for contig,label in preds.items():
			binsContigs[label].append(contig)

		constrained = list(set([v for line in self.markers for v in line]))
		contigs2marker = defaultdict(list)
		for idx,contigs in enumerate(self.markers):
			for contig in contigs:
				contigs2marker[contig].append(idx)

		bins2marker = defaultdict(list)
		for _bin,contigs in binsContigs.items():
			markers = []
			for contig in list(set(contigs).intersection(set(constrained))):
				for val in contigs2marker[contig]:
					markers.append(val)
			bins2marker[_bin] = markers #list(set(markers))

		_candidates2spliting = dict()
		for _bin,markers in bins2marker.items():
			selected = {marker:cnt for marker,cnt in Counter(markers).items() if cnt!=1}
			if selected != {}:
				_candidates2spliting[_bin] = len(selected)

		### SPLITING
		candidates2spliting = [pair[0] for pair in sorted(_candidates2spliting.items(),key=lambda item:item[1],reverse=True)]
		subBinsAll = []
		for _bin in candidates2spliting:
			# print('--- Bin Spliting: ',_bin)
			contigs = binsContigs[_bin]
			sortedContigs = [pair[0] for pair in sorted({c:len(contigs2marker[c]) for c in contigs}.items(),key=lambda item:item[1],reverse=False)]
			markersBins = defaultdict(list)
			subBins = defaultdict(list)
			INT = 0
			for contig in sortedContigs:
				contigmarkers = contigs2marker[contig]
				if len(markersBins)==0:
					markersBins[INT] = contigmarkers
					subBins[INT] = [contig]
				else:
					for i in range(INT, INT+1):
						intersection = list(set(markersBins[i]).intersection(set(contigmarkers)))
						if len(intersection)==0:
							for cm in contigmarkers:
								markersBins[INT].append(cm)
							subBins[INT].append(contig)
						else:
							INT += 1
							markersBins[INT] = contigmarkers
							subBins[INT] = [contig]
			subBinsAll.append(subBins)
		CNTBINS = max([_bin for _bin,_ in binsContigs.items()])+1
		binsContigs_spliting = defaultdict(list)
		for _bin,contigs in binsContigs.items():
			if _bin not in candidates2spliting:
				binsContigs_spliting[_bin] = contigs
			else:
				idx = candidates2spliting.index(_bin)
				selectedsubBins = subBinsAll[idx]
				for subID,contigs in selectedsubBins.items():
					if subID == 0:
						binsContigs_spliting[_bin] = contigs
					else:
						binsContigs_spliting[CNTBINS] = contigs
						CNTBINS += 1
		###RE-LABEL BINS
		binids = range(0, len(binsContigs_spliting))
		bin2id = dict(zip(binsContigs_spliting,binids))
		binsContigs_spliting = {bin2id[_bin]:contigs for _bin,contigs in binsContigs_spliting.items()}
		contigsBins_spliting = {contig:_bin for _bin,contigs in binsContigs_spliting.items() for contig in contigs}
		return contigsBins_spliting,binsContigs_spliting

	def Merging(self, binsContigs_spliting):
		constrained = list(set([v for line in self.markers for v in line]))
		contigs2marker = defaultdict(list)
		for idx,contigs in enumerate(self.markers):
			for contig in contigs:
				contigs2marker[contig].append(idx)

		iterations = 5
		THRESHOLD = [10,5,2,1,1]
		for i in range(iterations):
			thd = THRESHOLD[i]
			bins2marker_spliting = defaultdict(list)
			for _bin,contigs in binsContigs_spliting.items():
				markers = []
				for contig in list(set(contigs).intersection(set(constrained))):
					for val in contigs2marker[contig]:
						markers.append(val)
				bins2marker_spliting[_bin] = list(set(markers))
			bins2lengthmarker = {key:len(val) for key,val in bins2marker_spliting.items()}

			candidates = dict()
			for _bin,markers in bins2marker_spliting.items():
				for _bin_,_markers_ in bins2marker_spliting.items():
					if _bin!=_bin_:
						inter = list(set(markers).intersection(set(_markers_)))
						if len(inter)==0 and len(markers)>=thd and len(_markers_)>=thd:
							if abs(len(markers)-len(_markers_)) < 5000:
								if str(_bin)+'_'+str(_bin_) not in candidates and str(_bin_)+'_'+str(_bin) not in candidates:
									candidates[str(_bin)+'_'+str(_bin_)] = len(markers)+len(_markers_)
			candidates = [pair[0] for pair in sorted(candidates.items(),key=lambda item:item[1],reverse=True)]

			_candidates4merging = [[int(pair.split('_')[0]),int(pair.split('_')[1])] for pair in candidates]
			candidateBins = list(set([val for pair in _candidates4merging for val in pair]))

			bins2merging,_candidatesFlag = [],[]
			for pair in _candidates4merging:
				if pair[0] not in _candidatesFlag and pair[1] not in _candidatesFlag:
					bins2merging.append(pair)
					_candidatesFlag.append(pair[0])
					_candidatesFlag.append(pair[1])

			binsContigs_merging = defaultdict(list)
			for _bin,contigs in binsContigs_spliting.items():
				if _bin not in _candidatesFlag:
					binsContigs_merging[_bin] = contigs
			for merging in bins2merging:
				for _bin_ in merging:
					for val in binsContigs_spliting[_bin_]:
						binsContigs_merging[merging[0]].append(val)
			binids = range(0, len(binsContigs_merging))
			bin2id = dict(zip(binsContigs_merging,binids))
			binsContigs_merging = {bin2id[_bin]:contigs for _bin,contigs in binsContigs_merging.items()}

			binsContigs_spliting = binsContigs_merging
		contigsBins_merging = {contig:_bin for _bin,contigs in binsContigs_merging.items() for contig in contigs}			
		return contigsBins_merging,binsContigs_merging

	def Discarding(self, binsContigs_merging):
		constrained = list(set([v for line in self.markers for v in line]))
		contigs2marker = defaultdict(list)
		for idx,contigs in enumerate(self.markers):
			for contig in contigs:
				contigs2marker[contig].append(idx)

		### Discarding
		bins2marker_discarding = defaultdict(list)
		for _bin,contigs in binsContigs_merging.items():
			markers = []
			for contig in list(set(contigs).intersection(set(constrained))):
				for val in contigs2marker[contig]:
					markers.append(val)
			bins2marker_discarding[_bin] = list(set(markers))
		bins2lengthmarker = {key:len(val) for key,val in bins2marker_discarding.items()}
		bins2lengthmarker = {_bin:_len for _bin,_len in sorted(bins2lengthmarker.items(),key=lambda item:item[1],reverse=True)}

		thd2filter = int(self.N_MARKERS/4)
		print('### Number of MARKERS: {:d}, THRESHOLD to DISCARD (less than are removed): {:d}.'.format(self.N_MARKERS,thd2filter))
		finalBINs = [_bin for _bin,_len in bins2lengthmarker.items() if _len>thd2filter]

		finalBinsContigs = dict()
		for _bin,contigs in binsContigs_merging.items():
			if _bin in finalBINs:
				finalBinsContigs[_bin] = contigs
		print('### Number of final bins: {:d}'.format(len(finalBinsContigs)))

		binids = range(0, len(finalBinsContigs))
		bin2id = dict(zip(finalBinsContigs,binids))
		binsContigs_discarding = {bin2id[_bin]:contigs for _bin,contigs in finalBinsContigs.items()}
		contigsBins_discarding = {contig:_bin for _bin,contigs in binsContigs_discarding.items() for contig in contigs}	
		return contigsBins_discarding,binsContigs_discarding
