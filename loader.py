# !/usr/bin/env python
# -*- coding: utf8 -*-

import time,copy
import re,sys,os
import numpy as np
import networkx as nx
from Bio import SeqIO
import scipy.sparse as sp
from collections import Counter,defaultdict
from utils import *
from itertools import combinations,chain
from scipy.linalg import expm
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

### --- LOAD PART START --- ###
def load_assembly(dataset, filepath, segment2id):
	links,seqs = [],{}
	with open(filepath) as file:
		for line in file.readlines():
			if 'L' in line:
				strings = line.split("\t")
				nodeA,nodeB = int(strings[1]),int(strings[3])
				if nodeA!=nodeB: #remove self-loop segment
					links.append([nodeA,nodeB])
			if 'S' in line:
				strings = line.split("\t")
				segmentid,seq = strings[1],strings[2]
				seqs[int(segmentid)] = seq

	### RELABEL READS
	relabelLinks = []
	relabelLinks_dict = defaultdict(list)
	selected = [segment for segment,_id in segment2id.items()]
	for link in links:
		if link[0] in selected and link[1] in selected:
			relabelLinks.append([segment2id[link[0]],segment2id[link[1]]])
			relabelLinks_dict[segment2id[link[0]]].append(segment2id[link[1]])
			relabelLinks_dict[segment2id[link[1]]].append(segment2id[link[0]])
	n_segments = len(list(set([k for k,v in relabelLinks_dict.items()])))

	### COMPUTE COMPOSITION
	composition = dict()
	segmentids,sequences = [],[]
	for key,val in seqs.items():
		if key in selected:
			segmentids.append(segment2id[key])
			sequences.append(val)

	cpu_nums = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(cpu_nums)
	outputs = pool.map(compute_composition, sequences)
	pool.close()
	for idx in range(len(outputs)):
		composition[segmentids[idx]] = outputs[idx]
	print('### Loading assembly graph from: {:s} (links: {:d}, segments: {:d}).'.format(filepath,len(relabelLinks),n_segments))
	return relabelLinks,relabelLinks_dict,composition

def load_assembly_flye(dataset, filepath):
	links,seqs,paths = [],{},{}
	_contigMap,contigMap = dict(),dict()
	with open(filepath) as file:
		for line in file.readlines():
			if 'P' in line:
				strings = line.split("\t")
				contig = int(strings[1].split("_")[1])
				segments = [int(val.strip('+').split("_")[1]) if '+' in val else int(val.strip('-').split("_")[1]) for val in strings[2].split(",")]
				if contig not in paths:
					paths[contig] = segments
				_contigMap[contig] = strings[1]
			if 'L' in line:
				strings = line.split("\t")
				nodeA,nodeB = int(strings[1].split("_")[1]),int(strings[3].split("_")[1])
				if nodeA != nodeB: #remove self-loop segment
					links.append([nodeA,nodeB])
			if 'S' in line:
				strings = line.split("\t")
				segmentid,seq = int(strings[1].split("_")[1]),strings[2]
				seqs[segmentid] = seq

	### RELABEL READS
	contigs = list(set([contig for contig,segs in paths.items()]))
	contig_ids = range(0, len(contigs))
	contig2id = dict(zip(contigs, contig_ids))
	contigMap = {contig2id[k]:v for k,v in _contigMap.items()}

	segments = list(set([seg for contig,segs in paths.items() for seg in segs])) #934
	segment_ids = range(0, len(segments))
	segment2id = dict(zip(segments, segment_ids))
	print('### Loading contigs-paths from: {:s} (contigs: {:d}, segments: {:d}).'.format(filepath,len(contigs),len(segments)))

	rePaths = {}
	for contig,segs in paths.items():
		if contig in contigs:
			rePaths[contig2id[contig]] = [segment2id[seg] for seg in segs if seg in segments]

	relabelLinks = []
	relabelLinks_dict = defaultdict(list)
	for link in links:
		if link[0] in segments and link[1] in segments:
			relabelLinks.append([segment2id[link[0]],segment2id[link[1]]])
			relabelLinks_dict[segment2id[link[0]]].append(segment2id[link[1]])
			relabelLinks_dict[segment2id[link[1]]].append(segment2id[link[0]])
	n_segments = len(list(set([k for k,v in relabelLinks_dict.items()])))

	### COMPUTE COMPOSITION
	composition = dict()
	segmentids,sequences = [],[]
	for key,val in seqs.items():
		if key in segments:
			segmentids.append(segment2id[key])
			sequences.append(val)

	cpu_nums = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(cpu_nums)
	outputs = pool.map(compute_composition, sequences)
	pool.close()
	for idx in range(len(outputs)):
		composition[segmentids[idx]] = outputs[idx]

	print('### Loading assembly graph from: {:s} (links: {:d}, segments: {:d}) [{:.2f} seconds].'.format(filepath,len(relabelLinks),n_segments,	endTime-startTime))
	return relabelLinks,relabelLinks_dict,composition,rePaths,segment2id,contig2id,contigMap

def load_paths(dataset,filepath,contig2id):
	paths = dict()
	n_contigs,n_segments = 0,0
	selected = [contig for contig,_id in contig2id.items()]

	with open(filepath) as file:
		name = file.readline()
		path = file.readline()
		while name != "" and path != "":
			while ";" in path:
				path = path[:-2]+","+file.readline()
			contig = int(name.rstrip().split("_")[1])
			segments = [int(val.strip('+')) if '+' in val else int(val.strip('-')) for val in path.rstrip().split(",")]
			if contig not in paths and contig in selected:
				paths[contig2id[contig]] = segments #paths[contig] = segments
			name = file.readline()
			path = file.readline()

	### Relabel segment id
	segments = list(set([segment for contig,segments in paths.items() for segment in segments]))
	segmentids = range(0, len(segments))
	segment2id = dict(zip(segments, segmentids))

	relabelPaths = dict()
	for contig,segments in paths.items():
		relabels = [segment2id[segment] for segment in segments]
		relabelPaths[contig] = relabels

	print('### Loading contigs-paths from: {:s} (contigs: {:d}, segments: {:d}).'.format(filepath,len(relabelPaths),len(segment2id)))
	return relabelPaths,segment2id

def load_contigs(dataset, ffasta):
	composition = dict()
	contig2id,contigMap = dict(),dict()
	INT = 0
	for seq in SeqIO.parse(ffasta, "fasta"):
		if len(seq.seq) >= 0:
			contigid = int(seq.id.split('_')[1])
			contig2id[contigid] = INT
			contigMap[INT] = seq.id
			composition[INT] = compute_composition(seq.seq)
			INT+=1
	print('### Loading contigs-fasta from: {:s} (contigs: {:d}).'.format(ffasta,len(contig2id)))
	return composition,contig2id,contigMap

def load_contigs_flye(dataset, ffasta, segment2id, paths):
	composition,contigSeq,segmentSeq = dict(),dict(),dict()
	for seq in SeqIO.parse(ffasta, "fasta"):
		segid = segment2id[int(seq.id.split('_')[1])]
		segmentSeq[segid] = seq.seq
	selected = [k for k,v in segmentSeq.items()]
	for contig,segments in paths.items():
		seqs = ''
		for segment in segments:
			if segment in selected:
				seqs += segmentSeq[segment]
		contigSeq[contig] = seqs
	for contig,sequence in contigSeq.items():
		composition[contig] = compute_composition(sequence)
	print('### Loading contigs-fasta from: {:s} (contigs: {:d}).'.format(ffasta,len(composition)))
	return composition

def load_contigs_fasta_markers(filepath, contig2id):
	raw_data = np.loadtxt(filepath, delimiter=':', dtype=str)[:,1]
	raw_data = np.array([line.split(',') for line in raw_data])
	data = []
	FILTER_CONTIGS = [k for k,v in contig2id.items()]
	for line in raw_data:
		data.append([contig2id[int(val.split('_')[1])] for val in line if int(val.split('_')[1]) in FILTER_CONTIGS])
	nodes = list(set([v for line in data for v in line]))
	orderID = [pair[0] for pair in sorted({i:len(line) for i,line in enumerate(data)}.items(),key=lambda item:item[1],reverse=True)]
	data = [data[idx] for idx in orderID]
	print('### Loading single-copy marker genes from: {:s} (constraints: {:d}, constrainted contigs: {:d}).'.format(filepath,len(data),len(nodes)))
	return data

def load_contigs_fasta_markers_flye(dataset, contig2id, segment2id, paths):
	BACTERIA_MARKERS = 'data/'+dataset+'/Bacteria.ms'
	marker_path = 'data/'+dataset+'/marker_gene_stats.tsv'

	ref_sets = read_marker_gene_sets(BACTERIA_MARKERS)
	contig_markers = read_contig_genes(marker_path)
	marker_counts = get_markers_to_contigs(ref_sets, contig_markers) #{maker:segments}
	constraints_segment = []
	for marker,segments in marker_counts.items():
		constraints_segment.append([segment2id[int(segment.split("_")[1])] for segment in segments])

	segment2contig = defaultdict(list)
	for contig,segments in paths.items():
		for segment in segments:
			segment2contig[segment].append(contig)

	constraints_contig = []
	for line in constraints_segment:
		tmp = [v for val in line for v in segment2contig[val]]
		constraints_contig.append(list(set(tmp)))
	constrained_contigs = list(set([val for line in constraints_contig for val in line]))

	orderID = [pair[0] for pair in sorted({i:len(line) for i,line in enumerate(constraints_contig)}.items(),key=lambda item:item[1],reverse=True)]
	constraints_contig = [constraints_contig[idx] for idx in orderID]
	print('### Loading single-copy marker genes from: {:s} (constraints: {:d}, constrainted contigs: {:d}).'.format(marker_path,len(constraints_contig),len(constrained_contigs)))
	return constraints_contig

def load_groundTruth(filepath,contig2id):
	raw_data = np.loadtxt(filepath, delimiter=',', dtype=str)
	raw_labels = np.unique(raw_data[:,1])
	idx_map_labels = {j:i for i,j in enumerate(raw_labels)}
	FILTER_CONTIGS = [k for k,v in contig2id.items()]
	labels = dict()
	for line in raw_data:
		idx = int(line[0].split('_')[1])
		if idx in FILTER_CONTIGS:
			labels[contig2id[idx]] = int(idx_map_labels[line[1]])
	print('### Loading ground truth from: {:s} (labeled contigs: {:d}).'.format(filepath,len(labels)))
	return labels

def negative_samples_from_markers(links, links_dict, paths, markers, groundtruth):
	neg_contigs,neg_reads = [],[]
	triplets = [] #[read, pos, neg]
	for line in markers:
		combs = list(combinations(line, 2))
		for pair in combs:
			neg_contigs.append([pair[0], pair[1]])
			readsA,readsB = paths[pair[0]],paths[pair[1]]
			for readA in readsA:
				for readB in readsB:
					neg_reads.append([readA,readB])
	neg_reads = list(set([tuple(row) for row in neg_reads]))

	for line in neg_reads:
		readA,readB=line[0],line[1]
		neigA = links_dict[readA]
		for val in neigA:
			triplets.append([readA, val, readB])
		neigB = links_dict[readB]
		for val in neigB:
			triplets.append([readB, val, readA])
	triplets = list(set([tuple(row) for row in triplets]))
	print('### Sampling triplets from graph and markers (contigs: {:d}, segments: {:d}, triplets: {:d}).'.format(len(neg_contigs),len(neg_reads),len(triplets)))
	return triplets, neg_contigs

def sampling(args):
	links_dict, paths, marker = args
	neg_contigs,neg_reads,triplets = [],[],[] #[read, pos, neg]
	combs = list(combinations(marker, 2))
	for pair in combs:
		neg_contigs.append((pair[0],pair[1]))
		_readsA,_readsB = paths[pair[0]],paths[pair[1]]
		readsA,readsB = [_readsA[n] for n in (0,-1)],[_readsB[n] for n in (0,-1)]
		for readA in readsA:
			for readB in readsB:
				neg_reads.append((readA,readB))
	neg_reads = list(set(neg_reads))

	for line in neg_reads:
		readA,readB=line[0],line[1]
		neigA = links_dict[readA]
		for val in neigA:
			triplets.append((readA, val, readB))
		neigB = links_dict[readB]
		for val in neigB:
			triplets.append((readB, val, readA))
	return triplets,neg_contigs

def negative_samples_from_markers_parallel(links, links_dict, paths, markers):
	cpu_nums = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(cpu_nums)
	outputs = pool.map(sampling, [(links_dict, paths, marker) for marker in markers])
	pool.close()
	triplets = np.array(list(set(chain.from_iterable([opt[0] for opt in outputs]))))
	neg_contigs = np.array(list(set(chain.from_iterable([opt[1] for opt in outputs]))))
	print('### Sampling triplets from graph and markers (contigs: {:d}, triplets: {:d}).'.format(len(neg_contigs),len(triplets)))
	return triplets, neg_contigs

def graph_disentanglement(links, links_dict, paths, markers, compositions):
	n_segments = len(compositions)
	neg_contigs = []
	for line in markers:
		combs = list(combinations(line, 2))
		for pair in combs:
			neg_contigs.append([pair[0],pair[1]])

	_candidates_ = dict()
	for line in neg_contigs:
		contigA,contigB = line[0],line[1]
		pathA,pathB = paths[contigA],paths[contigB]
		intersection = list(set(pathA)&set(pathB))
		if len(intersection)!=0:
			A,B = str(contigA)+'_'+str(contigB),str(contigB)+'_'+str(contigA)
			if A not in _candidates_ and B not in _candidates_:
				_candidates_[str(contigA)+'_'+str(contigB)] = sum([len(links_dict[val]) for val in intersection])
	candidates = [pair[0] for pair in sorted(_candidates_.items(),key=lambda kv:(kv[1],kv[0]),reverse=False)]

	for pair in candidates:
		contigA,contigB = int(pair.split('_')[0]),int(pair.split('_')[1])
		pathA,pathB = paths[contigA],paths[contigB]
		intersection = list(set(pathA)&set(pathB))
		for val in intersection:
			old_fragment = val
			new_fragment = n_segments
			compositions[new_fragment] = compositions[old_fragment]
			tmp = [new_fragment if val==old_fragment else val for val in paths[contigB]] #replace new fragment
			paths[contigB] = tmp #update paths
			n_segments+=1 # num of fragment + 1

			relations = links_dict[old_fragment] # links in fragment graph of old node
			for val in relations:
				links.append([new_fragment,val])
	print('### Graph disentanglement (links: {:d}, segments: {:d}, contigs: {:d}).'.format(len(links),n_segments,len(paths)))
	return links, paths, compositions

def split_batches(paths, n_nodes, n_batches, postive, negative):
	def todict(inputs):
		outputs = defaultdict(set)
		for pair in inputs:
			outputs[pair[0]].add(pair[1])
			outputs[pair[1]].add(pair[0])
		return outputs

	posdict = todict(postive)  #1241
	negdict = todict(negative) #239

	n_paths = len(paths)
	batches = defaultdict(list)
	G = nx.Graph()
	G.add_edges_from(postive)
	isoComps = [list(val) for val in nx.connected_components(G)]
	connecteds = sorted(isoComps, key=lambda i:len(i), reverse=True)#[:-1]

	avgNODE = int(n_nodes/n_batches)
	CNTs = [0 for i in range(n_batches)]
	for sets in connecteds:
		idx = CNTs.index(min(CNTs))
		if len(sets)<avgNODE:
			for val in sets:
				batches[idx].append(val)
				CNTs[idx] += len(paths[val])
		else:
			for i,val in enumerate(sets):
				if i < int(len(sets)/2):
					batches[idx].append(val)
					CNTs[idx] += len(paths[val])
				else:
					batches[idx+1].append(val)
					CNTs[idx+1] += len(paths[val])

	assigned = [val for line in connecteds for val in line]
	unassigned = list(set([val for val in range(len(paths))]) - set(assigned))

	for val in unassigned:
		idx = CNTs.index(min(CNTs))
		batches[idx].append(val)
		CNTs[idx] += len(paths[val])

	readSize = []
	for i in range(len(batches)):
		readSize.append(len(list(set([val for ctg in batches[i] for val in paths[ctg]]))))
	print('### Spliting {:d} batches (size: {}).'.format(len(batches),readSize))
	return batches,readSize

def samplingContigs(args):
	contigA,pathA,paths,links_dict = args
	pos = []
	for contigB,pathB in paths.items():
		startendA = [pathA[0],pathA[-1]] #[start,end]
		startendB = [pathB[0],pathB[-1]] #[start,end]
		## TYPE-1
		if len(list(set(startendA).intersection(set(startendB))))>0 and contigA!=contigB:
			pos.append((contigA,contigB))
		## TYPE-2
		neighborA = links_dict[startendA[0]]+links_dict[startendA[1]]
		if len(list(set(neighborA).intersection(set(startendB))))>0 and contigA!=contigB: 
			pos.append((contigA,contigB))
		neighborB = links_dict[startendB[0]]+links_dict[startendB[1]]
		if len(list(set(neighborB).intersection(set(startendA))))>0 and contigA!=contigB:
			pos.append((contigA,contigB))
	pos = list(set(pos))
	return pos

def samplePostives_parallel(dataset, paths, links_dict):
	pos = []
	cpu_nums = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(cpu_nums)
	outputs = pool.map(samplingContigs, [(contigA,pathA,paths,links_dict) for contigA,pathA in paths.items()])
	pool.close()
	pos = np.array(list(set(chain.from_iterable([opt for opt in outputs]))))
	print('### Sampling positive contigs (contigs: {:d}).'.format(len(pos)))
	return pos

def filter_margers(markers, groundtruth):
	fmarkers = []
	labeled = [key for key,val in groundtruth.items()]
	for line in markers:
		val = [groundtruth[val] for val in line if val in labeled]
		if len(val)!=len(list(set(val))):
			errorBins= [key for key,val in Counter(val).items() if val!=1]
			# _line = [v for v in line if v in labeled and groundtruth[v] not in errorBins]
			_line = []
			for v in line:
				if v in labeled:
					if groundtruth[v] not in errorBins:
						_line.append(v)
				else:
					_line.append(v)
			fmarkers.append(_line)
		else:
			fmarkers.append(line)
	return fmarkers

def graph_augmentation(postive,paths,segment2id,links,links_dict):
	selected = [segment for segment,_ in links_dict.items()]
	for pair in postive:
		contigA,contigB = pair[0],pair[1]
		pathA,pathB = paths[contigA],paths[contigB]
		for segmentA in pathA:
			for segmentB in pathB:
				# if segmentA in selected and segmentB in selected and segmentA!=segmentB:
				if segmentA!=segmentB:
					links.append([segmentA,segmentB])
					if segmentA in selected:
						links_dict[segmentA].append(segmentB)
					else:
						links_dict[segmentA] = [segmentB]
					if segmentB in selected:
						links_dict[segmentB].append(segmentA)
					else:
						links_dict[segmentB] = [segmentA]
	print('### Graph augmentation (links: {:d}, segments: {:d}).'.format(len(links),len(links_dict)))
	return links,links_dict

def batchGraph(batches,paths,links,featV,readADJ):
	batchBlock = dict()
	readsAll,readsPair = [],[]
	for id,batch in batches.items():
		readList = list(set([val for ctg in batch for val in paths[ctg]]))
		ids = range(0, len(readList))
		read2id = dict(zip(readList, ids))
		readAdj = np.eye(len(readList))
		for link in links:
			if link[0] in readList and link[1] in readList:
				readAdj[read2id[link[0]],read2id[link[1]]]=1.0
				readAdj[read2id[link[1]],read2id[link[0]]]=1.0
		readFeat = featV[readList]
		batchBlock[id] = [readList,sp.csr_matrix(readAdj),sp.csr_matrix(readFeat)]
		readsAll+=readList
	readsIdx = [i for i in range(len(readsAll))]
	batchRead = [readsAll,readsIdx]
	for i in range(len(readsAll)-1):
		for j in range(i+1, len(readsAll)):
			if readsAll[i] == readsAll[j]:
				readsPair.append([i,j])
	return batchBlock,batchRead,readsPair

def adjMatrix(links, n_nodes):
	readADJ = np.eye(n_nodes)
	for link in links:
		readADJ[link[0],link[1]]=1.0
		readADJ[link[1],link[0]]=1.0
	return sp.csr_matrix(readADJ)

def load_data_flye(dataset):
	fassemblygraph = 'data/'+dataset+'/assembly_graph.gfa'
	ffasta = 'data/'+dataset+'/assembly.fasta'
	fmarkers = 'data/'+dataset+'/contigs.fasta.markers'

	links,links_dict,composition_segments,paths,segment2id,contig2id,contigMap  = load_assembly_flye(dataset,fassemblygraph)
	compositions_contigs = load_contigs_flye(dataset,ffasta,segment2id,paths)
	markers = load_contigs_fasta_markers_flye(dataset,contig2id,segment2id,paths)
	postive = samplePostives_parallel(dataset,paths,links_dict)
	links,links_dict = graph_augmentation(postive,paths,segment2id,links,links_dict)
	links,paths,composition_segments = graph_disentanglement(links,links_dict,paths,markers,composition_segments)
	triplets,neg_contigs = negative_samples_from_markers_parallel(links,links_dict,paths,markers)

	return paths,links,markers,triplets,neg_contigs,composition_segments,contigMap,compositions_contigs,postive

def load_data(dataset):
	fpaths = 'data/'+dataset+'/contigs.paths'
	fassemblygraph = 'data/'+dataset+'/assembly_graph_with_scaffolds.gfa'
	ffasta = 'data/'+dataset+'/contigs.fasta'
	fmarkers = 'data/'+dataset+'/contigs.fasta.markers'

	compositions_contigs,contig2id,contigMap = load_contigs(dataset,ffasta)
	paths,segment2id = load_paths(dataset,fpaths,contig2id)
	links,links_dict,composition_segments = load_assembly(dataset,fassemblygraph,segment2id)
	markers = load_contigs_fasta_markers(fmarkers,contig2id)
	postive = samplePostives_parallel(dataset,paths,links_dict)
	links,links_dict = graph_augmentation(postive,paths,segment2id,links,links_dict)
	links,paths,composition_segments = graph_disentanglement(links,links_dict,paths,markers,composition_segments)
	triplets,neg_contigs = negative_samples_from_markers_parallel(links,links_dict,paths,markers)

	return paths,links,markers,triplets,neg_contigs,composition_segments,contigMap,compositions_contigs,postive


class MetaData():
	def __init__(self, dataset, args):
		super(MetaData, self).__init__()
		if dataset in ['Sim5G','Sim10G','Sim20G','Sim50G','Sim100G','Sharon','DeepHPM','COPD']:
			paths,links,markers,triplets,neg_contigs,composition_segments,contigMap,compositions_contigs,postive = load_data(dataset)
		elif dataset in ['Strong100','Hjor','Viby','Damh','Mari','AalE','Hade']:
			paths,links,markers,triplets,neg_contigs,composition_segments,contigMap,compositions_contigs,postive = load_data_flye(dataset)
		n_nodes = len(composition_segments)
		self.num_nodes = n_nodes
		batchIDs,batchSize = split_batches(paths,n_nodes,args.nbatchGraph,postive,neg_contigs)
		self.paths = paths
		self.edge_index = np.array([[segment,contig] for contig,segments in paths.items() for segment in segments]).T
		featV = np.array([composition_segments[i] for i in range(len(composition_segments))])
		featContig = np.array([compositions_contigs[i] for i in range(len(compositions_contigs))])
		self.featContig = featContig
		self.triplets = triplets
		self.constraints = markers
		readADJ = adjMatrix(links, n_nodes)
		batches,batchReads,readsPair = batchGraph(batchIDs,paths,links,featV,readADJ)
		self.readADJ = readADJ
		self.batches = batches
		self.batchReads = batchReads
		self.readsPair = readsPair
		self.inputs_dim = [val+136 for val in batchSize]
		self.neg_contigs = neg_contigs
		self.pos_contigs = postive
		self.contigMap = contigMap
		self.n_fragment = n_nodes
		self.n_contig = len(contigMap)
