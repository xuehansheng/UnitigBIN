# !/usr/bin/env python
# -*- coding: utf8 -*-

import time
import sys,argparse
import loader
from model import UnitigBIN
from utils import save_dict_to_csv

def parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str, default='Sim20G', help='Dataset string.')
	parser.add_argument('-l', '--lr', type=float, default=0.01, help='Number of learning rate.')
	parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs.')
	parser.add_argument('-b', '--nbatchGraph', type=int, default=1, help='Number of batch size for assembly graph (>2).')
	parser.add_argument('--hid_dim', type=int, default=64, help='Dimension of hidden.')
	parser.add_argument('--alpha', type=float, default=0.05, help='Teleport probability in graph diffusion convolution operators.')
	parser.add_argument('--eps', type=float, default=0.001, help='Threshold epsilon to sparsify in graph diffusion convolution operators.')
	parser.add_argument('--lambda1', type=float, default=0.7, help='Lambda parameter for the constraint loss.')
	parser.add_argument('--lambda2', type=float, default=0.1, help='Lambda parameter for the propagating loss.')
	parser.add_argument('--nbatchConst', type=int, default=500, help='Number of batch size for constraints (default:10000).')
	parser.add_argument('--threshold', type=float, default=0.02, help='Threshold parameter for the matching algorithm.')
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
	parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping (# of epochs).')
	parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
	parser.add_argument('--input_dropout', default=0.0, type=float)
	parser.add_argument('--dropout', default=0.3, type=float)
	parser.add_argument('-r', '--runs', type=int, default=00, help='Number of runs.')

	args = parser.parse_args()
	print(args)
	return args

def run(args):
	data = loader.MetaData(dataset=args.dataset,args=args)
	model = UnitigBIN(data.inputs_dim, args)
	pred_labels = model.learning(data)
	preds = {data.contigMap[key]:val for key,val in pred_labels.items()}
	save_dict_to_csv(preds, 'UnitigBin_'+args.dataset+'_'+str(args.runs))

if __name__ == '__main__':
	run(parser())