# !/usr/bin/python 
# coding: utf-8

import sys 
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import defaultdict

vecdict = dict()

import numpy as np

with open('./vecs') as fi:
	for line in fi:
		arr = line.decode('utf-8').strip().split(' ')
		char = arr[0]
		vec = np.array(arr[1:], np.float32)
		vecdict[char] = vec

word_dict = defaultdict(int)


sub_dict = dict()
with open('/home/zhangyuxiang/Projects/name_reg/data/rnn_input/test_splited') as fi:
	for line in fi:
		arr = line.decode('utf-8').strip().split(' ')
		for x in arr[1:]:
			if x not in vecdict:
				break
		else:
			for x in arr[1:]:
				if x not in sub_dict:
					sub_dict[x] = vecdict[x]
			print ' '.join(arr[1:])
