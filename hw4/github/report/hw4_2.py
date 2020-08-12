# -*- coding: utf-8 -*-

# this is for filtering the warnings
import warnings
warnings.filterwarnings('ignore')

# utils.py
# 這個 block 用來先定義一些等等常用到的函式
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

def load_testing_data(path='testing_data'):
	# 把 testing 時需要的 data 讀進來
	sen1 = "today is a good day , but it is hot" 
	sen2 = "today is hot , but it is a good day"
	with open(path, 'r') as f:
		lines = f.readlines()
		X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
		X.append(sen1)
		X.append(sen2)
		X = [sen.split(' ') for sen in X]
	return X


def evaluation(outputs, labels):
	# outputs => probability (float)
	# labels => labels
	outputs[outputs>=0.5] = 1 # 大於等於 0.5 為有惡意
	outputs[outputs<0.5] = 0 # 小於 0.5 為無惡意
	correct = torch.sum(torch.eq(outputs, labels)).item()
	return correct

"""### Data Preprocess"""

# preprocess.py
# 這個 block 用來做 data 的預處理
from torch import nn
from gensim.models import Word2Vec

class Preprocess():
	def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
		self.w2v_path = w2v_path
		self.sentences = sentences
		self.sen_len = sen_len
		self.idx2word = []
		self.word2idx = {}
		self.embedding_matrix = []
	def get_w2v_model(self):
		# 把之前訓練好的 word to vec 模型讀進來
		self.embedding = Word2Vec.load(self.w2v_path)
		self.embedding_dim = self.embedding.vector_size
	def add_embedding(self, word):
		# 把 word 加進 embedding，並賦予他一個隨機生成的 representation vector
		# word 只會是 "<PAD>" 或 "<UNK>"
		vector = torch.empty(1, self.embedding_dim)
		torch.nn.init.uniform_(vector)
		self.word2idx[word] = len(self.word2idx)
		self.idx2word.append(word)
		self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
	def make_embedding(self, load=True):
		print("Get embedding ...")
		# 取得訓練好的 Word2vec word embedding
		if load:
			print("loading word to vec model ...")
			self.get_w2v_model()
		else:
			raise NotImplementedError
		# 製作一個 word2idx 的 dictionary
		# 製作一個 idx2word 的 list
		# 製作一個 word2vector 的 list
		for i, word in enumerate(self.embedding.wv.vocab):
			print('get words #{}'.format(i+1), end='\r')
			#e.g. self.word2index['he'] = 1
			#e.g. self.index2word[1] = 'he'
			#e.g. self.vectors[1] = 'he' vector
			self.word2idx[word] = len(self.word2idx)
			self.idx2word.append(word)
			self.embedding_matrix.append(self.embedding[word])
		print('')
		self.embedding_matrix = torch.tensor(self.embedding_matrix)
		# 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
		self.add_embedding("<PAD>")
		self.add_embedding("<UNK>")
		print("total words: {}".format(len(self.embedding_matrix)))
		return self.embedding_matrix
	def pad_sequence(self, sentence):
		# 將每個句子變成一樣的長度
		if len(sentence) > self.sen_len:
			sentence = sentence[:self.sen_len]
		else:
			pad_len = self.sen_len - len(sentence)
			for _ in range(pad_len):
				sentence.append(self.word2idx["<PAD>"])
		assert len(sentence) == self.sen_len
		return sentence
	def sentence_word2idx(self):
		# 把句子裡面的字轉成相對應的 index
		sentence_list = []
		for i, sen in enumerate(self.sentences):
			print('sentence count #{}'.format(i+1), end='\r')
			sentence_idx = []
			for word in sen:
				if (word in self.word2idx.keys()):
					sentence_idx.append(self.word2idx[word])
				else:
					sentence_idx.append(self.word2idx["<UNK>"])
			# 將每個句子變成一樣的長度
			sentence_idx = self.pad_sequence(sentence_idx)
			sentence_list.append(sentence_idx)
		return torch.LongTensor(sentence_list)
	def labels_to_tensor(self, y):
		# 把 labels 轉成 tensor
		y = [int(label) for label in y]
		return torch.LongTensor(y)

"""### Dataset"""

# data.py
# 實作了 dataset 所需要的 '__init__', '__getitem__', '__len__'
# 好讓 dataloader 能使用
import torch
from torch.utils import data

class TwitterDataset(data.Dataset):
	"""
		Expected data shape like:(data_num, data_len)
		Data can be a list of numpy array or a list of lists
		input data shape : (data_num, seq_len, feature_dim)
		
		__len__ will return the number of data
		"""
	def __init__(self, X, y):
		self.data = X
		self.label = y
	def __getitem__(self, idx):
		if self.label is None: return self.data[idx]
		return self.data[idx], self.label[idx]
	def __len__(self):
		return len(self.data)

"""### Model"""

# model.py
# 這個 block 是要拿來訓練的模型
import torch
from torch import nn
class LSTM_Net(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
		super(LSTM_Net, self).__init__()
		# 製作 embedding layer
		self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
		self.embedding.weight = torch.nn.Parameter(embedding)
		# 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
		self.embedding.weight.requires_grad = False if fix_embedding else True
		self.embedding_dim = embedding.size(1)
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
		self.classifier = nn.Sequential( nn.Dropout(dropout),
										nn.Linear(hidden_dim*2, 1),
										nn.Sigmoid() )
	def forward(self, inputs):
		inputs = self.embedding(inputs)
		x, _ = self.lstm(inputs, None)
		# x 的 dimension (batch, seq_len, hidden_size)
		# 取用 LSTM 最後一層的 hidden state
		x = x[:, -1, :]
		x = self.classifier(x)
		return x
"""### Test"""

# test.py
# 這個 block 用來對 testing_data.txt 做預測
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def testing(batch_size, test_loader, model, device):
	model.eval()
	ret_output = []
	with torch.no_grad():
		for i, inputs in enumerate(test_loader):
			inputs = inputs.to(device, dtype=torch.long)
			outputs = model(inputs)
			outputs = outputs.squeeze()
			outputs[outputs>=0.5] = 1 # 大於等於 0.5 為負面
			outputs[outputs<0.5] = 0 # 小於 0.5 為正面
			ret_output += outputs.int().tolist()
	
	return ret_output

def testing_raw(batch_size, test_loader, model, device):
	model.eval()
	ret_output = []
	with torch.no_grad():
		for i, inputs in enumerate(test_loader):
			inputs = inputs.to(device, dtype=torch.long)
			outputs = model(inputs)
			outputs = outputs.squeeze()
			ret_output += outputs.tolist()	
	
	return ret_output[-2:]

# main.py
import os
import sys
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
	testing_data = '../testing_data.txt'
	#output_path = sys.argv[2]
	path_prefix = '../model'
	w2v_path = os.path.join(path_prefix, 'w2v_all_gru.model')

	"""### Predict and Write to csv file"""

	batch_size = 64
	sen_len = 40
	print("loading testing data ...")
	test_x = load_testing_data(testing_data)
	preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
	embedding = preprocess.make_embedding(load=True)
	test_x = preprocess.sentence_word2idx()
	test_dataset = TwitterDataset(X=test_x, y=None)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
										  batch_size = batch_size,
										  shuffle = False,
										  num_workers = 8)
	print('\nload model ...')
	#model = torch.load(os.path.join(path_prefix, 'ckpt_gru.model'))
	model = torch.load('../ckpt_gru.model?dl=1')
	outputs = testing(batch_size, test_loader, model, device)
	report = testing_raw(batch_size, test_loader, model, device)
	print(report)
	print("\"today is a good day, but it is hot\": {}".format(report[0]))
	print("\"today is hot, but it is a good day\": {}".format(report[1]))
	# 寫到 csv 檔案供上傳 Kaggle
	tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
	#print("save csv ...")
	#tmp.to_csv(output_path, index=False)
	print("Finish Predicting")

if __name__ == "__main__":
	main()
