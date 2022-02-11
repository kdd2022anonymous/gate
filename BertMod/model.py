import numpy as np

import torch
import torch.nn as nn

from bert_transformers.modeling_bert import (BertModel, )

from transformers.optimization import (get_linear_schedule_with_warmup,
									   AdamW, )

from .encoding_utils import TreePositionalEncodings


class MultilabelClassification(nn.Module):
	def __init__(self, config, num_of_class):
		super().__init__()
		self.num_of_class = num_of_class
		
		self.tanhL = nn.Tanh()
		self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
		self.dense2 = nn.Linear(config.hidden_size, num_of_class)
		self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
	
	def forward(self, pooling_result, label_ids=None):
		tanh_layer = self.tanhL(self.dense1(pooling_result))
		logits = self.dense2(tanh_layer)
		
		loss = self.BCEWithLogitsLoss(logits, label_ids.float())
		return loss, logits


class Model(nn.Module):
	
	def __init__(self, num_of_class, config_bert=None):
		super(Model, self).__init__()
		self.num_of_class = num_of_class
		self.config = config_bert
		
		# compute huffman embedding based on huffman id through sample bert models
		self.TreePositionalEncodings = TreePositionalEncodings(depth=16, degree=16, n_feat=2, d_model=512)

		self.bert = BertModel(self.config)
		
		self.MultilabelClassificater = MultilabelClassification(self.config, self.num_of_class)

	
	def forward(self, input_ids=None, attention_mask=None,  two_dimensional_attention_mask=None, huffman_ids=None, type_ids=None, labels=None):
		huffman_tensor_size = huffman_ids.shape 
		batch_size = huffman_tensor_size[0]
		seq_length = huffman_tensor_size[1]

		one_hot_hf = torch.nn.functional.one_hot(huffman_ids.long(),num_classes=16)
		one_hot_hf = one_hot_hf.view(batch_size, seq_length, -1)
		huffman_embedding = self.TreePositionalEncodings(one_hot_hf)
		
		output = self.bert(input_ids=input_ids, huffman_embedding=huffman_embedding, attention_mask=attention_mask, two_dimensional_attention_mask=two_dimensional_attention_mask, type_ids=type_ids, return_dict=True)
		
		loss, logits = self.MultilabelClassificater(output.last_hidden_state[:,0,:], labels)
		# logits: [batch_size, num_of_class]

		return loss, logits


class initModel(object):
	def __init__(self, num_of_class, config_file):
		self.num_of_class = num_of_class
		self.config = config_file
		self.model = Model(self.num_of_class, self.config)
	
	def init_trainer(self, learning_rate, num_warm_up, num_train_step):
		self.model = self.model.cuda()
		
		optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warm_up, num_train_step)
		
		return self.model, optimizer, scheduler
	
	def load_pretraining_model(self, init_ckpt, device=torch.device("cuda")):
		model_dict = self.model.state_dict()
		checkpoint = torch.load(init_ckpt, map_location=device)
		if 'model_state_dict' in checkpoint:
			pret_dict = checkpoint['model_state_dict']
		else:
			pret_dict = checkpoint
		
		pretrained_dict = {}
		for key, weight in pret_dict.items():
			if 'module.model.' in key:
				key = key.replace('module.model.', 'bert.')
			elif 'module.' in key:
				key = key.replace('module.', '')
			if key in model_dict:
				pretrained_dict[key] = weight
		
		model_dict.update(pretrained_dict)
		self.model.load_state_dict(model_dict)
		return
	
	def load_model(self, model_file, device=torch.device("cuda")):
		if device.type == "cpu":
			self.model.load_state_dict(torch.load(model_file, map_location=device))
		else:
			self.model.load_state_dict(torch.load(model_file))
			self.model = self.model.cuda()
		
		self.model = torch.nn.DataParallel(self.model)
		seed = 0
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
		return