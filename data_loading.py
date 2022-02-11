import numpy as np
import os
import pandas as pd
import pickle
import re
from tqdm import trange, tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from bert_transformers.tokenization_bert import BertTokenizer


class GPTFlowDataset(Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		return self.dataset[idx]


def create_tokens(dict_token_list, tokenizer, max_huffman_length, max_seq_length, op_dict):
	item_id = int(dict_token_list['item_id']) if isinstance(dict_token_list['item_id'], str) else dict_token_list[
		'item_id']
	token_list = eval(dict_token_list['token_list']) if isinstance(dict_token_list['token_list'], str) else \
		dict_token_list['token_list']
	huffman_list = eval(dict_token_list['huffman_list']) if isinstance(dict_token_list['huffman_list'], str) else \
		dict_token_list['huffman_list']
	type_list = eval(dict_token_list['type_list']) if isinstance(dict_token_list['type_list'], str) else \
		dict_token_list['type_list']
	edge_list = eval(dict_token_list['edge_list']) if isinstance(dict_token_list['edge_list'], str) else \
		dict_token_list['edge_list']
	accepted_op = eval(dict_token_list['accepted_op']) if isinstance(dict_token_list['accepted_op'], str) else \
		dict_token_list['accepted_op']
	label_op = eval(dict_token_list['label_op']) if isinstance(dict_token_list['label_op'], str) else \
		dict_token_list['label_op']

	
	input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + token_list + ["[SEP]"])

	attention_mask = [1 for _ in range(len(input_ids))]
	
	huffman = []
	huffman.append([0] * max_huffman_length)
	for code in huffman_list:
		cur_code = list(map(int, list(code)))

		if len(cur_code) > max_huffman_length:
			return None

		cur_code += [0] * (max_huffman_length - len(cur_code))
		huffman.append(cur_code)

	huffman += [[0] * max_huffman_length] 

	char_type = [0] + type_list + [0] 
	edges = edge_list
	
	op_labels = [0] * len(op_dict)
	for op in label_op:
		op_labels[op_dict[int(op)]] = 1
	
	labels = op_labels
	
	if len(input_ids) > max_seq_length:
		return None
	
	huffman += [[0] * max_huffman_length] * (max_seq_length - len(huffman))

	input_ids += [0] * (max_seq_length - len(input_ids))
	attention_mask += [0] * (max_seq_length - len(attention_mask))
	char_type += [0] * (max_seq_length - len(char_type))
	edges += [(0, 0) for _ in range(max_seq_length - len(edges))]
	
	dict_result = dict()
	dict_result['item_id'] = item_id
	dict_result['input_ids'] = input_ids
	dict_result['attention_mask'] = attention_mask
	dict_result['huffman'] = huffman
	dict_result['char_type'] = char_type
	dict_result['edges'] = edges
	dict_result['labels'] = labels

	return dict_result


def token_data_tree(df_row, tokenizer, max_seq_length, max_huffman_length, op_dict):
	try:
		dict_token_list = dict(df_row)
	except:
		print("error:", df_row)
		return None
	
	dict_result = create_tokens(dict_token_list, tokenizer, max_huffman_length, max_seq_length, op_dict)
	
	return dict_result


def create_dataset_raw_tree(input_file, logger, random_seed=12345):
	output_path = os.path.dirname(input_file)
	fileType = 'train'
	
	raw_folder = os.path.join(output_path, fileType)
	
	if not os.path.exists(raw_folder):  # exist pkl folder, but not sure whether pkl file exists
		os.makedirs(raw_folder)
	
	logger.info(f'Creating raw dataset for {input_file}')
	len_list = []
	data = pd.read_csv(input_file, low_memory=False)
	data = data.dropna()
	data = data.sample(frac=1, random_state=random_seed)
	num_lines = len(data)
	num_batch = 1000000
	num_raw = num_lines // num_batch + 1
	for idx in trange(num_raw):
		data_part = data[num_batch * idx: num_batch * (idx + 1)]
		len_list.append(len(data_part))
		raw_data = os.path.join(raw_folder, f'{fileType}{idx}')
		if not os.path.exists(raw_data):
			data_part.to_csv(raw_data, mode='w', index=False)
	
	return fileType, raw_folder, num_raw, len_list


def rm_cache_data(index, raw_folder, fileType):
	cached_data = os.path.join(raw_folder, f'{fileType}-cache-{index}-')
	os.system(f'rm {cached_data}*')


def generate_name(raw_folder, fileType, index):
	cached_name = f'{fileType}-cache-{index}-'
	cached_data_0 = os.path.join(raw_folder, cached_name + str(0))
	raw_data = os.path.join(raw_folder, f'{fileType}{index}')
	return cached_name, cached_data_0, raw_data


def gen_2D_attention_mask(attention_mask, edges):
	two_dimensional_attention_mask = torch.zeros_like(attention_mask, dtype=torch.long) # [B,L]
	two_dimensional_attention_mask = two_dimensional_attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
	for batch_idx, batch_edges in enumerate(edges):
		valid_num = torch.sum(attention_mask[batch_idx])
		for i in range(valid_num):
			two_dimensional_attention_mask[batch_idx, 0, i] = 1 
			two_dimensional_attention_mask[batch_idx, i, i] = 1 
		for edge in batch_edges:
			two_dimensional_attention_mask[batch_idx, edge[0]+1, edge[1]+1] = 1 
	
	two_dimensional_sub_attention_mask = two_dimensional_attention_mask.clone()
	for batch_idx, batch_edges in enumerate(edges):
		valid_num = torch.sum(attention_mask[batch_idx])
		dict_child = dict()
		for edge in batch_edges:
			parent_id = edge[0]
			child_id = edge[1]
			if int(child_id) == 0:
				continue
			if int(parent_id) not in dict_child:
				dict_child[int(parent_id)] = [int(child_id)]
			else:
				dict_child[int(parent_id)].append(int(child_id))
			
		dict_allchild = dict()
		
		def find_all_child(node_id):
			if node_id in dict_allchild:
				return dict_allchild[node_id]
			if node_id not in dict_child:
				dict_allchild[node_id] = []
				return []
			else:
				allchild_list = []
				for child_nodes in dict_child[node_id]:
					allchild_list.append(child_nodes)
					childs = find_all_child(child_nodes)
					allchild_list.extend(childs)
				dict_allchild[node_id] = allchild_list
				return allchild_list
		
		for i in range(valid_num):
			cur_all_child_list = find_all_child(i)
			for child_node in cur_all_child_list:
				two_dimensional_sub_attention_mask[batch_idx, i + 1, child_node + 1] = 1
		
	return two_dimensional_attention_mask, two_dimensional_sub_attention_mask


def collate_fn_2D_attention(batch):
	ret_dict = {'item_id':[], 'input_ids':[], 'attention_mask':[], 'huffman':[], 'edges':[], 'char_type':[], 'labels':[]}
	for dict in batch:
		for key in ret_dict.keys():
			ret_dict[key] += [torch.as_tensor(dict[key])]
	for key in ret_dict.keys():
		ret_dict[key] = torch.stack(ret_dict[key])
	
	# generate 2D attention mask from edges
	two_dimensional_attention_mask, two_dimensional_sub_attention_mask = gen_2D_attention_mask(ret_dict["attention_mask"], ret_dict["edges"])
	ret_dict["two_dimensional_attention_mask"] = two_dimensional_attention_mask
	ret_dict["two_dimensional_sub_attention_mask"] = two_dimensional_sub_attention_mask
	return ret_dict


class Create_Data(object):
	def __init__(self, logger, lines_cache, vocab_file, max_seq_length, max_huffman_length, op_dict):
		self.logger = logger
		self.lines_cache = lines_cache
		self.vocab_file = vocab_file
		self.max_seq_length = max_seq_length
		self.max_huffman_length = max_huffman_length
		self.tokenizer = BertTokenizer(vocab_file=vocab_file)
		self.op_dict = np.load(op_dict, allow_pickle='TRUE').item()
		self.feature_name = ['item_id', 'input_ids', 'attention_mask', 'huffman',  'edges', 'char_type', 'labels']
	
	def raw_to_cache(self, raw_folder, fileType, index, start=0):
		
		cached_name, cached_data_0, raw_data = generate_name(raw_folder, fileType, index)
		
		if not os.path.exists(cached_data_0):
			cached = cached_name + str(start)
			cached_path = os.path.join(raw_folder, cached)
			
			# logger.info(f'Creating cache for {raw_data}: {cached}')
			dataset = self.create_dataset_cache_tree(raw_data, self.lines_cache * start,
														self.lines_cache * (start + 1))
			with open(cached_path, "wb") as f:
				pickle.dump(dataset, f)
			while os.path.getsize(cached_path) <= 0:
				self.logger.warning(f'file {cached_path} writing empty: {len(dataset)}')
				dataset = self.create_dataset_cache_tree(raw_data, self.lines_cache * start,
															self.lines_cache * (start + 1))
				with open(cached_path, "wb") as f:
					pickle.dump(dataset, f)
		
		return
	
	def create_dataset_cache_tree(self, raw_data, start, end):
		dataset = []
		data = pd.read_csv(raw_data, low_memory=False)
		length = len(data)
		data = data[start:end] if end < length else data[start:]
		for index, row in data.iterrows():
			dict_result = token_data_tree(row, self.tokenizer, self.max_seq_length, self.max_huffman_length, self.op_dict)
			if dict_result:
				dict_current = dict()
				try:
					for name in self.feature_name:
						dict_current[name] = np.array(dict_result[name])
				except Exception as e:
					print(e)
				else:
					dataset.append(dict_current)
		return dataset
	
	def load_dataloader(self, raw_folder, fileType, index, batch_size):
		cached_name, cached_data_0, raw_data = generate_name(raw_folder, fileType, index)
		if not os.path.exists(cached_data_0):
			self.logger.error('No cache file!')
			return None
		else:
			dataset = []
			filesname = os.listdir(raw_folder)
			cache_list = [x for x in filesname if re.search(f'{cached_name}\d+', x)]
			for cached in tqdm(cache_list):
				cache_index = int(cached.replace(cached_name, ''))
				cached_path = os.path.join(raw_folder, cached)
				try:
					with open(cached_path, "rb") as f:
						data_tmp = pickle.load(f)
				except (pickle.UnpicklingError, ImportError, EOFError, IndexError, TypeError) as e:
					self.logger.error(e)
					self.logger.error(f'file {cached_path} size {os.path.getsize(cached_path)}')
					self.logger.info(f'recreating file {cached}')
					
					data_tmp = self.create_dataset_cache_tree(raw_data, self.lines_cache * cache_index,
																self.lines_cache * (cache_index + 1))
					with open(cached_path, "wb") as f:
						pickle.dump(data_tmp, f)
				finally:
					dataset.extend(data_tmp)
			
			GPTFdataset = GPTFlowDataset(dataset)
			dataloader = DataLoader(dataset=GPTFdataset, batch_size=batch_size, num_workers=16, shuffle=False, collate_fn=collate_fn_2D_attention,
									pin_memory=True)
		return dataloader
	
	def load_test_dataloader(self, test_file, batch_size):
		test_df = pd.read_csv(test_file, low_memory=False)
		test_cache_dir = os.path.join(os.path.dirname(test_file), 'test')
		if not os.path.exists(test_cache_dir):
			os.makedirs(test_cache_dir)
		test_cache_file = os.path.join(test_cache_dir, "test-cache")
		if os.path.exists(test_cache_file):
			print("Load from test cache...")
			try:
				with open(test_cache_file, "rb") as f:
					dataset = pickle.load(f)
			except (pickle.UnpicklingError, ImportError, EOFError, IndexError, TypeError) as e:
				self.logger.error(e)
				dataset = []
				for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
					dict_result = token_data_tree(row, self.tokenizer, self.max_seq_length, self.max_huffman_length,
												  self.op_dict)
					if dict_result:
						dict_current = {name: np.array(dict_result[name]) for name in self.feature_name}
						dataset.append(dict_current)
				with open(test_cache_file, "wb") as f:
					pickle.dump(dataset, f)
		else:
			print("Load from test csv...")
			dataset = []
			for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
				dict_result = token_data_tree(row, self.tokenizer, self.max_seq_length, self.max_huffman_length,
											  self.op_dict)
				if dict_result:
					dict_current = {name: np.array(dict_result[name]) for name in self.feature_name}
					dataset.append(dict_current)
			
			with open(test_cache_file, "wb") as f:
				pickle.dump(dataset, f)
		
		GPTFdataset = GPTFlowDataset(dataset)
		dataloader = DataLoader(dataset=GPTFdataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn_2D_attention, pin_memory=True)
		return dataloader
	