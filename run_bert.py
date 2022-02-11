import argparse
from multiprocessing import Process
import math
import numpy as np
import os
from tqdm import trange, tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from logger import get_logger
from data_loading import Create_Data, create_dataset_raw_tree, rm_cache_data
from BertMod.model import initModel
from bert_transformers.configuration_bert import BertConfig


def setup_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def evaluate(model, test_dataloader):
	model.eval()
	total = 0
	hit = 0

	with torch.no_grad():
		for batch in tqdm(test_dataloader):
			# load batch data from dataloader
			input_ids = batch["input_ids"].cuda()  # [B, L]
			attention_mask = batch["attention_mask"].cuda()  # [B, L]
			two_dimensional_attention_mask = batch["two_dimensional_attention_mask"].cuda()  # [B, L ,L]
			huffman = batch["huffman"].cuda()  # [B, L, E]
			char_type = batch["char_type"].cuda() # [B, L]
			labels = batch["labels"].cuda()  # [B, len(op_dict)]

			# feed batch data to model
			loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, two_dimensional_attention_mask=two_dimensional_attention_mask,huffman_ids=huffman, type_ids=char_type, labels=labels, )


			probs = torch.sigmoid(logits)
			
			label_op_index = torch.argmax(labels, dim=-1)
			pred_op_index = torch.argmax(probs, dim=-1)

			hit += torch.sum(label_op_index == pred_op_index).item()
			total += labels.size(0)
			
	pt_logger.info(f"Whole-tree Evaluation begins...")
	pt_logger.info(f"Total:{total}, hit:{hit}, ACC:{hit / total}")
	print(f"Total:{total}, hit:{hit}, ACC:{hit / total}")


def main():
	configFile = BertConfig.from_json_file(args.config_file)

	# prepare raw dataset: split to seperate data files
	fileType, raw_folder, num_raw, len_list = create_dataset_raw_tree(args.train_file, pt_logger)
	pt_logger.info(f'number of raw files: {num_raw}, total lenth of data: {sum(len_list)}')
	
	num_train_step = (args.num_of_epoch) * sum(len_list) // args.train_batch_size
	num_warm_up = num_train_step // 10
	pt_logger.info(f'training step: {num_train_step}, warmup step: {num_warm_up}, num of batch: {args.train_batch_size}')
	
	Bert_Model = initModel(num_of_class, configFile)
	
	# if pretraining ckpt is given
	if args.pretrained_model and os.path.exists(args.pretrained_model):
		pt_logger.info(f'loading initial checkpoint {args.pretrained_model}')
		Bert_Model.load_pretraining_model(args.pretrained_model)
	
	model, optimizer, scheduler = Bert_Model.init_trainer(args.learning_rate, num_warm_up, num_train_step)
	
	model = torch.nn.DataParallel(model)
	
	lines_cache = 50000
	DataCreate = Create_Data(pt_logger,lines_cache, args.vocab_file, args.max_seq_length, configFile.max_tree_layers, os.path.join(args.output_dir_data, 'op_dict.npy'))
	

	for cur_index in range(num_raw):
		fileType = 'train'
		# create first raw data's cache
		process_list = []
		# number of cache
		num_cache = math.ceil(len_list[cur_index] / lines_cache)
		for i in range(num_cache):
			p = Process(target=DataCreate.raw_to_cache, args=(raw_folder, fileType, cur_index, i))
			process_list.append(p)
			p.start()
		for p in process_list:
			p.join()

	# load test data
	test_dataloader = DataCreate.load_test_dataloader(args.test_file, args.eval_batch_size)

	cur_index = 0
	steps = 0
	for epoch in trange(args.num_of_epoch):
		model.train()
		
		epoch_loss = 0
		num_batches = 0
		last_loss = 0
		for index in trange(cur_index, num_raw):
			dataloader_batch_num = 0
			dataloader_loss = 0
			# load cache
			train_dataloader = DataCreate.load_dataloader(raw_folder, fileType, index, args.train_batch_size)
			
			if not train_dataloader:
				continue
			
			if index != num_raw - 1:
				next_file_index = index + 1
			else:
				next_file_index = 0
			
			# create next raw data's cache
			process_list = []
			num_cache = math.ceil(len_list[next_file_index] / lines_cache)
			for i in range(num_cache):
				p = Process(target=DataCreate.raw_to_cache, args=(raw_folder, fileType, next_file_index, i))
				process_list.append(p)
				p.start()
			
			for batch in tqdm(train_dataloader):
				steps += 1
				dataloader_batch_num += 1

				# load batch data from dataloader
				input_ids = batch["input_ids"].cuda()  # [B, L]
				attention_mask = batch["attention_mask"].cuda()  # [B, L]
				two_dimensional_attention_mask = batch["two_dimensional_attention_mask"].cuda()  # [B, L ,L]
				huffman = batch["huffman"].cuda()  # [B, L, E]
				char_type = batch["char_type"].cuda() # [B, L]
				labels = batch["labels"].cuda()  # [B, L]
				
				# feed batch data to model
				loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, two_dimensional_attention_mask=two_dimensional_attention_mask, huffman_ids=huffman, type_ids=char_type, labels=labels)

				loss = loss.mean().cuda()
				epoch_loss += loss.item()
				dataloader_loss += loss.item()
				num_batches += 1
				if not (steps % 100):
					last_loss = loss.item()
					print(f'Epoch-{epoch} Current step training loss {last_loss}, {steps}')
					writer.add_scalars('loss', {'Current step training loss': last_loss}, steps)
				
				# compute gradiant
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
				optimizer.step()
				scheduler.step()
				model.zero_grad()
			
			del train_dataloader
			
			for p in process_list:
				p.join()
			
			# rm_cache_data(index, raw_folder, fileType)
			pt_logger.info(f" Epoch-{epoch} dataloader {index} loss: {dataloader_loss / dataloader_batch_num}")
		
		cur_index = 0
		
		pt_logger.info(f" Epoch-{epoch} loss: {epoch_loss / num_batches}. Current step loss: {last_loss}")
		writer.add_scalars('loss', {'Cumulative training loss': epoch_loss / num_batches}, steps)
		
		# save model state_dict
		torch.save(model.module.state_dict(), f"{args.output_dir_model}/Epoch_{epoch}")
		
		# eval model
		print("Eval...")
		evaluate(model, test_dataloader)
		

if __name__ == '__main__':
	setup_seed(1340)
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_file", help="The training file.", type=str,
						default="training_data/output/train_tree.csv")
	parser.add_argument("--test_file", help="The test file.", type=str, default="training_data/output/test_tree.csv")
	parser.add_argument("--tensorboard", help="Location for tensorboard", type=str, default="tensorboard")
	parser.add_argument("--pretrained_model", help="The path of the pretrained model if exists.",
						type=str, default='')
	parser.add_argument("--output_dir_model", help="The output directory where the model checkpoints will be written.",
						type=str, default="op_model")
	parser.add_argument("--output_dir_data", help="The output directory where the data files will be written.",
						type=str, default="training_data/output")
	parser.add_argument("--output_dir_log", help="The output directory where the log files will be written.", type=str,
						default="log")
	parser.add_argument("--vocab_file", help="The vocabulary file that the BERT model was trained on.", type=str,
						default="vocabs/vocab_tree.txt")
	parser.add_argument("--config_file", help="The config json file corresponding to the pre-trained BERT model. ",
						type=str, default="config/config_tree.json")
	parser.add_argument("--max_seq_length", help="The max length of sequence", type=int, default=128)
	parser.add_argument("--train_batch_size", help="Train batch size", type=int, default=512)
	parser.add_argument("--eval_batch_size", help="Evaluation batch size", type=int, default=128)
	parser.add_argument("--learning_rate", help="Learning rate", type=int, default=5e-5)
	parser.add_argument("--num_of_epoch", help="Number of epoch", type=int, default=30)
	
	args = parser.parse_args()
	
	pt_logger = get_logger("op_pre", os.path.join(args.output_dir_log, "pt_logger.log"))
	writer = SummaryWriter(args.tensorboard)
	
	op_dict = np.load(f'{args.output_dir_data}/op_dict.npy', allow_pickle='TRUE').item()
	op_reverse_dict = {value: key for key, value in op_dict.items()}
	num_of_class = len(op_dict)
	
	main()