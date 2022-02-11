# Learn2Compute

Released code for KDD, 2022.
 
## Dataset

For the experiments, we prepared a dataset for **symbolic computation**, which will be released afterward. The dataset consists of two sub-datasets, self-generated data and collected data from high-school mathematic teaching materials. The self-generated dataset includes math expressions, transformations that could be applied on those expressions and the corresponding results, and is used for model training and evaluation. The collected dataset only includes 300 pairs of math expression and its targeted expressions, which is used for validation. 

In this repo, the program takes two csv files ([train_tree.csv](training_data/output/train_tree.csv) and [test_tree.csv](training_data/output/test_tree.csv)) as input, which saved the features of expressions in tree format. The original LaTeX-format dataset for math expressions will be released later. 


## Files You Need

- Files in [bert_transformers](bert_transformers): Basic model structure for BERT model. 
- Files in [BertMod](BertMod): Model structure we proposed (GATE).
- [configuration](config/config_tree.json): Configuration files for model.
- [main.sh](shell/main.sh): Shell script to run the training program.
- [training_settings.conf](shell/training_settings.conf): predefined configuration settings for training program. 
- [training_data](training_data): Necessary datasets for the program. 
- [vocabulary](vocabs/vocab_tree.txt): vocabulary files for the model. 


## How to Run

```python3
sh shell/main.sh
```

### Attentions

- All changeable configuration settings are included in config file [training_settings.conf](shell/training_settings.conf).

```bash
op_data_Path='training_data'  # folder path to save datasets 
output_dir_model='op_model'  # folder path to save trained model checkpoints
output_dir_event='tensorboard'  # folder path to save tensorboard events
output_dir_log='log'  # folder path to save training log

num_of_epoch=100  # self-defined training epoch
single_batch=32  # self-defined batch size per gpu
gpu_num=4  # number of gpu available
max_seq_length=256  #self-defined max sequence length for model input 
pretrained_ckpt_path=''  # file path of pretraining model checkpoint

vocab_file=vocabs/vocab_tree.txt  # vocabulary file for pretrained BERT model
config_file=config/config_tree.json  # configuration file for pretrained BERT model
```

- In script [main.sh](shell/main.sh), the config settings are read and passed to python script [run_bert.py](run_bert.py) as input for model training.

```bash
python3 run_bert.py \
	--train_file=${output_dir_data}/train_tree.csv \  # training data created from data.csv
	--test_file=${output_dir_data}/test_tree.csv \  # evaluation data created from test300.csv
	--tensorboard=${output_dir_event} \  # folder path to save tensorboard events
	--pretrained_model=${pretrained_ckpt_path} \  # pretraining model ckpt path if exists
	--output_dir_model=${output_dir_model} \  # folder path to save trained model ckpt
	--output_dir_data=${output_dir_data} \  # folder path to save data temp pkls while training
	--output_dir_log=${output_dir_log} \  # folder path to save training log file
	--vocab_file=${vocab_file} \  # vocabulary file for pretrained BERT model
	--config_file=${config_file} \  # config file for pretrained BERT model
	--max_seq_length=${max_seq_length} \  # self-defined max sequence length for model input
	--train_batch_size=$((single_batch * gpu_num)) \  # self-defined training batch size. Here, single_batch is the batch size on each single gpu. 
	--eval_batch_size=$((single_batch * gpu_num)) \  # self-defined evaluation batch size. 
	--num_of_epoch=${num_of_epoch}  # self-defined number of epoch to be trained.
```

