import sys
import os
import torch
from datasets import Dataset,load_dataset, DownloadConfig, load_from_disk
from transformers import AutoTokenizer
import datasets
from utils import CfgNode as CN


def get_config():

    C = CN()

    C.dataset = CN()
    C.dataset.path = '/gpfs/u/home/LSMC/LSMCshnk/scratch/yikang/datasets/tiiuae/falcon-refinedweb' #'wikitext'#'tiiuae/falcon-refinedweb'
    #python3 /Users/guoyd/Downloads/SparseGPT/preprocess_data.py
    C.dataset.name = 'text' #wikitext-2-v1'#'text'
    C.dataset.num_proc = 8
    C.dataset.tokenizer = 'EleutherAI/pythia-410m' #/Users/guoyd/Downloads/SparseGPT
    C.dataset.cache_dir = '/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/datasets/dataset_cache' #/gpfs/u/home/LSMC/LSMCshnk/scratch

    C.model = CN()
    C.model.block_size = 8192

    C.trainer = CN()
    C.trainer.continuous_chunks = 48

    return C


def load_tokenizer(config):
    print("Loading vocab...")
    vocab_path = os.path.join('/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/datasets', 'tiiuae/falcon-refinedweb', 'tokenizer')
    if os.path.exists(vocab_path):
        tokenizer = AutoTokenizer.from_pretrained(vocab_path)
    else:
        print("Downloading vocab...")
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)
        os.makedirs(vocab_path)
        tokenizer.save_pretrained(vocab_path)
    return tokenizer


def preprocess_data(config, tokenizer):
    dataset_path = os.path.join('/dccstor/obsidian_llm/yiduo/', 'pubmed') 
    tokenized_dataset_path =  dataset_path + 'pubmed_tokenized' #dataset_path + 'pile_llama_v2_tokenized' #'medical_new_gemma_tokenized'
    bsize = config.model.block_size * config.trainer.continuous_chunks
    chunked_dataset_path = dataset_path + ' pile_pubmed_llamav2_hq_chunked_%d' % bsize #50b_v2
    import pdb
    from math import inf
    from model import KenlmModel
    ngram_model = KenlmModel.from_pretrained("wikipedia", "en") #model = KenlmModel.from_pretrained("wikipedia", "en")
    if os.path.exists(tokenized_dataset_path):

        print("Loading tokenized dataset...")
        tokenized_dataset = load_from_disk(tokenized_dataset_path)
    else:
        print("Loading dataset...")
        data=[]
        with open('the-pile-pubmed-abstract-refine-result.jsonl', 'r') as file:
            for line in file:
                data.append(json.loads(line))
        with open('the-pile-pubmed-central-refine-result.jsonl', 'r') as file:
            for line in file:
                data.append(json.loads(line))
        raw_dataset=Dataset.from_dict({key: [str(dic[key]) for dic in data] for key in data[0]}) 
        #dict_keys(['meta', 'text', 'stats', 'simhash']) 
        print("Encoding dataset...")
        def get_ppl(examples):
            content=examples['text'] #examples['input_ids'] #tokenizer(examples['content'],return_tensors="pt")['input_ids']
            if content=='.' or content=='./n' or content=='.\n' or content=='\n.' or content ==', .\n':
                return {
                'ppl': inf,}
            elif '_________________' in content or '__________'in content or '*************************'in content or '.........' in content or '__________' in content or '~~~~~~~~~~~~~~' in content or '===========' in content or '----------------' in content:
                return {
                'ppl': inf,
            }
            elif '\n\n\n' in content:
                return {'ppl':inf,} #'_________________' in content or '__________'in content or '**********************$
            elif '....' in content:
                return {'ppl':inf,}
            elif len(content)<50:
                return {
                'ppl': inf,
            }
            else:
                return {
                'ppl': ngram_model.get_perplexity(content),
            }
        raw_dataset = raw_dataset.map(get_ppl,num_proc=32,desc="calculating ppl on dataset",)
        def get_ppl_batch(examples):
            return {
            'ppl': [get_ppl(example) for example in examples],
             }
        def process_example(id, example):
            ppl_result = get_ppl(example)
            raw_dataset['train'][id]['ppl'] = ppl_result['ppl']
        tokenizer_llama = AutoTokenizer.from_pretrained('/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b') #'google/gemma-2b') #'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T') #'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T') #    desc="calculating ppl on dataset",) #
        raw_dataset=raw_dataset.sort('ppl')
        pdb.set_trace() #        )
        def encode(examples):
            import pdb
           # pdb.set_trace()
        #    print('###',tokenizer(examples['content']))
            return {
                'input_ids': tokenizer_llama(examples['text'])['input_ids'],
            }
        tokenized_dataset = raw_dataset.map(
            encode, 
            batched=True, 
            remove_columns=['text','meta','stats', 'simhash'], #, 'ppl','dump', 'segment', 'image_urls'], 
            num_proc=16, #32, #64,
            desc="Running tokenizer on dataset",
            writer_batch_size=200)
        print("Saving dataset to disk...")
        tokenized_dataset.save_to_disk(
            dataset_path + '_tokenized',
            max_shard_size="10GB"
            )
    import torch
    from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
    def get_ppl(examples):
            #import pdb
            #pdb.set_trace() #tokenizer(examples['content'])['input_ids']
            tokenized_content=examples['input_ids'] #tokenizer(examples['content'],return_tensors="pt")['input_ids']
            seq_lengths = torch.tensor([len(seq) for seq in tokenized_content])
            tokenized_content = [torch.tensor(seq) for seq in tokenized_content]
            tokenized_content = pad_sequence(tokenized_content, batch_first=True, padding_value=0)
            #tokenized_content = pack_padded_sequence(tokenized_content, seq_lengths, batch_first=True, enforce_sorted=False) #pad_sequence(tokenized_content, batch_first=True, padding_value=0) #pdb.set_trace()
            #if len(tokenized_content) > max_seq_length:
            #    x_i = torch.tensor(tokenized_content[:max_seq_length-1]).unsqueeze(0)
            #attention_mask = (x_i != -1).float() #    y_i = torch.tensor(tokenized_content[1:max_seq_length]).unsqueeze(0)
            #pdb.set_trace() #else:
        # Pad sequences with a special token (assuming 0 is the special token)
            #    padding_length = max_seq_length - len(tokenized_content)
            #    x_i = torch.tensor(tokenized_content + [0] * padding_length).unsqueeze(0)
            #    y_i = torch.tensor(tokenized_content + [0] * padding_length).unsqueeze(0) 
            x_i=tokenized_content[:,:-1]
            attention_mask = (x_i != 0).float() #pdb.set_trace()
            y_i=tokenized_content[:,1:]
            pdb.set_trace()
            logits=ngram_model(input_ids=x_i,labels=y_i,attention_mask=attention_mask)[1]
            #pdb.set_trace() #logits=outputs[1]
            from torch.nn import functional as F
            ppls = torch.exp(F.cross_entropy(logits.view(-1, logits.size(-1)), y_i.reshape(-1), ignore_index=-1, reduction='none').mean())
            pdb.set_trace()
            return {
                'ppl': ppls.item(),
            } # chunk dataset
    import pdb
    #from torch.nn.parallel import DataParallel #pdb.set_trace()
    #ngram_model = ngram_model.to("cpu")
    #ngram_model = DataParallel(ngram_model)
#    tokenized_dataset = tokenized_dataset.map(get_ppl,batched=True,batch_size=200,desc="calculating ppl on dataset",) #num_processes = 4 #raw_dataset = raw_dataset.map(
       # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
       #     futures = [executor.submit(process_example, id, example) for id, example in enumerate(raw_dataset['train'])] #    get_ppl, 
        #concurrent.futures.wait(futures) #    num_proc=32,
        #    desc="calculating ppl on dataset",) #
 #   tokenized_dataset=tokenized_dataset.sort('ppl')
  #  pdb.set_trace()
    def sort_by_timestamp(example):
        return example['timestamp'].month
    #pdb.set_trace()
    #print(tokenized_dataset)
    #for year_id in range(10):
    #    def filter_examples(example):
    #        return example if str(2013+year_id) in str(example['timestamp']) else []
    #    print(year_id+2013,tokenized_dataset.filter(filter_examples,num_proc=config.dataset.num_proc*8,batched=True,writer_batch_size=500))
    #pdb.set_trace()
    #tokenized_dataset=tokenized_dataset.filter(filter_examples,remove_columns=[ 'url', 'dump', 'segment', 'image_urls'],num_proc=config.dataset.num_proc*16)
    #tokenized_dataset = tokenized_dataset.remove_columns(['timestamp'])
    #pdb.set_trace()
    #except:
    #pdb.set_trace()
    #indices_to_keep = [i for i, example in enumerate(tokenized_dataset) if '2021' in str(example['timestamp'])]
    #tokenized_dataset = tokenized_dataset.select(indices_to_keep)
    #pdb.set_trace()
    def encode(example):
            import pdb
           # pdb.set_trace()
        #    print('###',tokenizer(examples['content']))
            return {
                'month': example['timestamp'].month,
            }
    #tokenized_dataset = tokenized_dataset.map(
    #        encode, 
            #batched=True, 
            #remove_columns=['content','url', 'dump', 'segment', 'image_urls'], 
    #        num_proc=128,
    #        desc="Running tokenizer on dataset",)
    from torch.nn import functional as F #bsize=2048 #        #writer_batch_size=100)
    def group_texts(examples):
        chunks = []
        remaining = []
        for sentence in examples["input_ids"]:
            sentence = remaining + sentence
            length = len(sentence)
           # import pdb
            #pdb.set_trace()
            chunked_length = ((length - 1) // bsize) * bsize
            chunks += [sentence[i:i + bsize + 1] for i in range(0, chunked_length, bsize)]
            remaining = sentence[chunked_length:]
        return {"input_ids": chunks}
    def get_ppl(examples):
            #import pdb
            #pdb.set_trace() #tokenizer(examples['content'])['input_ids']
            tokenized_content=examples['input_ids'] #tokenizer(examples['content'],return_tensors="pt")['input_ids']
            x_i=torch.tensor(tokenized_content).unsqueeze(0)[:,:-1]
            y_i=torch.tensor(tokenized_content).unsqueeze(0)[:,1:]
            logits=ngram_model(input_ids=x_i,labels=y_i)[1]
            #logits=outputs[1]
            #from torch.nn import functional as F
            ppls = torch.exp(F.cross_entropy(logits.view(-1, logits.size(-1)), y_i.reshape(-1), ignore_index=-1, reduction='none')).mean()
            return {
                'ppl': ppls.item(),
            }
    def get_ppl_batch(examples):
     #       pdb.set_trace()
            return {
            'ppl': [get_ppl(example) for example in examples['input_ids']],
             }
    #pdb.set_trace()
    # tokenized_dataset = tokenized_dataset.map(get_ppl,desc="calculating ppl on dataset") #tokenized_dataset=tokenized_dataset.sort('month')
    #tokenized_dataset=tokenized_dataset.sort('ppl') #tokenized_dataset=tokenized_dataset.remove_columns(['timestamp'])
    #pdb.set_trace() #tokenized_dataset=tokenized_dataset.remove_columns(['month'])
    #bsize=2048 #pdb.set_trace() #;import pdb
    #lm_dataset = lm_dataset.map(get_ppl,batched=True,batch_size=5,desc="calculating ppl on dataset") 
    #lm_dataset=lm_dataset.sort('ppl')
    pdb.set_trace()
    lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            writer_batch_size=1000,
            num_proc=config.dataset.num_proc*16,
            desc=f"Grouping texts in chunks of {bsize}",
            #remove_columns=list(tokenized_dataset.column_names.values())[0],
        ) 
    pdb.set_trace()
    '''
    pdb.set_trace()
    
    lm_dataset=None
    for month_id in range(1,13):
        month_dataset=tokenized_dataset.filter(lambda example: example['month'] == month_id)
        month_dataset = month_dataset.remove_columns(['timestamp'])
        month_dataset = month_dataset.remove_columns(['month'])
        #pdb.set_trace()
        
        lm_dataset_new = month_dataset.map(
            group_texts,
            batched=True,
            writer_batch_size=100,
            num_proc=config.dataset.num_proc*16,
            desc=f"Grouping texts in chunks of {bsize}",
            remove_columns=list(tokenized_dataset.column_names.values())[0],
        )
        if lm_dataset is None:
            lm_dataset=lm_dataset_new
        else:
            lm_dataset=datasets.concatenate_datasets([lm_dataset,lm_dataset_new])
    lm_dataset_shuffled = lm_dataset.shuffle()
   # import pdb
   # pdb.set_trace()
    #tokenized_dataset = tokenized_dataset.remove_columns(['timestamp'])
    '''
    #pdb.set_trace()
    #from datasets import load_from_disk
    #lm_dataset=load_from_disk('/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/datasets/RedBook_3_chunked_393216') #'/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/datasets/tiiuae/falcon-refinedweb/textmedical_full_openllama_v2_5b_chunked_98304') #"/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/datasets/tiiuae/falcon-refinedweb/textmedical_full_openllama_v2_50b_chunked_98304")
#    tokenizer_llama = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
 #   tokenizer_gemma = AutoTokenizer.from_pretrained('google/gemma-2b')##tokenizer_llama(tokenizer.decode(lm_dataset['train'][0]['input_ids'][:100]))['input_ids']
    #pdb.set_trace()
    inputs_list=[]
    def encode_llama(examples):
        text=tokenizer.decode(examples['input_ids'])
        return {'input_ids':tokenizer_llama(text)['input_ids'],} 
    #for id in range(len(lm_dataset['train'])):
    #    inputs_list.append(encode_llama(lm_dataset['train'][id]))
    #    print(id)
    def encode_llama(examples):
        text=tokenizer.decode(examples['input_ids'])
        return {'input_ids':tokenizer_llama(text)['input_ids'],} 
    #pdb.set_trace() #print("Saving dataset to disk...")
    #chunked_dataset_path='/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/datasets/RedBook_3_95MB_chunked_393216' #'/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/datasets/tiiuae/falcon-refinedweb/textmedical_full_openllama_v2_5b_95MB_chunked_98304'
    #lm_dataset = lm_dataset.map(encode_llama,num_proc=16)
    lm_dataset_shuffled = lm_dataset.shuffle(seed=2022)
    #pdb.set_trace()
    #pdb.set_trace() #lm_dataset_shuffled=lm_dataset
    lm_dataset_shuffled.save_to_disk(
        chunked_dataset_path,
        max_shard_size="95MB",
        num_proc=100, #config.dataset.num_proc,
        )

    return lm_dataset

        
if __name__ == '__main__':
    config = get_config()
    config.merge_from_args(sys.argv[1:])

    tokenizer = load_tokenizer(config)
    preprocess_data(config, tokenizer)
