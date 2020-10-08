import transformers
import pandas as pd
import os, numpy as np
from transformers import MarianMTModel, MarianTokenizer
from time import time
from tqdm import tqdm


BATCH_SIZE = 10


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# def save_to_file(sents, dump_fname):
#     with open(dump_fname,'w') as fh:
#         print(sents, file=fh)


# def backup_save_to_file(sents, dump_fname):
#     dump_fname = dump_fname.replace('.txt', '_bkup.txt')
#     with open(dump_fname,'w') as fh:
#         for s in sents:
#             print(s, file=fh)


def trans_using_marianmt(text,tgt_lang):
    print('*' * 80)
    print(f'processing translation to: {tgt_lang}')
    trans_sents = []

    model_name = 'Helsinki-NLP/opus-mt-en-' + tgt_lang
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model = model.cuda()
    print(f'using tokenizer: {tokenizer} and \nmodel: {model_name}')

    data_batches = list(chunks(text[:], BATCH_SIZE))
    epoch_t0 = time()
    for batch_text in tqdm(data_batches):
        batch_text_inputs = tokenizer.prepare_seq2seq_batch(batch_text, max_length=512)
        batch_text_inputs = {k: v.cuda() for k, v in batch_text_inputs.items()}
        translated = model.generate(**batch_text_inputs)
        translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        trans_sents.extend(translated)
    full_tras_time = round(time() - epoch_t0)
    print(f'Translated all en sentences into {tgt_lang}, using {model_name} in {full_tras_time} sec')
    return trans_sents


def translate_csv_sentences(ip_csv_fname):

    #1. load DF
    df = pd.read_csv(ip_csv_fname)
    # df = df.sample(frac=0.01)
    print(f'Loaded dataframe form: {ip_csv_fname}')
    print(f'Dataframe shape: {df.shape}')

    # 2. load sentence pairs
    s1 = list(df.s1)
    s2 = list(df.s2)
    print(f'Len of the sentence lists: {len(s1)} and {len(s2)}')

    # 3. trans to each of the target languages
    tgt_langs = ['trk', 'es', 'ar']
    for t in tgt_langs:
        trans_s1 = trans_using_marianmt(s1,t)
        trans_s2 = trans_using_marianmt(s2,t)
        df[f's1_{t}'] = trans_s1
        df[f's2_{t}'] = trans_s2

    # 4. save the output
    op_csv_fname = ip_csv_fname.replace('.csv','_w_trans.csv')
    df.to_csv(op_csv_fname, index=False)
    print(f'Done translating and saving to {op_csv_fname}')

def main():
    data_dir = '/home/anna/xsts/data/stsbenchmark'
    files = ['pp_stsb_train.csv', 'pp_stsb_dev.csv', 'pp_stsb_test.csv']
    files = [os.path.join(data_dir,f) for f in files]
    for f in files:
        translate_csv_sentences(f)

if __name__ == '__main__':
    main()
