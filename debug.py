import json
import lmdb

meta = json.load(open('./pretrain/txt_db/vcr_val.db/meta.json', 'r'))
s_t=json.load(open('./pretrain/txt_db/vcr_val.db/special_tokens.json', 'r'))
id2len_qa = json.load(open('./pretrain/txt_db/vcr_val.db/id2len_qa.json', 'r'))
id2len_qar=json.load(open('./pretrain/txt_db/vcr_val.db/id2len_qar.json', 'r'))
txt2img = json.load(open('./pretrain/txt_db/vcr_val.db/txt2img.json', 'r'))
data_lmdb=lmdb.open('./pretrain/txt_db/vcr_val.db/data.mdb', max_readers=1, readonly=True,
                            lock=False, readahead=False, meminit=False)
lock_lmdb=lmdb.open('./pretrain/txt_db/vcr_val.db/lock.mdb', max_readers=1, readonly=True,
                            lock=False, readahead=False, meminit=False)
print('')
