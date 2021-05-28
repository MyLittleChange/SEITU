import json
from tqdm import tqdm
from transformers.models.bert import BertTokenizer
comet_path = '/raid/yq/visualcompet/experiments/image-inference-80000-ckpt/test_sample_1_num_5_top_k_0_top_p_0.9.json'
with open(comet_path, 'r') as f:
    data = json.load(f)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
for line in tqdm(data):
    text = line['generations']
    generations_ids=[]
    for sent in text:
        ids=[]
        word_list = sent.strip().split()
        # 原句中没有主语,主语就是region_id
        for i in range(len(word_list)):
            if word_list[i].isdigit() and int(word_list[i]) <= 81:
                ws_id = 28995 + int(word_list[i])
                ids.extend([ws_id])
            else:
                ws = tokenizer.tokenize(word_list[i])
                if not ws:
                    # some special char
                    continue
                ids.extend(tokenizer.convert_tokens_to_ids(ws))
        generations_ids.append(ids)
    line['generations_ids']=generations_ids
json_str=json.dumps(data)
comet_path = '/raid/yq/visualcompet/experiments/image-inference-80000-ckpt/test_sample_1_num_5_top_k_0_top_p_0.9_ids.json'
with open(comet_path, 'w') as f:
    f.write(json_str)