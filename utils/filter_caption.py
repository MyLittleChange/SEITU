import json

caption_path='/raid/yq/UNITER/pretrain/caption_result_val_10.json'

with open(caption_path, 'r') as f:
    caption_dict = json.load(f)
caption_dict_filter=caption_dict
for img in caption_dict:
    caption=caption_dict[img][0]['caption']
    sents = caption.split('.')
    sent_set = set(sents)
    sent_new=[]
    for sent in sent_set:
        if sent=='':
            continue
        word_list = sent.strip().split()
        i = 0
        sen=''
        while i < len(word_list):
            if word_list[i] == 'person':
                if i < (len(word_list) - 2):
                    if word_list[i + 2].isdigit() and word_list[i+1]=='_':
                        sen+='person_'+word_list[i+2]+' '
                        i += 3
                        # person _ 0形式
                    else:
                        sen+=word_list[i]+' '
                        i += 1
                else:
                    sen += word_list[i] + ' '
                    i += 1
            else:
                sen += word_list[i] + ' '
                i += 1
        sen=sen[:-1]+'.'
        sent_new.append(sen)
    sent_new=list(set(sent_new))
    caption_dict_filter[img][0]['caption']=sent_new
json_str_new=json.dumps(caption_dict_filter)
caption_path='/raid/yq/UNITER/pretrain/caption_result_val_10_filter.json'
with open(caption_path, 'w') as f:
   f.write(json_str_new)