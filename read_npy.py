import numpy as np
import os
import json
def get_gt_annot(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if os.path.splitext(file)[1] == '.json':
                files.append(os.path.join(dirpath, file))
    res= {}
    for file in files:
        # dic={}
        with open(file, 'r') as f:
            data = json.load(f)
        # boxes = []
        # for box in data['boxes']:
        #     x = box[0]
        #     y = box[1]
        #     w = box[2] - box[0]
        #     h = box[3] - box[1]
        #     boxes.append([x, y, w, h])
        # boxes = np.array(boxes)
        # dic['file_name'] = file.split('/')[-1][:-5] + '.jpg'
        # file_path = ''
        # for dir in file.split('/')[:-1]:
        #     file_path += str(dir)
        #     file_path += '/'
        # dic['file_path'] = os.path.join(file_path, dic['file_name'])
        # dic['bbox'] = boxes
        # dic['num_box'] = len(data['boxes'])
        res[file.split('/')[-1][:-5]]=data['names']
    output_dir = os.path.join('/raid/yq/UNITER/pretrain', 'gt_annot.json')
    with open(output_dir, "w") as f:
        json.dump(res, f)
        print("写入文件完成...")


if __name__ == "__main__":
    #npy_path='gt.npy'
    get_gt_annot('/raid/yq/VCR/vcr1images/')