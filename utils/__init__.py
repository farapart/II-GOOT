import os
import numpy as np
# import ujson as json
import json
import torch


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()



def load_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        return json.load(f)



def load_line_json(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            yield json.loads(line)



def mkdir_if_not_exist(path):
    dir_name, file_name = os.path.split(path)
    if dir_name:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(json_obj, file_name):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode='w', encoding='utf-8-sig') as f:
        json.dump(json_obj, f, indent=4, cls=NpEncoder)




def append_json(file_name, obj, mode='a'):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode=mode, encoding='utf-8') as f:
        if type(obj) is dict:
            string = json.dumps(obj)
        elif type(obj) is list:
            string = ' '.join([str(item) for item in obj])
        elif type(obj) is str:
            string = obj
        else:
            raise Exception()

        string = string + '\n'
        f.write(string)


def append_new_line(file_name, string):
    mkdir_if_not_exist(file_name)
    with open(file_name, mode='a', encoding='utf-8') as f:
        string = string + '\n'
        f.write(string)


# model related

def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)

def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

class CustomBatchSampler:
    # 使active user 和非active user不在同一个batch中
    # 数据集处理时，已经将所有active user 的id放在非active user的id前
    def __init__(self, dataset, batch_size, drop_last):
        self.data = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        print(f'dataset length: {len(self.data)}')

    def __iter__(self):
        batch = []
        idx_list = list(range(len(self.data)))
        for i,idx in enumerate(idx_list):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(idx_list) - 1
                and self.data.is_active(i)
                != self.data.is_active(idx_list[i + 1])
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        group1 = [i for i in range(len(self.data)) if self.data.is_active(i) ]
        group2 = [i for i in range(len(self.data)) if not self.data.is_active(i)]

        if self.drop_last:
            return len(group1) // self.batch_size + len(group2) // self.batch_size
        else:
            return (len(group1) + self.batch_size - 1) // self.batch_size + (len(group2) + self.batch_size - 1) // self.batch_size