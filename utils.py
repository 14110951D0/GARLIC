import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import copy
from collections import Counter

from data.data_utils import *


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - pt) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_factor * focal_term * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def load_data(data):
    proc_path = f'./data/processed_data/{data}/'
    if not os.path.exists(proc_path):
        raw_path = f'./data/rawdata/{data}'
        os.makedirs(proc_path, exist_ok=True)
        process_func = {
            'P12': process_p12data,
            'P19': process_p19data,
            'MIMICIII': process_mimiciii
        }[data]
        processed_data = process_func(raw_path)
        print('Parsed data:', len(processed_data))
    else:
        processed_data = np.load(proc_path + 'tsdict_list.npy', allow_pickle=True)
        print('Loaded data:', len(processed_data))

    processed_data = z_score_normalize(processed_data)
    return processed_data


def load_data_params(data):
    return np.load(f'./data/processed_data/{data}/ts_params.npy', allow_pickle=True)


def get_data_split(data):
    labels = np.array([b['label'] for b in data])
    static = np.array([b['static'] for b in data])
    mask = [x in (0, 1) for x in static[0]]
    static_selected = static[:, mask]

    static_str = ["_".join(map(str, s)) for s in static_selected]
    combined = np.array([f"{s}_{l}" for s, l in zip(static_str, labels)])

    def is_valid_stratify(y, min_count=2):
        counts = Counter(y)
        return all(v >= min_count for v in counts.values())

    if is_valid_stratify(combined):
        stratify_main = combined
    else:
        stratify_main = labels

    train, temp, strat_train, strat_temp = train_test_split(
        data, stratify_main, test_size=0.2, random_state=0, stratify=stratify_main
    )

    if is_valid_stratify(strat_temp):
        stratify_temp = strat_temp
    else:
        stratify_temp = [b['label'] for b in temp]

    test, val = train_test_split(
        temp, test_size=0.5, random_state=0, stratify=stratify_temp
    )

    return train, val, test



def get_logger(logpath, filepath=__file__, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if saving:
        file_handler = logging.FileHandler(logpath, mode='w')
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.addHandler(file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.addHandler(console_handler)
    logger.info(filepath)
    return logger


def data_downsampling(data, rate):
    np.random.shuffle(data)
    return data[:int(len(data) * rate)]


def cal_tau(tp, mask):
    if tp.ndim == 2:
        tp = mask * np.expand_dims(tp, -1)
    b, l, k = tp.shape
    mask[:, 0, :] = 1
    tp[mask == 0] = np.nan
    tp = tp.transpose(1, 0, 2).reshape(l, -1)
    tp = pd.DataFrame(tp).ffill().values.reshape(l, b, k).transpose(1, 0, 2)
    tp[:, 1:] -= tp[:, :-1]
    return tp * mask


def batch_add_static(batch):
    batch = copy.deepcopy(batch)
    D = batch[0]['arr'].shape[1]
    D_1 = len(batch[0]['static'])
    for b in batch:
        static = np.zeros((len(b['mask']), D_1))
        static[0, :] = np.nan_to_num(b['static'], nan=0)
        static_mask = np.zeros((len(b['mask']), D_1))
        static_mask[0, :] = 1 - np.isnan(b['static']).astype(int)
        b['arr'] = np.concatenate((static, b['arr']), axis=1)
        b['mask'] = np.concatenate((static_mask, b['mask']), axis=1)

    for i in range(D + D_1):
        values = np.concatenate([b['arr'][:, i] for b in batch])
        max_val = np.max(values)
        if max_val != 0:
            for b in batch:
                b['arr'][:, i] /= max_val

    all_times = np.concatenate([b['time'] for b in batch])
    max_time = np.max(np.nan_to_num(all_times))
    for b in batch:
        b['time'] = np.nan_to_num(b['time']) / max_time if max_time != 0 else b['time']

    return batch


def z_score_normalize(orig_data):
    batch = copy.deepcopy(orig_data)
    D = batch[0]['arr'].shape[1]
    D_1 = len(batch[0]['extended_static'])

    # Normalize per variable
    for i in range(D):
        data = np.concatenate([b['arr'][b['mask'][:, i] == 1, i] for b in batch])
        if len(data) > 0:
            mean_value = np.mean(data)
            std_value = np.std(data)

            for b in batch:
                mask = b['mask'][:, i] == 1
                if std_value != 0:
                    b['arr'][mask, i] = (b['arr'][mask, i] - mean_value) / std_value
                else:
                    b['arr'][mask, i] = b['arr'][mask, i] - mean_value

    for i in range(D_1):
        data = np.concatenate([[b['extended_static'][i]] for b in batch])
        data = np.nan_to_num(data, nan=0)
        for b in batch:
            b['extended_static'][i] = np.nan_to_num(b['extended_static'][i], nan=0)
    for b in batch:
        b['time'] = np.nan_to_num(b['time'], nan=0)

    return batch

