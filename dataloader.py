import torch
import os
from time import time
import pickle

from dataset import ImageDataset, EvalImageDataset
from utils import AverageCounter, StdevCounter


def valid_data(data):
    if data[0]['case_label'] != 2:
        return True

    for view_data in data:
        if view_data['lesions'] and max(map(lambda x: x['label'], view_data['lesions'])) == 2:
            return True

    return False


def _get_data_from_pickle(pickle_path, total_fold_number, target_fold_number, image_prefix_path):
    # Load metadata
    with open(pickle_path, 'rb') as f:
        meta_data = pickle.load(f, encoding='iso-8859-1')

    train_data, val_data = [], []
    for i, (k, v) in enumerate(meta_data.items()):
        case_data = []
        for view_type in ['LMLO', 'LCC', 'RMLO', 'RCC']:
            d = {
                'case_id': k,
                'case_label': v['case_label'],
                'image_size': v[view_type]['image_size'],
                'lesions': v[view_type]['lesions']
            }
            image_keys = [k for k in v[view_type].keys() if '_image' in k]
            for imkey in image_keys:
                d['{}_path'.format(imkey)] = os.path.join(image_prefix_path, v[view_type][imkey])
            case_data.append(d)

        if i % total_fold_number + 1 == target_fold_number:
            val_data.extend(case_data)
        elif valid_data(case_data):
            train_data.extend(case_data)

    return train_data, val_data


def _get_stats(dataloader):
    mean_counter = AverageCounter()
    std_counter = StdevCounter()
    start_time = time()
    for i, (images, masks) in enumerate(dataloader):
        mean_counter(images)
        std_counter(images)
        print("{} / {}, took {} secs".format(i, len(dataloader), time() - start_time))
        start_time = time()

    stats = {'mean': mean_counter.value, 'std': std_counter.value}
    return stats


def get_loader(args):
    train_data, val_data = _get_data_from_pickle(
        args.data, args.fold_number, args.current_fold_number, args.prefix_image_dir_path
    )

    train_dataset = ImageDataset(train_data, args, use_compressed=args.use_compressed_train)
    val_dataset = EvalImageDataset(val_data, args, use_compressed=args.use_compressed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    # Get normalization statistics
    stats_pickle_path = 'dataset-stats-{}-{}.pkl'.format(args.fold_number, args.current_fold_number)
    if os.path.isfile(stats_pickle_path):
        print("=> loading dataset stats ...")
        with open(stats_pickle_path, 'rb') as f:
            stats = pickle.load(f)
    else:
        print("=> calculating dataset stats ...")
        stats = _get_stats(train_loader)
        # save statistics
        with open(stats_pickle_path, 'wb') as fp:
            pickle.dump(stats, fp)
    print("mean: {}, std: {}".format(stats['mean'], stats['std']))

    train_dataset.stats = stats
    val_dataset.stats = stats

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    return train_loader, val_loader
