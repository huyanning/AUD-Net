import numpy as np
import utils as hyper
import scipy.io as sio
import torch
import torch.utils.data as Data
import os
import lmdb
import os.path as osp
import pyarrow as pa


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.center = next(self.loader)
        except StopIteration:
            self.data = None
            self.center = None
            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.center = self.center.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        center = self.center
        self.preload()
        return data, center


# def writeCache(env, cache):
#     with env.begin(write=True) as txn:
#         for k, v in cache.items():
#             txn.put(k.encode(), v.encode())
#
#
# def save_to_lmdb(save_path, input, sample_num):
#     """
#     :param save_path: lmdb path(dir, not file)
#     :param imgs: img path and label list
#     """
#     db = lmdb.open(save_path, map_size=1073741824*5)
#     txn = db.begin(write=True)
#     cache = {}
#     data, center = input
#
#     # count = 0
#     for i in range(sample_num):
#         dataKey = 'image-%09d' % i
#         centerKey = 'label-%09d' % i
#         cache[dataKey] = str(data[i, :, :, :].reshape(-1).tolist()).strip('[').strip(']')
#         cache[centerKey] = str(center[i, :, :, :].reshape(-1).tolist()).strip('[').strip(']')
#         if i % 1000 == 999:
#             writeCache(db, cache)
#             cache = {}
#             print('Written %d / %d' % (i+1, input[0].shape[0]))
#     cache['num-samples'] = str(sample_num)
#     writeCache(db, cache)
#     print('Created dataset with %d samples' % sample_num)


class LmdbDataset(Data.Dataset):
    def __init__(self, dir_path, name):
        super(LmdbDataset, self).__init__()
        db_path = osp.join(dir_path, "%s.lmdb" % name)
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data, center = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        data = unpacked[0]
        center = unpacked[1]

        return data, center

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(input, dpath, sample_num, name="train", write_frequency=5000):
    directory = osp.expanduser(osp.join(dpath, name))
    print("outputting dataset to %s" % directory)
    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    data, center = input
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1073741824*5, readonly=False,
                   meminit=False, map_async=True)
    # print(len(data), len(center))
    txn = db.begin(write=True)

    for i in range(sample_num):
        txn.put(u'{}'.format(i).encode('ascii'), dumps_pyarrow((data[i], center[i])))
        if i % write_frequency == 0:
            print('Written %d / %d' % (i+1, input[0].shape[0]))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(i + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def generate_train_test(data3d, inner_size, outer_size, datan_size, train_num, log_dir):
    output_path = "data"
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        train_data = checkpoint['train_data']
        test_data = checkpoint['test_data']
        print("Successful loading data!")
        return train_data, test_data
    else:
        print("Generating data!")
        rows, cols, bands = data3d.shape
        test_data, test_center = hyper.dual_win(data3d, inner_size, outer_size, datan_size)
        neibor_num = outer_size ** 2 - inner_size ** 2
        test_data = torch.from_numpy(test_data)
        test_data_ar = torch.zeros((test_data.shape[0] * neibor_num, bands, datan_size, datan_size))
        test_center = torch.from_numpy(test_center).unsqueeze(1).expand(rows * cols, neibor_num, bands,
                                                                      datan_size, datan_size)
        test_center_ar = torch.zeros((rows * cols * neibor_num, bands, datan_size, datan_size))
        for i in range(rows*cols):
            test_data_ar[i * neibor_num:(i + 1) * neibor_num, :, :, :] = test_data[i, :, :, :, :]
            test_center_ar[i * neibor_num:(i + 1) * neibor_num, :, :, :] = test_center[i, :, :, :, :]

        index = np.random.randint(rows * cols, size=train_num)
        train_data_ar = torch.zeros((train_num * neibor_num, bands, datan_size, datan_size))
        train_center_ar = torch.zeros((train_num * neibor_num, bands, datan_size, datan_size))
        for i in range(train_num):
            train_data_ar[i * neibor_num:(i + 1) * neibor_num, :, :, :] = test_data[index[i], :, :, :, :]
            train_center_ar[i * neibor_num:(i + 1) * neibor_num, :, :, :] = test_center[index[i], :, :, :, :]

        folder2lmdb((train_data_ar.unsqueeze(1).numpy(), train_center_ar.unsqueeze(1).numpy()), output_path, train_num*neibor_num)
        train_data = Data.TensorDataset(train_data_ar.unsqueeze(1), train_center_ar.unsqueeze(1))
        test_data = Data.TensorDataset(test_data_ar.unsqueeze(1), test_center_ar.unsqueeze(1))
        checkpoint = {'train_data': train_data, 'test_data': test_data}
        if not os.path.isdir('log'):
            os.mkdir('log')
        torch.save(checkpoint, log_dir)
        return train_data, test_data


def generate_data(data3d, inner_size, outer_size, datan_size, train_num, log_dir):
    output_path = "data"
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        train_data = checkpoint['train_data']
        test_data = checkpoint['test_data']
        print("Successful loading data!")
        return train_data, test_data
    else:
        print("Generating data!")
        rows, cols, bands = data3d.shape
        r_inner = int((inner_size - 1) / 2)
        r_outer = int((outer_size - 1) / 2)
        r_n = int((datan_size - 1) / 2)
        r = r_outer - r_inner
        new_rows = rows - outer_size + 1
        new_cols = cols - outer_size + 1
        neibor_num = outer_size ** 2 - inner_size ** 2
        data3d = torch.from_numpy(data3d).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).cuda()
        weight = torch.ones((1, 1, 1, datan_size, datan_size)).cuda()
        data = torch.nn.functional.conv3d(data3d, weight, stride=1, padding=(0, r_n, r_n)).cuda()
        dual_win_mask = torch.ones(outer_size, outer_size).cuda()
        dual_win_mask[r:(r+inner_size), r:(r+inner_size)] = 0
        dual_win_weight = torch.zeros((neibor_num, 1, 1, outer_size, outer_size)).cuda()
        index = 0
        for i in range(outer_size):
            for j in range(outer_size):
                if dual_win_mask[i, j] == 1:
                    dual_win_weight[index, :, :, i, j] = 1
                    index += 1
        out_data = torch.nn.functional.conv3d(data, dual_win_weight, stride=1, padding=0).cuda()
        test_data = out_data.cpu().squeeze().permute(1, 0, 2, 3).contiguous().view(bands, -1).t()
        test_center = data[:, :, :, r_outer:rows-r_outer, r_outer:cols-r_outer].cpu()\
            .expand(1, neibor_num, bands, new_rows, new_cols).squeeze().permute(1, 0, 2, 3)\
            .contiguous().view(bands, -1).t()
        index = np.random.randint(new_rows * new_cols, size=train_num)
        train_data = out_data.cpu().squeeze().contiguous().view(neibor_num, bands, -1)[:, :, index]
        train_data = train_data.permute(1, 0, 2).contiguous().view(bands, -1).t()
        train_center = data[:, :, :, r_outer:rows-r_outer, r_outer:cols-r_outer].cpu()\
            .expand(1, neibor_num, bands, new_rows, new_cols).squeeze().view(neibor_num, bands, -1)[:, :, index]
        train_center = train_center.permute(1, 0, 2).contiguous().view(bands, -1).t()
        train_data = Data.TensorDataset(train_data.unsqueeze(1), train_center.unsqueeze(1))
        test_data = Data.TensorDataset(test_data.unsqueeze(1), test_center.unsqueeze(1))
        checkpoint = {'train_data': train_data, 'test_data': test_data}
        if not os.path.isdir('log'):
            os.mkdir('log')
        torch.save(checkpoint, log_dir)
        return train_data, test_data


def grid_select(groundtruth, data3d, grid_size=10, num=10):
    rows, cols, bands = data3d.shape
    row_part = int(rows / grid_size)
    col_part = int(cols / grid_size)
    select_samples = []
    select_label = []
    for i in range(row_part):
        for j in range(col_part):
            row_index = np.random.randint(grid_size, size=num)
            col_index = np.random.randint(grid_size, size=num)
            tmp_smaple = data3d[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size, :]
            tmp_label = groundtruth[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            select_samples.append(tmp_smaple[row_index, col_index, :])
            select_label.append(tmp_label[row_index, col_index])

    select_samples = np.vstack(select_samples)
    select_label = np.hstack(select_label)
    return select_samples, select_label




