import sys
import os
import os.path as osp
import shutil
import h5py
import numpy as np
from tqdm import tqdm


"""
path_patchi: level = L,   size = 256
path_patcho: level = L-1, size = 256
path_patchi * patch_scale -> path_patcho
"""
def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def get_scaled_matrix(width, height, scale=4):
    mat = np.zeros((scale, scale, 2))
    for j in range(scale):
        for i in range(scale):
            mat[j][i] = np.array([i * width, j * height])
    mat = np.reshape(mat, (-1, 2))
    return mat

def get_scaled_attrs(origin_attrs, scale=4):
    attrs = {
        'downsample': origin_attrs['downsample'] / scale,
        'downsampled_level_dim': origin_attrs['downsampled_level_dim'] * scale,
        'level_dim': origin_attrs['level_dim'] * scale,
        'name': origin_attrs['name'],
        'patch_level': origin_attrs['patch_level'],
        'patch_size': int(origin_attrs['patch_size']/scale),
    }
    return attrs

def coords_x5_to_x20(path_patchi, path_patcho, patch_scale=4):
    scaled_coords = np.zeros((1,2), dtype=np.int32)
    scaled_attrs  = None

    with h5py.File(path_patchi, 'r') as hf:
        data_coords = hf['coords']
        scaled_attrs = get_scaled_attrs(data_coords.attrs, patch_scale)

        psize = data_coords.attrs['patch_size']
        scaled_mat = get_scaled_matrix(psize, psize, patch_scale)
        coords = data_coords[:]
        for coord in coords:
            cur_coords = scaled_mat + coord
            scaled_coords = np.concatenate((scaled_coords, cur_coords), axis=0)

    scaled_coords = scaled_coords[1:] # ignore the first row
    scaled_attrs['save_path'] = osp.dirname(path_patcho)

    # compatible with OpenSlide - read_region(slide, buf, x, y, level, w, h)
    # refer to issues https://github.com/liupei101/DSCA/issues/8 
    #             and https://github.com/liupei101/DSCA/issues/3
    scaled_coords = scaled_coords.astype(np.int64) 
    save_hdf5(path_patcho, {'coords': scaled_coords}, {'coords': scaled_attrs}, mode='w')

def process_coords(dir_read, dir_save, low_mag, high_mag):
    dir_read = os.path.join(dir_read, 'patches')
    dir_save = os.path.join(dir_save, 'patches')
    if not osp.exists(dir_save):
        os.makedirs(dir_save)

    files = os.listdir(dir_read)
    for fname in tqdm(files):
        if fname[-2:] != 'h5':
            print('invalid file {}, skipped'.format(fname))
            continue

        path_read = osp.join(dir_read, fname)
        path_save = osp.join(dir_save, fname)
        scale = int(int(high_mag)/int(low_mag))
        coords_x5_to_x20(path_read, path_save, scale)

# python3 big_to_small_patching.py READ_PATCH_DIR SAVE_PATCH_DIR 5 10
if __name__ == '__main__':
    READ_PATCH_DIR = sys.argv[1] # full read path to the patch coordinates at level = 2.
    SAVE_PATCH_DIR = sys.argv[2] # full save path to the patch coordinates at level = 1.
    LOW_MAG = sys.argv[3]
    HIGH_MAG = sys.argv[4]
    process_coords(READ_PATCH_DIR, SAVE_PATCH_DIR, LOW_MAG, HIGH_MAG)
    # at the same time, copy the processing record file to SAVE_PATCH_DIR
    shutil.copy(osp.join(READ_PATCH_DIR, 'process_list_autogen.csv'), SAVE_PATCH_DIR)