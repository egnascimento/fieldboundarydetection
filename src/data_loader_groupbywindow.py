import tensorflow as tf
import tensorflow_io as tfio
from functools import partial
import numpy as np
import csv
import os

parquet_dict = {
                    # 'start_date': tf.string,
                    # 'end_date': tf.string,
                    'FIELD_OPERATION_GUID': tf.string,
                    'scene_id': tf.string,
                    'hex': tf.string,
                    # 'SCL_val': tf.int64,
                    # 's2_tile': tf.object,
                    'B01': tf.float64,
                    'B02': tf.float64,
                    'B03': tf.float64,
                    'B04': tf.float64,
                    'B05': tf.float64,
                    'B06': tf.float64,
                    'B07': tf.float64,
                    'B08': tf.float64,
                    'B8A': tf.float64,
                    'B09': tf.float64,
                    'B11': tf.float64,
                    'B12': tf.float64,
                }

band_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


band_values = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
number_of_bands = len(band_values)


def get_dataset(dir_paths, batch_size, num_bands=12, shuffle_buffer=200000, crop_season_only=False,
                num_hexes_per_field=None, min_imgs=10, max_imgs=30, predict_on_N_imgs=None, perc_field_fill=0.8,
                min_num_hexes_per_field=50):
    '''
    :param dir_path: path of the field op guid partition
    :param max_imgs: Must match the model input size
    :param num_bands: sentinel 2 = 12 bands
    :param flatten_output: True for dense models
    :param get_full_timseries_output: only for investigations
    :param predict_on_N_imgs: number of images to use for prediction.  Like max_imgs, it must match the model training
    :return:
    '''

    mean_band_vals = tf.constant([1256, 1396, 1601, 1757, 2066, 2822, 3279, 3410, 3491, 3570, 2573, 1868], tf.float64)
    std_band_vals = tf.constant([2199, 2247, 2171, 2233, 2204, 1916, 1891, 1907, 1777, 1816, 1151, 1079], tf.float64)

    ds = tf.data.Dataset.from_tensor_slices(dir_paths).shuffle(buffer_size=shuffle_buffer,
                                                               reshuffle_each_iteration=True)

    print('Folders to extract data from: ', len(dir_paths))
    print('Files to extract data from: ', tf.shape(ds))

    def parquet_ds(h3_level_dir):
        pds = tf.data.Dataset.list_files(h3_level_dir + '/*.parquet')

        pds = pds.interleave(lambda x: tfio.IODataset.from_parquet(x, parquet_dict),
                             num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        # Scale bands --------------------------------------------------------------------------------------------------
        def normalize_bands(row):
            try:
                for idx, values in enumerate(band_values):
                    row[values] = (row[values] - mean_band_vals[idx]) / std_band_vals[idx]
            except Exception as e:
                print('Error in normalize_bands:', str(e))
            return row
        pds = pds.map(normalize_bands)

        # create array with band values --------------------------------------------------------------------------------
        def consolidate_bands(row):
            try:
                row['band_vals'] = [row['B01'], row['B02'], row['B03'], row['B04'], row['B05'], row['B06'], row['B07'],
                                    row['B08'], row['B8A'], row['B09'], row['B11'], row['B12']]
            except Exception as e:
                print('Error in normalize_bands:', str(e))

            return {'FIELD_OPERATION_GUID': row['FIELD_OPERATION_GUID'],
                    'hex': row['hex'],
                    'band_vals': row['band_vals']}
        pds = pds.map(consolidate_bands)

        def stack_fields(row):
            unique_fields = tf.unique_with_counts(row['FIELD_OPERATION_GUID'])

            def stack_hexes(field_idx):
                # Collect the hexes from this particular field
                hexes_from_field = tf.gather(row['hex'], tf.reshape(tf.where(unique_fields.idx == field_idx), [-1]))
                fields = tf.gather(row['FIELD_OPERATION_GUID'],
                                   tf.reshape(tf.where(unique_fields.idx == field_idx), [-1]))
                bands_from_field = tf.gather(row['band_vals'],
                                             tf.reshape(tf.where(unique_fields.idx == field_idx), [-1]))

                unique_hexes = tf.unique_with_counts(hexes_from_field)

                def num_of_samples_per_hex(idx):
                    # Collect the band_vals from this particular hex
                    pad_to = 10
                    bands_from_hexes = tf.gather(bands_from_field, tf.reshape(tf.where(unique_hexes.idx == idx), [-1]))
                    pad_to_add = pad_to - len(bands_from_hexes)

                    paddings = tf.concat(([[0, pad_to_add]], [[0, 0]]), axis=0)
                    bands_from_hexes = tf.pad(bands_from_hexes, paddings=paddings, mode='CONSTANT', constant_values=0)
                    hexes = tf.gather(hexes_from_field, tf.reshape(tf.where(unique_hexes.idx == idx), [-1]))
                    hexes = tf.concat([hexes, tf.zeros((pad_to - tf.shape(hexes)), dtype=tf.string)], axis=0)
                    return hexes, bands_from_hexes

                band_vals = tf.map_fn(num_of_samples_per_hex, tf.range(len(unique_hexes.y)),
                                      fn_output_signature=(tf.string, tf.float64))

                hexes = tf.reshape(band_vals[0], [-1])
                band_vals = tf.reshape(band_vals[1], [len(hexes), 12])
                fields = tf.fill(dims=[len(hexes)], value=fields[0])

                # padding
                pad_to = 10000
                pad_to_add = pad_to - len(hexes)
                hexes = tf.concat([hexes, tf.zeros((pad_to_add), dtype=tf.string)], axis=0)
                paddings = tf.concat(([[0, pad_to_add]], [[0, 0]]), axis=0)
                band_vals = tf.pad(band_vals, paddings, mode='CONSTANT',
                                   constant_values=0)
                fields = tf.fill(dims=[len(hexes)], value=fields[0])

                return fields, \
                       hexes, \
                       band_vals

            band_vals = tf.map_fn(stack_hexes, tf.range(len(unique_fields.y)),
                                  fn_output_signature=(tf.string, tf.string, tf.float64))

            fields = tf.reshape(band_vals[0], [-1])
            hexes = tf.reshape(band_vals[1], [-1])
            bands = tf.reshape(band_vals[2], (150000, 12))

            return {'fop': fields, 'hex': hexes, 'band_vals': bands}

        pds = pds.map(stack_fields)

        return pds

    ds = ds.interleave(parquet_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)

    for d in ds:
        print(d)

    return ds