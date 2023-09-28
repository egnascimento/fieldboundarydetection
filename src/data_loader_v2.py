import tensorflow as tf
import tensorflow_io as tfio
from functools import partial
import numpy as np
import csv
import os
import winsound

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

    winsound.Beep(1000, 5000)
    mean_band_vals = tf.constant([1256, 1396, 1601, 1757, 2066, 2822, 3279, 3410, 3491, 3570, 2573, 1868], tf.float64)
    std_band_vals = tf.constant([2199, 2247, 2171, 2233, 2204, 1916, 1891, 1907, 1777, 1816, 1151, 1079], tf.float64)

    def parquet_ds(h3_level_dir):
        pds = tf.data.Dataset.list_files(h3_level_dir + '\\*.parquet')

        pds = pds.interleave(lambda x: tfio.IODataset.from_parquet(x, parquet_dict),
                             num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(1000000000)


        def test_batch(x):
            print(x)
            return x

        for p in pds:
            test_batch(p)

        pds = pds.map(test_batch)


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

        # create unique key to remove duplicates ------------------------------------------------------------------------
        def combine_fop_and_hex(row):
            try:
                row['unique_index'] = row['FIELD_OPERATION_GUID'] + '_' + row['hex']
            except Exception as e:
                print('Error in normalize_bands:', str(e))
            return row
        pds = pds.map(combine_fop_and_hex)

        return pds

    ds = tf.data.Dataset.from_tensor_slices(dir_paths).shuffle(buffer_size=shuffle_buffer,
                                                               reshuffle_each_iteration=True)

    for d in ds.take(1):
        parquet_ds(d)

    ds = ds.interleave(parquet_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)

    def batch_by_fop(pds):
        fields_list = []
        if os.path.exists('fields.csv'):
            with open('fields.csv', newline='') as f:
                reader = csv.reader(f)
                fields_list = list(reader)[0]
                for idx, t in enumerate(fields_list):
                    fields_list[idx] = t.encode()

        fields_tensor = tf.constant(fields_list)
        value_tensor = tf.range(0, len(fields_list))

        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(fields_tensor, value_tensor),
            default_value=-1)

        def create_hash(row):
            row['field'] = table.lookup(row['FIELD_OPERATION_GUID'])
            return row

        pds = pds.map(create_hash)

        window_size = 50000
        pds = pds.group_by_window(
            key_func=lambda row: tf.cast(row['field'], tf.int64),
            reduce_func=lambda key, rows: rows.batch(window_size),
            window_size=window_size
        )

      #  def clean_unkown_fields(ds):
      #      return ds.filter(lambda row: row['field'] != -1)
      #  pds = pds.apply(clean_unkown_fields)

        return pds

    for d in ds.take(3):
        print(d)

    ds = ds.apply(batch_by_fop)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    def stack_hexes(row):
        uniques = tf.unique_with_counts(row['hex'])  # idx like 0 1 2 0 0 2 2 2 3 0

        def num_of_samples_per_hex(idx):
            ret = tf.gather(row['band_vals'], tf.reshape(tf.where(uniques.idx == idx)[0:5], [-1]))
            return ret
        band_vals = tf.map_fn(num_of_samples_per_hex, tf.range(len(uniques.y)), dtype=tf.float64)

        num_hexes_per_field = 100
        band_vals = band_vals[0:num_hexes_per_field]

        return row['FIELD_OPERATION_GUID'][0:tf.shape(band_vals)[0]], \
               row['hex'][0:tf.shape(band_vals)[0]], \
               band_vals

    for d in ds.take(3):
        stack_hexes(d)

    ds = ds.map(stack_hexes)

    for d in ds.take(3):
        print(d)

    return ds