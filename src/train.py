import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_io as tfio
import datetime
import numpy as np
import os
import collections
from data_loader import get_dataset


#tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()

if os.name == 'nt':
    top_level_dir_path = 'D:\\173_seeding_harvest_joined_USCA.parquet'
else:
    top_level_dir_path = '/home/ubuntu/173_seeding_harvest_joined_USCA.parquet'

lowest_dirs = list()
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs.append(root)

val_files   = [x for x in lowest_dirs if '8348b' in x]
test_files  = [x for x in lowest_dirs if '8348d' in x]

train_files = [x for x in lowest_dirs if ('8348b3fffffffff' in  x)]  # small dataset
train_files = [x for x in lowest_dirs if ('83488' in  x)] # medium dataset
train_files = [x for x in lowest_dirs if ('83489' in  x)] # big dataset

train_files_positive = [x for x in train_files if ('positive_samples' in x)]
train_files_negative = [x for x in train_files if ('negative_samples' in x)]

num_bands=12
shuffle_buffer=200000
batch_size = 10
crop_season_only = True
num_hexes_per_field=100
min_imgs=10
max_imgs=30
predict_on_N_imgs = max_imgs
perc_field_fill=0.8
min_num_hexes_per_field=50

tf.config.run_functions_eagerly(True)

print('Tensorflow version:', tf.__version__)
print('Tensorflow IO version', tfio.__version__)
train_p_ds = get_dataset(train_files_positive,shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, num_hexes_per_field=num_hexes_per_field,
                       predict_on_N_imgs=10, max_imgs=max_imgs,num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field,
                       min_imgs=min_imgs).prefetch(buffer_size=tf.data.AUTOTUNE)

for t in train_p_ds:
    print(t['fop'], tf.shape(t['hex']), tf.shape(t['band_vals']))

quit(0)

train_n_ds = get_dataset(train_files_negative,shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, num_hexes_per_field=num_hexes_per_field,
                       predict_on_N_imgs=None, max_imgs=max_imgs,num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field,
                       min_imgs=min_imgs).prefetch(buffer_size=tf.data.AUTOTUNE)

for t in train_n_ds.take(3):
    winsound.Beep(1000, 1000)
    print(t)

train_ds = train_p_ds.concatenate(train_n_ds)

for t in train_ds.take(3):
    winsound.Beep(1000, 1000)
    print(t)

quit(0)

val_ds = get_dataset(val_files,shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, num_hexes_per_field=num_hexes_per_field,
                       predict_on_N_imgs=predict_on_N_imgs, max_imgs=max_imgs,num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field,
                       min_imgs=min_imgs).prefetch(buffer_size=tf.data.AUTOTUNE)

