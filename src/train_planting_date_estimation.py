import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import datetime
import numpy as np
import os
import collections
from data_loader_planting_date_estimation import get_dataset

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

top_level_dir_path = 'D:\\hexes'

lowest_dirs = list()
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs.append(root)

print(lowest_dirs)
val_files   = [x for x in lowest_dirs if '8348b' in x]
test_files  = [x for x in lowest_dirs if '8348d' in x]
train_files = [x for x in lowest_dirs if ('2021-06' in  x)]

print(train_files)

num_bands=12
shuffle_buffer=200000
batch_size = 10
crop_season_only = True
num_hexes_per_field=None
min_imgs=10
max_imgs=30
predict_on_N_imgs = max_imgs
perc_field_fill=0.8
min_num_hexes_per_field=50

tf.config.run_functions_eagerly(True)

train_ds = get_dataset(train_files,shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, num_hexes_per_field=num_hexes_per_field,
                       predict_on_N_imgs=None, max_imgs=max_imgs,num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field,
                       min_imgs=min_imgs).prefetch(buffer_size=tf.data.AUTOTUNE)

quit(0)

val_ds = get_dataset(val_files,shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, num_hexes_per_field=num_hexes_per_field,
                       predict_on_N_imgs=predict_on_N_imgs, max_imgs=max_imgs,num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field,
                       min_imgs=min_imgs).prefetch(buffer_size=tf.data.AUTOTUNE)
# test_Ds = get_dataset(test_files,shuffle_buffer=2000000,max_imgs=max_imgs,num_bands=num_bands,flatten_output=True).batch(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

inputs = tf.keras.Input(shape=((num_bands+1)*max_imgs,))
dense = tf.keras.layers.Dense(10, activation="swish")(inputs) #selu, mish
outputs = tf.nn.softplus(tf.keras.layers.Dense(1)(dense))
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="out")

optimizer = tf.keras.optimizers.Adam(0.02)

##################################### tensorboard ####################################
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/home/ubuntu/Documents/keras_model_training/tensorboard/regression/yield_index/' + current_time + '/train'
val_log_dir = '/home/ubuntu/Documents/keras_model_training/tensorboard/regression/yield_index/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)
######################################################################################

r2_scores_queue = collections.deque(30*[0.], 30)
best_r2_score = 0.
for epoch in range(500):
    for batch in train_ds:
        # y = tf.reshape(y, shape=[-1, *tf.shape(y)[2:]])
        # x = tf.reshape(x, shape=[-1, *tf.shape(x)[2:]])
        batch = list(zip(*batch))
        def local_regression_fit(idx):
            x_sub, y_sub, z_sub = batch[idx]
            y_not_zero = tf.not_equal(y_sub, -1)
            y_sub = y_sub[y_not_zero]
            y_sub = tf.reshape(y_sub, (-1, 1))
            x_sub = tf.gather(x_sub, tf.where(y_not_zero)[:, 0], axis=0)
            y_pred = model(x_sub)

            lst_sq_fit = tf.linalg.lstsq(tf.concat([tf.ones_like(y_pred), y_pred], axis=1), y_sub)
            y_pred_lst_sq_corrected = tf.matmul(tf.concat([tf.ones_like(y_pred), y_pred], axis=1), lst_sq_fit)
            unexplained_error = tf.reduce_sum(tf.square(y_sub - y_pred_lst_sq_corrected))
            total_error = tf.reduce_sum(tf.square(y_sub - tf.reduce_mean(y_sub)))
            R_squared = 1. - unexplained_error / total_error

            loss = tf.math.sqrt(tf.reduce_mean((y_sub - y_pred_lst_sq_corrected) ** 2))
            return loss, R_squared - z_sub

        with tf.GradientTape() as tape:
            loss, batch_r2 = tf.map_fn(local_regression_fit, tf.range(len(batch)), fn_output_signature=(tf.float32, tf.float32))
            loss = tf.reduce_mean(loss)
            batch_r2 = tf.reduce_mean(batch_r2)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=optimizer.iterations)
            # tf.summary.scalar('r2', r2_score(y.numpy(), y_pred.numpy(), sample_weight=weights.numpy()), step=optimizer.iterations)
            tf.summary.scalar('r2', batch_r2, step=optimizer.iterations)

        if optimizer.iterations % 50 == 0:
            ## validation
            batch = val_ds.as_numpy_iterator().next()
            batch = list(zip(*batch))

            def local_regression_fit(idx):
                x_sub, y_sub, z_sub = batch[idx]
                y_not_zero = tf.not_equal(y_sub, -1)
                y_sub = y_sub[y_not_zero]
                y_sub = tf.reshape(y_sub, (-1, 1))
                x_sub = tf.gather(x_sub, tf.where(y_not_zero)[:, 0], axis=0)
                y_pred = model(x_sub)

                lst_sq_fit = tf.linalg.lstsq(tf.concat([tf.ones_like(y_pred), y_pred], axis=1), y_sub)
                y_pred_lst_sq_corrected = tf.matmul(tf.concat([tf.ones_like(y_pred), y_pred], axis=1), lst_sq_fit)
                unexplained_error = tf.reduce_sum(tf.square(y_sub - y_pred_lst_sq_corrected))
                total_error = tf.reduce_sum(tf.square(y_sub - tf.reduce_mean(y_sub)))
                R_squared = 1. - unexplained_error / total_error

                loss = tf.math.sqrt(tf.reduce_mean((y_sub - y_pred_lst_sq_corrected) ** 2))
                return loss, R_squared - z_sub

            loss, batch_r2 = tf.map_fn(local_regression_fit, tf.range(len(batch)), fn_output_signature=(tf.float32, tf.float32))
            loss = tf.reduce_mean(loss)
            r2_score_latest = tf.reduce_mean(batch_r2)
            r2_scores_queue.appendleft(r2_score_latest)
            if np.mean(r2_scores_queue) > best_r2_score:
                model.save(os.path.split(train_log_dir)[0] + '/my_model')
                best_r2_score = np.mean(r2_scores_queue)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=optimizer.iterations)
                # tf.summary.scalar('r2', r2_score(y.numpy(), y_pred.numpy(), sample_weight=weights.numpy()), step=optimizer.iterations)
                tf.summary.scalar('r2', r2_score_latest, step=optimizer.iterations)