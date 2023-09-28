import tensorflow as tf
import tensorflow_io as tfio
from functools import partial
from map import save_to_map

parquet_dict = {
                #'seeding_date': tf.int32,
                #'harvest_date': tf.int32,
                'L14_RSum_harvest': tf.float64,
                'hex_index_L12': tf.string,
                # 'n_epochs_seeding': tf.int32,
                # 'n_epochs_harvest': tf.int32,
                #'mean_YieldVolumePerArea_bu_per_ha': tf.float64,
                # 'mean_HarvestMoisture_prcnt': tf.float64,
                'num_images': tf.int64,
                'bands': tf.string,
                'scene_ids': tf.string,
                # 'tiles': tf.string,
                # 'img_dates_int': tf.string
                'img_dates': tf.string
                }

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

    mean_band_vals = tf.constant([1256, 1396, 1601, 1757, 2066, 2822, 3279, 3410, 3491, 3570, 2573, 1868], tf.float32)
    std_band_vals = tf.constant([2199, 2247, 2171, 2233, 2204, 1916, 1891, 1907, 1777, 1816, 1151, 1079], tf.float32)

    def parse(row, fop):

        split_scenes = tf.strings.split(tf.reshape(row['scene_ids'], [-1]), ',')[0]

        ################### remove duplicate scenes by randomly choosing one of the duplicates #########################
        _, _, scene_counts = tf.unique_with_counts(split_scenes)

        unique_idxs = tf.concat([[0], tf.cumsum(scene_counts)[:-1]], axis=0)
        
        def rand_sample(maxval):
            return tf.random.uniform(shape=(), minval=0, maxval=maxval, dtype=tf.int32)

        unique_idxs = unique_idxs + tf.map_fn(fn=rand_sample, elems=scene_counts, fn_output_signature=tf.int32)
        ###################### constrain to crop season only ###########################################################
        def convert_bytes_dates(idx):
            img_date_bytes = tf.strings.substr(row['img_dates'], idx * 2, 2)  # %band_lengths guarantees idx is in range
            img_date_ints = tf.io.decode_raw(img_date_bytes, out_type=tf.uint16, little_endian=False, fixed_length=2)
            return tf.cast(img_date_ints, tf.int32)

        img_dates = tf.map_fn(fn=convert_bytes_dates, elems=unique_idxs, fn_output_signature=tf.int32)




        def convert_bytes(idx):
            random_img_bytes = tf.strings.substr(row['bands'], idx * num_bands * 2,
                                                 num_bands * 2)  # %band_lengths guarantees idx is in range
            random_img_ints = tf.io.decode_raw(random_img_bytes, out_type=tf.uint16, little_endian=False,
                                               fixed_length=24)
            return (tf.cast(random_img_ints, tf.float32) - mean_band_vals) / std_band_vals

       # for i,idx in enumerate(unique_idxs):
       #     print(convert_bytes(idx))
       #     if i == 5:
       #         break;

        band_vals = tf.map_fn(fn=convert_bytes, elems=unique_idxs, fn_output_signature=tf.float32)




        squeezed_dates = tf.squeeze(img_dates)

        return band_vals, \
               squeezed_dates, \
               row['hex_index_L12'], \
               row['L14_RSum_harvest'], \
               fop

#########################################################################################3
    def image_selector_proc_fn(input_val):
        print(tf.executing_eagerly())

        band_vals, img_dates, hexes, L14_RSum_harvest, fop = input_val
        print(tf.shape(band_vals)[0])
        tf.Assert(tf.shape(band_vals)[0] > min_num_hexes_per_field, [1])

        unique_cnts = tf.unique_with_counts(tf.reshape(img_dates, [-1]))
        unique_dates = tf.sort(tf.boolean_mask(unique_cnts.y, tf.logical_and(unique_cnts.y > 0,
                                                                             unique_cnts.count / tf.shape(img_dates)[
                                                                                 0] > perc_field_fill)))

        '''
        if encoding time series:
        -- need entire field
        -- use consistent dates (same across all hexes for that field)
        '''
        # Regression Quality Estimation
        if num_hexes_per_field:
            num_hexes_per_field_ = tf.minimum(num_hexes_per_field, min_num_hexes_per_field)
            rsq_max = 0

            def regression_quality_sampling(idx):
                if predict_on_N_imgs:
                    r_samp_len = tf.minimum(tf.minimum(predict_on_N_imgs, max_imgs), len(unique_dates))
                else:
                    r_samp_len = tf.random.uniform(shape=(), minval=tf.minimum(min_imgs, len(unique_dates)),
                                                   maxval=tf.minimum(max_imgs, len(unique_dates)) + 1, dtype=tf.int32)

                r_samp_dates = tf.sort(tf.random.shuffle(unique_dates)[:r_samp_len])
                r_samp_idxs = tf.where(img_dates[idx] == tf.reshape(r_samp_dates, [-1, 1]))[:, -1]
                r_samp_dates = tf.sort(
                    tf.sets.intersection(r_samp_dates[None, :], tf.gather(img_dates[idx], r_samp_idxs)[None, :]).values)
                r_samp_len = len(r_samp_idxs)
                gt_rsq = 0

                img_dates_out = tf.gather_nd(img_dates[idx], tf.reshape(r_samp_idxs, [-1, 1]))
                img_diffs = tf.cast(img_dates_out[-1] - img_dates_out, tf.float32) / 365
                band_vals_out = tf.gather_nd(band_vals[idx], tf.reshape(r_samp_idxs, [-1, 1]))
                band_vals_out = tf.concat([band_vals_out, tf.reshape(img_diffs, (-1, 1))], axis=1)
                band_vals_out = tf.pad(band_vals_out, [[(max_imgs - r_samp_len), 0], [0, 0]])
                band_vals_out = tf.reshape(band_vals_out, [-1])

                return band_vals_out, gt_rsq / rsq_max

            random_hex_idxs = tf.random.uniform(shape=[num_hexes_per_field_], minval=0, maxval=len(img_dates),
                                                dtype=tf.int32)
            band_vals_out_, gt_rsq_out_ = tf.map_fn(regression_quality_sampling, random_hex_idxs,
                                                    fn_output_signature=(tf.float32, tf.float32))

            return band_vals_out_, gt_rsq_out_

        # encode time series of imagery
        else:
            tf.Assert(crop_season_only, [1])
            ## one random sample of dates for all fields
            if predict_on_N_imgs:
                r_samp_len = tf.minimum(tf.minimum(predict_on_N_imgs, max_imgs), len(unique_dates))
            else:
                r_samp_len = tf.random.uniform(shape=(), minval=tf.minimum(min_imgs, len(unique_dates)),
                                               maxval=tf.minimum(max_imgs, len(unique_dates)) + 1, dtype=tf.int32)
            r_samp_dates = tf.sort(tf.random.shuffle(unique_dates)[:r_samp_len])

            ## boolean mask keep dims by substituting zeros
            # def get_imgs(date_val):
            #     # tf.where(array, tensor, tf.zeros_like(tensor))
            #     return tf.where(tf.tile(tf.expand_dims(img_dates == date_val, axis=2), [1,1,tf.shape(band_vals)[-1]]), band_vals, tf.zeros_like(band_vals))
            #
            # band_vals_out = tf.map_fn(get_imgs, r_samp_dates, fn_output_signature=tf.float32)
            # band_vals_out = tf.reduce_sum(band_vals_out, axis=0)

            def embedding_time_series(idx):
                r_samp_idxs = tf.where(img_dates[idx] == tf.reshape(r_samp_dates, [-1, 1]))[:, -1]
                r_samp_len = len(r_samp_idxs)

                img_dates_out = tf.gather_nd(img_dates[idx], tf.reshape(r_samp_idxs, [-1, 1]))
                img_diffs = tf.cast(img_dates_out[-1] - img_dates_out, tf.float32) / 365
                band_vals_out = tf.gather_nd(band_vals[idx], tf.reshape(r_samp_idxs, [-1, 1]))
                band_vals_out = tf.concat([band_vals_out, tf.reshape(img_diffs, (-1, 1))], axis=1)
                band_vals_out = tf.pad(band_vals_out, [[(max_imgs - r_samp_len), 0], [0, 0]])
                band_vals_out = tf.reshape(band_vals_out, [-1])

                return band_vals_out

            for b in tf.range(len(band_vals)):
                print(b)
                embedding_time_series(b)

            band_vals_out_ = tf.map_fn(embedding_time_series, tf.range(len(band_vals)), fn_output_signature=tf.float32)

            ## best Rsq of dates sampled (not global across all dates available for a give field.  If that is desired, use Rsq_max under regression quality switch above
            rsq_max = 0
            print(band_vals_out_, y, rsq_max)
            return band_vals_out_, y, rsq_max

    def parquet_ds(field_op_dir):
        ds = tf.data.Dataset.list_files(field_op_dir + '\\*.parquet')

        ds = ds.interleave(lambda x: tfio.IODataset.from_parquet(x, parquet_dict), num_parallel_calls=tf.data.AUTOTUNE,
                           deterministic=False)

        # Use this to debug "parse"
        p=partial(parse, fop=tf.strings.split(field_op_dir, 'FIELD_OPERATION_GUID=')[-1])
        for i,d in enumerate(ds):
            print(d)
            p(d)
            if i == 3:
                break


        ds = ds.map(partial(parse, fop=tf.strings.split(field_op_dir, 'FIELD_OPERATION_GUID=')[-1]),num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Use this to debug view parse output
        for d in ds:
            print(d)
            break

        ret = ds.apply(tf.data.experimental.ignore_errors()).padded_batch(10000,
                                                                           (
                                                                               [None, 12], #band_vals
                                                                               [None], # squeezed_dates
                                                                               [], # row['hex_index_L12']
                                                                               [], #L14_RSum_harvest
                                                                               [] # fop
                                                                           ) )

        '''
        d = {}
        l = []
        fields = 10
        for r in ret:
            for h in r[2]:
                l.append(h.numpy().decode('utf-8'))
            print(r[3][0].numpy().decode('utf-8'))
            d[r[3][0].numpy().decode('utf-8')] = l

            fields = fields - 1
            if fields == 0:
                break

        save_to_map(d)
        print(d)
        '''

        return ret


    ds = tf.data.Dataset.from_tensor_slices(dir_paths).shuffle(buffer_size=shuffle_buffer,
                                                                reshuffle_each_iteration=True)
    # Use this to debug parque_ds
    for i,d in enumerate(ds):
        print(d)
        parquet_ds(d)
        if i == 3:
            break

    ds = ds.interleave(parquet_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)

    #fields = 90
    # Use this to debug parquet_ds output
    #for d in ds:
    #    print(d)
    #    dic = {}
    #    l = []
    #    dic_pcts = {}
    #    pcts = []


    #    for h in d[2]:
    #        l.append(h.numpy().decode('utf-8'))
    #    for p in d[3]:
    #        pcts.append(p.numpy())
    #    print(d[4][0].numpy().decode('utf-8'))
    #    dic[d[4][0].numpy().decode('utf-8')] = l
    #    dic_pcts[d[4][0].numpy().decode('utf-8')] = pcts

    #    save_to_map(dic, pcts=dic_pcts)

    #    fields = fields - 1
    #    if fields == 0:
    #        break

    #    quit(0)


    print(tf.executing_eagerly())
    for d in ds:
        tf.print(d)
        image_selector_proc_fn(d)

    quit(0)
    ds = ds.map(image_selector_proc_fn).apply(tf.data.experimental.ignore_errors())
    if num_hexes_per_field:
        ds = ds.batch(batch_size)
    else:
        ds = ds.padded_batch(batch_size, padded_shapes=([None, (num_bands + 1) * max_imgs], [None], []),
                             padding_values=-1.)
    # ds = ds.padded_batch(batch_size, padded_shapes=([None, None, 12], [None], [None], [None, None], [None, None]))
    # ds = ds.padded_batch(batch_size, padded_shapes=([None, None, 12], [None], [None, None], [None, None], [None], [None]))
    # ds = ds.map(image_selector_proc_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).apply(tf.data.experimental.ignore_errors())

    return ds