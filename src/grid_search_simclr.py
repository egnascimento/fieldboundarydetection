import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from pathlib import Path

#--------------------------------------------------------------------------------------
parameters = {
                'temporal_samples': [15],
                'band_features' : [['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']],
                'width': [30],
                'temperature': [0.2],
                'steps_per_epoch': [5, 10, 20, 50, 100],
                'shuffle_buffer': [5],
                'num_epochs' : [40],
                'queue_size': [10000],
                'kernel_size': [5],
                'strides': [1],
                'contrastive_augmentation' : [{"jitter": 0.30}],
                'classification_augmentation' : [{"jitter": 0.15}],
}

count = 0
for i in list(ParameterGrid(parameters)):
    count = count + 1
    temporal_samples = i['temporal_samples']
    band_features = i['band_features']
    width = i['width'] # output vector size
    temperature = i['temperature'] # empirical temperature value
    steps_per_epoch = i['steps_per_epoch']
    AUTOTUNE = tf.data.AUTOTUNE
    shuffle_buffer = i['shuffle_buffer']
    num_epochs = i['num_epochs']
    queue_size = i['queue_size']
    kernel_size = i['kernel_size']
    strides = i['strides']
    # Stronger augmentations for contrastive, weaker ones for supervised training
    contrastive_augmentation = i['contrastive_augmentation']
    classification_augmentation = i['classification_augmentation']


    #--------------------------------------------------------------------------------------
    positive_samples_folder = Path(
        'D:\\173_seeding_harvest_joined_USCA.parquet\\positive_samples\\hex_index_L3=8348b3fffffffff')
    negative_samples_folder = Path(
        'D:\\173_seeding_harvest_joined_USCA.parquet\\negative_samples\\hex_index_L3=8348b3fffffffff')
    #--------------------------------------------------------------------------------------

    df_positive = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in positive_samples_folder.rglob('*.parquet')
             )

    df_negative = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in negative_samples_folder.rglob('*.parquet'))

    print('The shape of loaded positive dataframe is:', df_positive.shape)
    print('The shape of loaded negative dataframe is:', df_negative.shape)

    df_positive = df_positive.drop_duplicates()
    df_negative = df_negative.drop_duplicates()

    df_positive['timestamp'] = df_positive.scene_id.str[11:26]
    df_negative['timestamp'] = df_negative.scene_id.str[11:26]

    print('The shape of loaded positive dataframe is:', df_positive.shape)
    print('The shape of loaded negative dataframe is:', df_negative.shape)

    #-----------------------------------------------------------------------------------------------------------

    print('In this dataset there are ', df_positive.hex.unique().size, ' different positive hexes')
    print('In this dataset there are ', df_negative.hex.unique().size, ' different negative hexes')

    positive_l12_hexes = df_positive.hex.unique()
    negative_l12_hexes = df_negative.hex.unique()

    ambiguous_l12_hexes = set(positive_l12_hexes).intersection(negative_l12_hexes)
    print('There are ', len(ambiguous_l12_hexes), ' hexes labeled both as positive and negative')

    df_positive = df_positive[~df_positive['hex'].isin(ambiguous_l12_hexes)]
    df_negative = df_negative[~df_negative['hex'].isin(ambiguous_l12_hexes)]

    print('In this dataset there are ', df_positive.hex.unique().size, ' different positive hexes')
    print('In this dataset there are ', df_negative.hex.unique().size, ' different negative hexes')

    #-----------------------------------------------------------------------------------------------------------

    number_of_bands = len(band_features)

    # Keep only [temporal samples] samples
    df_positive = df_positive.sort_values(by=['hex'])
    df_positive = df_positive.groupby('hex').head(temporal_samples)

    df_negative = df_negative.sort_values(by=['hex'])
    df_negative = df_negative.groupby('hex').head(temporal_samples)

    # Associate labels picked manually
    df_negative = df_negative.assign(label=0)
    df_positive = df_positive.assign(label=1)
    df = pd.concat([df_positive, df_negative], axis=0)

    grouped_df = df.groupby(['hex'])

    new_df = pd.DataFrame(columns=df.columns)

    # Loop through each group in the grouped DataFrame
    count = 0;
    hexes_count = df['hex'].nunique()
    for group_name, group_data in grouped_df:
        count = count + 1
        pct_complete = count / hexes_count * 100
        print('Packing {0:.2f}'.format(pct_complete) + '% (' + str(count) + '/' + str(hexes_count) + ')', end='\r')

        # Check if the group has more than 5 rows
        if len(group_data) > temporal_samples:
            # If yes, randomly sample 5 rows and add them to the new DataFrame
            new_rows = group_data.sample(n=5, replace=False)
            new_df = pd.concat([new_df, new_rows])
        else:
            # If no, repeat the existing rows until there are 5 rows and add them to the new DataFrame
            num_repeats = temporal_samples // len(group_data) + 1
            repeated_rows = pd.concat([group_data] * num_repeats, ignore_index=True)
            new_rows = repeated_rows.iloc[:temporal_samples]
            new_df = pd.concat([new_df, new_rows])

    df = new_df.copy()

    df[band_features] = StandardScaler().fit_transform(df[band_features])

    # Organize the 2D samples in numpy arrays
    sample = np.zeros((temporal_samples, number_of_bands), dtype=np.float64)
    X_array = np.empty((0, temporal_samples, number_of_bands), dtype=np.float64)

    labels = []
    hexes = []
    timestamp_tracking = []
    timestamp_sample = []

    sub_index = 0
    count = 0;
    for index, row in df.iterrows():
        pct_complete = count / df.shape[0] * 100
        print('Sampling {0:.2f}'.format(pct_complete) + '%', end='\r')

        # fill the band values in a temporal row
        for idx, b in enumerate(band_features):
            sample[sub_index][idx] = row[b]

        #sample[sub_index][12] = row.month

        timestamp_sample.append(row.timestamp)

        # increment row number
        sub_index = sub_index + 1

        # if reached last row of temporal samples, increment to next sample
        if sub_index == temporal_samples:
            shuffler = np.random.permutation(sample.shape[0])
            sample = sample[shuffler]
            timestamp_sample = list(np.array(timestamp_sample)[shuffler])

            if len(timestamp_sample) != temporal_samples:
                print('Invalid sample!')

            X_array = np.append(X_array, [sample], axis=0)

            labels.append(row.label)
            hexes.append(row.hex)
            timestamp_tracking.append(timestamp_sample.copy())
            timestamp_sample.clear()
            sub_index = 0

    print(X_array.shape)
    print(len(labels))
    print(len(hexes))

    # -----------------------------------------------------------------------------------------------------------

    print('Total of samples:', X_array.shape)

    labels_and_hexes = np.vstack((hexes, labels)).T

    #-----------------------------------------------------------------------------------------------------------

    # 80% for training, 10% test and 10% validation
    X_train, X_test, yl_train, yl_test = train_test_split(X_array, labels_and_hexes, test_size=0.2, random_state=42)
    X_val, X_test, yl_val, yl_test = train_test_split(X_test, yl_test, test_size=0.5, random_state=42)

    labeled_train_samples = X_train.shape[0]
    hexes_train, y_train = np.hsplit(yl_train, 2)
    hexes_test, y_test = np.hsplit(yl_test, 2)
    hexes_val, y_val = np.hsplit(yl_val, 2)

    y_train = [int(y) for y in y_train]
    y_test = [int(y) for y in y_test]
    y_val = [int(y) for y in y_val]

    print('Total of training samples:',X_train.shape, len(y_train))
    print('Total of test samples:', X_test.shape, len(y_test))
    print('Total of validation samples:', X_val.shape, len(y_val))


    def prepare_dataset():
        labeled_batch_size = labeled_train_samples // steps_per_epoch
        batch_size = labeled_batch_size

        train_dataset = tf.data.Dataset \
            .from_tensor_slices((X_train, y_train)) \
            .shuffle(buffer_size=10 * labeled_batch_size) \
            .batch(labeled_batch_size, drop_remainder=True)

        test_dataset = tf.data.Dataset \
            .from_tensor_slices((X_test, y_test)) \
            .batch(batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

        validation_dataset = tf.data.Dataset \
            .from_tensor_slices((X_val, y_val)) \
            .batch(5) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

        return batch_size, train_dataset, test_dataset, validation_dataset


    batch_size, train_dataset, test_dataset, validation_dataset = prepare_dataset()

    #-----------------------------------------------------------------------------------------------------------

    # Distorts the color distibutions of images
    class RandomAugmentSatelliteBands(layers.Layer):
        def __init__(self, jitter=0, **kwargs):
            super().__init__(**kwargs)

            self.jitter = jitter

        def get_config(self):
            config = super().get_config()
            config.update({"jitter": self.jitter})
            return config

        def call(self, images, training=True):
            if training:
                batch_size = tf.shape(images)[0]

                jitter_matrices = tf.random.uniform(
                    (batch_size, temporal_samples, 1), minval=1 - self.jitter, maxval=1 + self.jitter
                )

                jm = [jitter_matrices] * number_of_bands
                jitter_matrices = tf.concat(jm, axis=2)

                images = tf.round(images * jitter_matrices)

            return images


    # Define the encoder architecture
    def get_encoder():
        return keras.Sequential(
            [
                keras.Input(shape=(temporal_samples, number_of_bands)),
                layers.Conv1D(width, kernel_size=kernel_size, strides=strides, activation="relu"),
                layers.Conv1D(width, kernel_size=kernel_size, strides=strides, activation="relu"),
                #layers.Conv1D(width, kernel_size=kernel_size, strides=strides, activation="relu"),
                #layers.Conv1D(width, kernel_size=kernel_size, strides=strides, activation="relu"),
                layers.Flatten(),
                layers.Dense(width, activation="relu"),
            ],
            name="encoder",
        )


    # Image augmentation module
    def get_augmenter(jitter):
        return keras.Sequential(
            [
                keras.Input(shape=(temporal_samples, number_of_bands)),
                RandomAugmentSatelliteBands(jitter),
            ]
        )

    # Baseline supervised training with random initialization
    baseline_model = keras.Sequential(
        [
            layers.Input(shape=(temporal_samples, number_of_bands)),
            get_augmenter(**classification_augmentation),
            get_encoder(),
            layers.Dense(1, activation='sigmoid'),
        ],
        name="baseline_model",
    )
    baseline_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    baseline_history = baseline_model.fit(
        train_dataset, epochs=num_epochs, validation_data=test_dataset
    )

    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(baseline_history.history["val_acc"]) * 100
        )
    )


    # Define the contrastive model with model-subclassing

    class ContrastiveModel(keras.Model):
        def __init__(self):
            super().__init__()

            self.temperature = temperature
            self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
            self.classification_augmenter = get_augmenter(**classification_augmentation)
            self.encoder = get_encoder()

            # Non-linear MLP as projection head
            self.projection_head = keras.Sequential(
                [
                    keras.Input(shape=(width,)),
                    layers.Dense(width, activation="relu"),
                    layers.Dense(width),
                ],
                name="projection_head",
            )
            # Single dense layer for linear probing
            self.linear_probe = keras.Sequential(
                [layers.Input(shape=(width,)), layers.Dense(1, activation='sigmoid')], name="linear_probe"
            )

            self.encoder.summary()
            self.projection_head.summary()
            self.linear_probe.summary()

        def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
            super().compile(**kwargs)

            self.contrastive_optimizer = contrastive_optimizer
            self.probe_optimizer = probe_optimizer

            # self.contrastive_loss will be defined as a method
            self.probe_loss = keras.losses.BinaryCrossentropy(from_logits=False)

            self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
            self.contrastive_accuracy = keras.metrics.BinaryAccuracy(
                name="c_acc"
            )
            self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
            self.probe_accuracy = keras.metrics.BinaryAccuracy(name="p_acc")

        @property
        def metrics(self):
            return [
                self.contrastive_loss_tracker,
                self.contrastive_accuracy,
                self.probe_loss_tracker,
                self.probe_accuracy,
            ]

        def contrastive_loss(self, projections_1, projections_2):
            # InfoNCE loss (information noise-contrastive estimation)
            # NT-Xent loss (normalized temperature-scaled cross entropy)

            # Cosine similarity: the dot product of the l2-normalized feature vectors
            projections_1 = tf.math.l2_normalize(projections_1, axis=1)
            projections_2 = tf.math.l2_normalize(projections_2, axis=1)

            similarities = (
                    tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
            )

            # The similarity between the representations of two augmented views of the
            # same hex should be higher than their similarity with other views
            batch_size = tf.shape(projections_1)[0]

            contrastive_labels = tf.range(batch_size)
            self.contrastive_accuracy.update_state(contrastive_labels, similarities)
            self.contrastive_accuracy.update_state(contrastive_labels, tf.transpose(similarities))

            # The temperature-scaled similarities are used as logits for cross-entropy
            # a symmetrized version of the loss is used here
            loss_1_2 = keras.losses.binary_crossentropy(
                contrastive_labels, similarities, from_logits=False
            )

            loss_2_1 = keras.losses.binary_crossentropy(
                contrastive_labels, tf.transpose(similarities), from_logits=False
            )

            return (loss_1_2 + loss_2_1) / 2

        def train_step(self, data):
            (labeled_images, labels) = data

            # Both labeled and unlabeled images are used, without labels
            images = labeled_images

            # Each image is augmented twice, differently
            augmented_images_1 = self.contrastive_augmenter(images, training=True)
            augmented_images_2 = self.contrastive_augmenter(images, training=True)

            with tf.GradientTape() as tape:
                features_1 = self.encoder(augmented_images_1, training=True)
                features_2 = self.encoder(augmented_images_2, training=True)
                # The representations are passed through a projection mlp
                projections_1 = self.projection_head(features_1, training=True)
                projections_2 = self.projection_head(features_2, training=True)
                contrastive_loss = self.contrastive_loss(projections_1, projections_2)
            gradients = tape.gradient(
                contrastive_loss,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
            self.contrastive_optimizer.apply_gradients(
                zip(
                    gradients,
                    self.encoder.trainable_weights + self.projection_head.trainable_weights,
                )
            )
            self.contrastive_loss_tracker.update_state(contrastive_loss)

            # Labels are only used in evalutation for an on-the-fly logistic regression
            preprocessed_images = self.classification_augmenter(
                labeled_images, training=True
            )
            with tf.GradientTape() as tape:
                # the encoder is used in inference mode here to avoid regularization
                # and updating the batch normalization paramers if they are used
                features = self.encoder(preprocessed_images, training=False)
                class_logits = self.linear_probe(features, training=True)
                probe_loss = self.probe_loss(labels, class_logits)
            gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
            self.probe_optimizer.apply_gradients(
                zip(gradients, self.linear_probe.trainable_weights)
            )
            self.probe_loss_tracker.update_state(probe_loss)
            self.probe_accuracy.update_state(labels, class_logits)

            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            labeled_images, labels = data

            # For testing the components are used with a training=False flag
            preprocessed_images = self.classification_augmenter(
                labeled_images, training=False
            )
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=False)
            probe_loss = self.probe_loss(labels, class_logits)
            self.probe_loss_tracker.update_state(probe_loss)
            self.probe_accuracy.update_state(labels, class_logits)

            # Only the probe metrics are logged at test time
            return {m.name: m.result() for m in self.metrics[2:]}


    # Contrastive pretraining
    pretraining_model = ContrastiveModel()

    pretraining_model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
        run_eagerly=True
    )

    pretraining_history = pretraining_model.fit(
        train_dataset, epochs=num_epochs, validation_data=test_dataset
    )

    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(pretraining_history.history["val_p_acc"]) * 100
        )
    )

    # Encoder for validation
    encoder_model = keras.Sequential(
        [
            layers.Input(shape=(temporal_samples, number_of_bands)),
            get_augmenter(**classification_augmentation),
            pretraining_model.encoder,
        ],
        name="encoder_model",
    )

    encoded_vectors = encoder_model.predict(X_val)

    print(encoded_vectors.shape)


    # Supervised finetuning of the pretrained encoder
    finetuning_model = keras.Sequential(
        [
            layers.Input(shape=(temporal_samples, number_of_bands)),
            get_augmenter(**classification_augmentation),
            pretraining_model.encoder,
            layers.Dense(1, activation='sigmoid'),
        ],
        name="finetuning_model",
    )
    finetuning_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    finetuning_history = finetuning_model.fit(
        train_dataset, epochs=num_epochs, validation_data=test_dataset
    )
    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(finetuning_history.history["val_acc"]) * 100
        )
    )

    baseline = "{:.2f}%".format(max(baseline_history.history["val_acc"]) * 100)
    pretraining = "{:.2f}%".format(max(pretraining_history.history["val_p_acc"]) * 100)
    finetuning = "{:.2f}%".format(max(finetuning_history.history["val_acc"]) * 100)

    parameters_str = ''
    for j in i:
        parameters_str = parameters_str + str(i[j]) + ','

    with open('results.txt', 'a') as file:
        file.writelines('SIMCLR,' + str(count) + ',' + parameters_str + baseline + ',' + pretraining + ',' + finetuning + '\n')
