import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import math

import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from pathlib import Path

#--------------------------------------------------------------------------------------
parameters = {
                'temporal_samples': [15, 20, 30],
                'band_features' : [['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']],
                'width': [30],
                'temperature': [0.2],
                'steps_per_epoch': [20],
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


    #------------------------------------------------------------------------------------------------------------

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

    from sklearn.model_selection import train_test_split

    X_array_n = X_array[np.array(labels) == 0]
    X_array_p = X_array[np.array(labels) == 1]

    print('Total of samples:', X_array.shape)

    labels_and_hexes = np.vstack((hexes, labels)).T



    X_array_anchors = X_array_p[0:math.floor((X_array_p.shape[0] / 2))]
    X_array_positives = X_array_p[math.floor(X_array_p.shape[0] / 2):-1]
    X_array_negatives = X_array_n[0:math.floor((X_array_p.shape[0] / 2))]

    print('Total of anchors samples:', X_array_anchors.shape)
    print('Total of positive samples:', X_array_positives.shape)
    print('Total of negative samples:', X_array_negatives.shape)

    samples_count = X_array_anchors.shape[0]
    print('Number of samples:', samples_count)


    def prepare_dataset():
        anchor_dataset = tf.data.Dataset \
            .from_tensor_slices(X_array_anchors)

        positive_dataset = tf.data.Dataset \
            .from_tensor_slices(X_array_positives)

        negative_dataset = tf.data.Dataset \
            .from_tensor_slices(X_array_negatives)

        return anchor_dataset, positive_dataset, negative_dataset


    anchor_dataset, positive_dataset, negative_dataset = prepare_dataset()
    print(anchor_dataset, positive_dataset, negative_dataset)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=128)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(samples_count * 0.8))
    val_dataset = dataset.skip(round(samples_count * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    # train_dataset = train_dataset.prefetch(8)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    # val_dataset = val_dataset.prefetch(8)

    print(train_dataset)
    print(val_dataset)

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


    embedding = get_encoder()
    from tensorflow.keras import Model


    class DistanceLayer(layers.Layer):
        """
        This layer is responsible for computing the distance between the anchor
        embedding and the positive embedding, and the anchor embedding and the
        negative embedding.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, anchor, positive, negative):
            ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
            an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
            return (ap_distance, an_distance)


    anchor_input = layers.Input(name="anchor", shape=(temporal_samples, number_of_bands))
    positive_input = layers.Input(name="positive", shape=(temporal_samples, number_of_bands))
    negative_input = layers.Input(name="negative", shape=(temporal_samples, number_of_bands))

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )


    class SiameseModel(Model):
        """The Siamese Network model with a custom training and testing loops.

        Computes the triplet loss using the three embeddings produced by the
        Siamese Network.

        The triplet loss is defined as:
           L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
        """

        def __init__(self, siamese_network, margin=0.5):
            super().__init__()
            self.siamese_network = siamese_network
            self.margin = margin
            self.loss_tracker = keras.metrics.Mean(name="loss")

        def call(self, inputs):
            return self.siamese_network(inputs)

        def train_step(self, data):
            # GradientTape is a context manager that records every operation that
            # you do inside. We are using it here to compute the loss so we can get
            # the gradients and apply them using the optimizer specified in
            # `compile()`.
            with tf.GradientTape() as tape:
                loss = self._compute_loss(data)

            # Storing the gradients of the loss function with respect to the
            # weights/parameters.
            gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

            # Applying the gradients on the model using the specified optimizer
            self.optimizer.apply_gradients(
                zip(gradients, self.siamese_network.trainable_weights)
            )

            # Let's update and return the training loss metric.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):
            loss = self._compute_loss(data)

            # Let's update and return the loss metric.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def _compute_loss(self, data):
            # The output of the network is a tuple containing the distances
            # between the anchor and the positive example, and the anchor and
            # the negative example.
            ap_distance, an_distance = self.siamese_network(data)

            # Computing the Triplet Loss by subtracting both distances and
            # making sure we don't get a negative value.
            loss = ap_distance - an_distance
            loss = tf.maximum(loss + self.margin, 0.0)
            return loss

        @property
        def metrics(self):
            # We need to list our metrics here so the `reset_states()` can be
            # called automatically.
            return [self.loss_tracker]


    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=keras.optimizers.Adam(0.0001))
    siamese_model.fit(train_dataset, epochs=50, validation_data=val_dataset)

    sample = next(iter(train_dataset))

    anchor, positive, negative = sample

    anchor_embedding, positive_embedding, negative_embedding = (
        embedding(anchor),
        embedding(positive),
        embedding(negative),
    )

    cosine_similarity = keras.metrics.CosineSimilarity()

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    print("Positive similarity:", positive_similarity.numpy())

    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    print("Negative similarity", negative_similarity.numpy())

    posneg_similarity = cosine_similarity(positive_embedding, negative_embedding)
    print("PosNeg similarity", posneg_similarity.numpy())

    from sklearn.model_selection import train_test_split

    print('Total of samples:', X_array.shape)

    labels_and_hexes = np.vstack((hexes, labels)).T

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

    print('Total of training samples:', X_train.shape, len(y_train))
    print('Total of test samples:', X_test.shape, len(y_test))
    print('Total of validation samples:', X_val.shape, len(y_val))


    def prepare_dataset():

        labeled_batch_size = labeled_train_samples
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
            .batch(batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

        return batch_size, train_dataset, test_dataset, validation_dataset


    batch_size, train_dataset, test_dataset, validation_dataset = prepare_dataset()

    # Supervised finetuning of the pretrained encoder
    finetuning_model = keras.Sequential(
        [
            layers.Input(shape=(temporal_samples, number_of_bands)),
            get_encoder(),
            layers.Dense(1, activation='sigmoid'),
        ],
        name="finetuning_model",
    )
    finetuning_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    print(train_dataset)
    print(val_dataset)
    finetuning_history = finetuning_model.fit(
        train_dataset, epochs=num_epochs * 10, validation_data=validation_dataset
    )
    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(finetuning_history.history["val_acc"]) * 100
        )
    )


    baseline = "{:.2f}%".format(0 * 100)
    pretraining = "{:.2f}%".format(0 * 100)
    finetuning = "{:.2f}%".format(max(finetuning_history.history["val_acc"]) * 100)

    parameters_str = ''
    for j in i:
        parameters_str = parameters_str + str(i[j]) + ','

    with open('results.txt', 'a') as file:
        file.writelines('TRIPLET_LOSS,' + str(count) + ',' + parameters_str + baseline + ',' + pretraining + ',' + finetuning + '\n')
