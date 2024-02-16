import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


def load_rating_movies(local: bool = True):

    if local:
        # Load the dataset from local
        ratings = tf.data.Dataset.load("data/retrieval/ratings.tfrecord")
        movies = tf.data.Dataset.load("data/retrieval/movies.tfrecords")
    else:
        # Ratings data.
        ratings = tfds.load("movielens/100k-ratings", split="train")
        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
        })
        ratings.save("data/retrieval/ratings.tfrecord")

        movies = tfds.load("movielens/100k-movies", split="train")
        movies = movies.map(lambda x: x["movie_title"])
        movies.save("data/retrieval/movies.tfrecords")

    for x in ratings.take(1).as_numpy_iterator():
        pprint.pprint(x)

    for x in movies.take(1).as_numpy_iterator():
        pprint.pprint(x)

    return ratings, movies


def split_train_test(tf_data):
    tf.random.set_seed(42)
    shuffled = tf_data.shuffle(
        100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    train.save("data/retrieval/train.tfrecord")

    test = shuffled.skip(80_000).take(20_000)
    test.save("data/retrieval/test.tfrecord")

    return train, test


def get_unique_user_movie(tf_uesr: object, tf_movie: object):
    movie_titles = tf_movie.batch(1_000)
    user_ids = tf_uesr.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    pprint.pprint(unique_user_ids[:10])
    pprint.pprint(unique_movie_titles[:10])

    return unique_user_ids, unique_movie_titles


def data_process():
    ratings, movies = load_rating_movies(local=True)
    train, test = split_train_test(ratings)
    unique_user_ids, unique_movie_titles = get_unique_user_movie(
        ratings, movies)


def create_user_model(unique_user_ids: object, embedding_dimension: int):
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(
            len(unique_user_ids) + 1, embedding_dimension)
    ])

    return user_model


def create_movie_model(unique_movie_titles: object, embedding_dimension: int):
    movie_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(
            len(unique_movie_titles) + 1, embedding_dimension)
    ])

    return movie_model


def loss(movies: object, movie_model: object, batch_size: int):

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(batch_size).map(movie_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )

    return task


class MovielensModel(tfrs.Model):

    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)


def fit():
    ratings, movies = load_rating_movies(local=True)
    train, test = split_train_test(ratings)
    unique_user_ids, unique_movie_titles = get_unique_user_movie(
        ratings, movies)

    user_model = create_user_model(
        unique_user_ids=unique_user_ids, embedding_dimension=32)
    movie_model = create_movie_model(
        unique_movie_titles=unique_movie_titles, embedding_dimension=32)

    task = loss(movies=movies, movie_model=movie_model, batch_size=128)
    model = MovielensModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=3)

    model.evaluate(cached_test, return_dict=True)

    return model


def predict(model: object):
    _, movies = load_rating_movies(local=True)
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip(
            (movies.batch(100), movies.batch(100).map(model.movie_model)))
    )

    # Get recommendations.
    x, titles = index(tf.constant(["42"]))
    print(x)
    print(f"Recommendations for user 42: {titles[0, :3]}")

    # Export the query model.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model")

        # Save the index.
        tf.saved_model.save(index, path)

        # Load it back; can also be done in TensorFlow Serving.
        loaded = tf.saved_model.load(path)

        # Pass a user id in, get top predicted movie titles back.
        scores, titles = loaded(["42"])

        print(f"Recommendations: {titles[0][:3]}")


# TODO ? compute loss: training=False
# why the basic structure of tf.keras.Model
