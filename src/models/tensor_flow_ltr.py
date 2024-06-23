import sys
import os
import numpy as np
# import tensorflow as tf
# import tensorflow_ranking as tfr
# import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.utils import *
from src.experiment_helpers import *

# Convert games to features and labels for list-wise ranking
def prepare_data(games):
    data = []
    for i, game in enumerate(games):
        for idx, player in enumerate(game):
            data.append({'game': i, 'player': player, 'position': idx})
    return data

class RankingModel(tfrs.Model):
    def __init__(self, unique_player_ids, unique_game_ids):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for players.
        self.player_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_player_ids),
            tf.keras.layers.Embedding(len(unique_player_ids) + 2, embedding_dimension)
        ])

        # Compute embeddings for games.
        self.game_embeddings = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=unique_game_ids),
            tf.keras.layers.Embedding(len(unique_game_ids) + 2, embedding_dimension)
        ])

        # Compute predictions.
        self.score_model = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
    
            tf.keras.layers.Dense(64, activation="relu"),
            # Make ranking predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

    def call(self, features):
        # Convert the id features into embeddings.
        # Player embeddings are a [batch_size, embedding_dim] tensor.
        player_embeddings = self.player_embeddings(tf.strings.as_string(features["player_id"]))

        # Game embeddings are a [batch_size, embedding_dim] tensor.
        game_embeddings = self.game_embeddings(features["game_id"])

        # Reshape player_embeddings to match the shape of game_embeddings.
        player_embedding_repeated = tf.expand_dims(player_embeddings, 1)

        # Concatenate and pass into the dense layers to generate predictions.
        concatenated_embeddings = tf.concat([player_embedding_repeated, tf.expand_dims(game_embeddings, 1)], axis=2)

        return self.score_model(concatenated_embeddings)

def predict_ratings(data, model):
    # Prepare the dataset for prediction
    player_ids = [str(item['player']) for item in data]
    game_ids = [item['game'] for item in data]
    dataset = tf.data.Dataset.from_tensor_slices({
        "player_id": tf.convert_to_tensor(player_ids, dtype=tf.string),
        "game_id": tf.convert_to_tensor(game_ids, dtype=tf.int32)
    }).batch(1)

    # Predict the ratings
    predictions = model.predict(dataset)

    # Create a dictionary of player ratings
    player_ratings = {}
    for data_row, pred in zip(data, predictions):
        player = data_row['player']
        rating = pred[0][0]  # Assuming pred shape is (1, 1)
        if player in player_ratings:
            player_ratings[player].append(rating)
        else:
            player_ratings[player] = [rating]

    return player_ratings


def compute_predicited_rankings_tensor_flow(games, pi_values):

    # Prepare the data
    prepared_data = prepare_data(games)

    # Define unique IDs
    unique_player_ids = np.unique([str(item['player']) for item in prepared_data])
    unique_game_ids = np.unique([item['game'] for item in prepared_data])

    # Create the model
    model = RankingModel(unique_player_ids=unique_player_ids, unique_game_ids=unique_game_ids)

    # Compile the model (even though we are not training, compilation is required for prediction)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    # Predict ratings for the prepared data
    player_ratings = predict_ratings(prepared_data, model)

    for key, value in player_ratings.items():
        pi_values[key] = np.mean(value)

    # shift data into positive space
    min = sorted(pi_values.values())[0] - .1
    pi_values = { key: value - min for key, value in pi_values.items()}
    normalize_scores(pi_values)

    return pi_values



if __name__ == '__main__':



    # Read data
    data, pi_values = read_strict_ordered_dataset('datasets/processed_data/00045-00000014.soc')


    train, test = train_test_split(data)
    
    
    ratings = compute_predicited_rankings_tensor_flow(train, pi_values)
    # print(ratings)
   
    print(np.mean(compute_likelihood(ratings, test)))



