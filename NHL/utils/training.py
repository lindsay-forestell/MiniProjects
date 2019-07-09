# initial data generator

# Regular stuff
import numpy as np
import pandas as pd
import os

# normalize
from sklearn.preprocessing import MinMaxScaler

# Tensorflow and Keras stuff
from keras.models import load_model
from keras.utils import to_categorical, Sequence
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.layers import (Input, Dense, Activation, Dropout, LSTM, Concatenate)
from keras.models import Model
from keras import backend as K

# Save stuff
from sklearn.externals import joblib

# Modelling and Metrics
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve,
                             f1_score, auc, average_precision_score, matthews_corrcoef, recall_score, precision_score)

# Plotting stuff
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pydot
import keras
import seaborn as sns
keras.utils.vis_utils.pydot = pydot

# Current CAP value
CURRENT_CAP = 81.5e6


# Helper functions and classes
def plot_history(history, validation=False, save_file=None, display=True):
    """
    Plot the history that model.fit() will produce
    If you didn't include validation data, set validation flag to false
    """
    _, ax = plt.subplots()
    # As the first loss is usually very inaccurate, just drop it
    num_epochs = len(history.history['loss'])
    if validation:
        plt.plot(history.history['val_loss'][1:], c='b', linestyle='-', label='Validation Loss')
    plt.plot(history.history['loss'][1:], c='b', linestyle='--', label='Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax.set_xticks(range(num_epochs - 1))
    ax.set_xticklabels(range(1, num_epochs))
    plt.legend(loc='upper right')
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close()


def new_run_log_dir(base_dir):
    """
    create log directories for tensorboard
    :param base_dir:
    :return:
    """
    log_dir = os.path.join('./logs', base_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_id = len([name for name in os.listdir(log_dir)])
    run_log_dir = os.path.join(log_dir, str(run_id))
    return run_log_dir


def plot_results(player_name, signing_date, signing_status, engine_length, engine_cap, load_models=None,
                 save_file=True):
    # TODO reload a model at the very end so that it's not broken later
    """
    Plot the most likely contract length, with most likely contract sizes as well
    :param player_name: player name of interest
    :param signing_date: player signing date, (day, month, year)
    :param signing_status: 0 (RFA) or 1 (UFA)
    :param engine_length: engine that stores length models/data
    :param engine_cap: engine that stores cap hit models/data
    :param load_models: list of models to load into the engines.
                        should be a list of strings that will have length_model_ or cap_hit_model_
                        prepended to get the full path
                        assumes they are in the ./models/ folder, and that they end with .h5
    :param save_file: if true saves file to ./figs/player_name
    :return: nice lil plot
    """

    # Get the length predictions
    _, ax = plt.subplots()
    if load_models is None:
        lengths = engine_length.predict(player_name, signing_date, signing_status, verbose=False)
        max_height = np.max(lengths) + 0.07
        plt.bar(range(0, len(lengths)), lengths, color='lightsteelblue')
        plt.bar(np.argmax(lengths), np.max(lengths), color='mediumorchid')
        min_height = 0
    else:
        palette = ['lightsteelblue', 'lightsteelblue', 'lightsteelblue', 'lightsteelblue', 'lightsteelblue',
                   'lightsteelblue', 'lightsteelblue', 'lightsteelblue']
        base = './models/length_model_'
        lengths_all = [[], [], [], [], [], [], [], []]
        for ii, ext in enumerate(load_models):
            K.clear_session()
            model_file = base + ext
            engine_length.load_model(model_file)
            lengths_model_x = engine_length.predict(player_name, signing_date, signing_status, verbose=False)
            for jj in range(8):
                lengths_all[jj].append(lengths_model_x[jj])
        # lengths = np.mean(lengths_all, axis=1)
        # yerr = np.std(lengths_all, axis=1)
        # height = lengths+yerr
        # max_height = np.max(height)+0.07
        interquartile_range = np.percentile(lengths_all, 75, axis=1) - np.percentile(lengths_all, 25, axis=1)
        third_quartile = np.percentile(lengths_all, 75, axis=1)
        fences = third_quartile + 1.5 * interquartile_range
        median = np.median(lengths_all, axis=1)
        lengths = []
        # find height of upper whisker
        for kk, fence in enumerate(fences):
            lengths_kk = lengths_all[kk]
            best = 0
            for x in lengths_kk:
                if (x < fence) & (x > best):
                    best = x
            lengths.append(best)

        palette[int(np.argmax(median))] = 'mediumorchid'
        sns.boxplot(data=lengths_all, palette=palette)
        max_height = np.max(lengths) + 0.07
        min_height = -0.05

    # Plot the lengths as a bar plot
    # plt.plot(range(1, len(lengths) + 1), lengths, color='red')
    plt.xlabel('Length of Contract')
    plt.ylabel('Probability of Length')
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(1, 9))
    plt.ylim([min_height, max_height])
    plt.title(player_name)

    # Get the cap prediction for each length
    for length in range(1, 9):
        if load_models is None:
            cap_hit, _ = engine_cap.predict(player_name, signing_date, signing_status, length, verbose=False)
            cap_hit_std = 0
        else:
            base = './models/cap_hit_model_'
            cap_hit_all = np.empty(len(load_models))
            for ii, ext in enumerate(load_models):
                K.clear_session()
                model_file = base + ext
                engine_cap.load_model(model_file)
                cap_hit_all[ii], _ = engine_cap.predict(player_name, signing_date, signing_status, length,
                                                        verbose=False)
            cap_hit = np.mean(cap_hit_all)
            cap_hit_std = np.std(cap_hit_all)

        cap_hit = cap_hit / 1e6
        cap_hit_std = cap_hit_std / 1e6
        ax.text(length - 1, lengths[length - 1] + 0.02, '{:0.1f}({:0.1f})'.format(cap_hit, cap_hit_std),
                horizontalalignment='center')

    if save_file:
        plt.savefig('./figs/' + player_name.lower().replace(' ', '_') + '.png', bbox_inches='tight')
    plt.show()


def make_model(prediction_type, lstm_layers, dense_layers, activation='relu'):
    """
    Produce LSTM models that can be used for prediction
    :param prediction_type: either cap_hit or length (as the output differs)
    :param lstm_layers: list of hidden nodes wanted for the lstm layers
    :param dense_layers: list of hidden nodes wanted for the dense layers
    :param activation: activation to use in the dense layers
    :return: model
    """
    num_game_features = 105
    if prediction_type == 'cap_hit':
        num_static_features = 8
        output_size = 1
    else:
        num_static_features = 7
        output_size = 8

    input_1 = Input(batch_shape=(None, None, num_game_features))
    input_2 = Input(batch_shape=(None, num_static_features))

    # Loop through all the LSTM Layers
    lstm_layer = LSTM(lstm_layers[0], return_sequences=True)(input_1)
    for hidden_size in lstm_layers[1:-1]:
        lstm_layer = LSTM(hidden_size, return_sequences=True)(lstm_layer)
    lstm_final = LSTM(lstm_layers[-1])(lstm_layer)

    # Add in the static info
    fc = Concatenate()([lstm_final, input_2])

    # Loop through all the Dense Layers
    for hidden_size in dense_layers:
        fc = Dense(hidden_size)(fc)
        fc = Activation(activation)(fc)

    out = Dense(output_size)(fc)

    if prediction_type == 'length':
        out = Activation('softmax')(out)

    # Create the actual model and return it
    model = Model(inputs=[input_1, input_2], outputs=out)

    return model


class DataGenerator(Sequence):
    """
    Generates data to predict the avg cap hit OR length of contract for a player.
    Uses as input the length of the contract for avg cap hit, or uses it as target for length
    """

    def __init__(self, player_file,
                 contract_file,
                 game_file,
                 prediction_type,
                 batch_size=32,
                 shuffle=True,
                 val_size=0.3,
                 test_size=0.1):
        """
        Initialize the Data Generator
        :param player_file: location of the CSV containing player info
        :param contract_file: location of the CSV containing contract info
        :param game_file: location of the CSV containing game info
        :param prediction_type: whether this is to be used for the 'cap_hit' or 'length' prediction
        :param batch_size: size of batch to use
        :param shuffle: whether to shuffle the training batches each epoch
        :param val_size: size of validation set
        :param test_size: size of test set
        """
        # Fixed inputs
        self.prediction_type = prediction_type
        self.num_game_features = 105

        # different num features for 2 types of predictions
        if self.prediction_type == 'cap_hit':
            self.num_static_features = 8
        else:
            self.num_static_features = 7

        # Set up the data
        # dataframe containing open, close, high and low data
        self.player_stats = pd.read_csv(player_file).reset_index(drop=True)

        # make sure games are sorted so that the earliest time step is first
        self.game_stats = pd.read_csv(game_file).reset_index(drop=True)
        self.game_stats = self.game_stats.sort_values(by=['season']).reset_index(drop=True)

        # set scaler for game data
        self.game_scaler = MinMaxScaler()
        numerical_game_stats = self.game_stats.drop(
            columns=['player_id', 'player', 'season', 'player_season', 'position']).values
        self.game_scaler.fit(numerical_game_stats)
        # columns = list(self.game_stats.columns)
        # for col in ['player_id', 'player', 'season', 'player_season', 'position']:
        #     columns.remove(col)
        # self.game_stats[columns] = numerical_transformed

        # only look at standard contracts
        self.contract_stats = pd.read_csv(contract_file).reset_index(drop=True)
        self.contract_stats = self.contract_stats[self.contract_stats.contract_type == 'STANDARD CONTRACT'].reset_index(
            drop=True)

        # set scaler for static data
        self.static_scaler = MinMaxScaler()
        if self.prediction_type == 'cap_hit':
            scale_columns = ['signing_age', 'length']
        else:
            scale_columns = ['signing_age']
        numerical_contract_stats = self.contract_stats[scale_columns].values
        self.static_scaler.fit(numerical_contract_stats)
        # self.contract_stats[scale_columns] = numerical_contract_transformed

        # Variable inputs
        self.batch_size = batch_size
        self.num_data = len(self.contract_stats)
        self.shuffle = shuffle
        self.val_size = val_size
        self.test_size = test_size

        self.num_train = int(np.ceil((1 - test_size - val_size) * self.num_data))
        self.num_val = int(val_size * self.num_data)
        self.num_test = self.num_data - self.num_train - self.num_val

        self.num_train_batches = np.ceil(self.num_train / self.batch_size)

        # Train, test, val indices
        self.indices = np.arange(self.num_data)
        np.random.shuffle(self.indices)

        self.train_indices = self.indices[:self.num_train]
        self.val_indices = self.indices[self.num_train:self.num_train + self.num_val]
        self.test_indices = self.indices[self.num_train + self.num_val:]

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: number of training batches
        """
        # We want to go all the way through our TRAINING dataset each epoch
        return int(self.num_train_batches)

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of interest
        :return: batch of training data
        """
        # Generate indexes of the batch
        if index < 0:
            index = self.num_train_batches + index
        # if index would take us over the size of our data, use a truncated batch
        if index * self.batch_size + self.batch_size >= self.num_train:
            indexes = self.train_indices[index * self.batch_size:self.num_train]
        else:
            indexes = self.train_indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """
        Shuffles the batches if required
        :return: nothing. shuffles in place.
        """
        if self.shuffle:
            np.random.shuffle(self.train_indices)

    def data_generation(self, indexes):
        """
        Generates data based on given indicies
        :param indexes: can be training, validation, or test indices
        :return: [X1, X2], y
        """
        # Initialization
        batch_size = len(indexes)

        # When training, we need to have the same size tensors every iteration
        # Each player has at most 12 seasons, so we zero pad for players with fewer seasons
        x1 = np.zeros((batch_size, 12, self.num_game_features))
        x2 = np.zeros((batch_size, self.num_static_features))
        y = []

        # Generate data
        for i, idx in enumerate(indexes):

            player_id = self.contract_stats.loc[idx, 'player_id']

            contract_date = self.contract_stats.loc[idx, 'signing_date'].strip('()').split(', ')
            contract_year = int(contract_date[2])
            contract_month = int(contract_date[1])
            if contract_month <= 6:
                contract_year -= 2
            else:
                contract_year -= 1
            # this is the last year that should be INCLUDED in the game stats
            contract_season = int(str(contract_year) + str(contract_year + 1))

            # All the data for seasons previous to contract season
            game_stats = self.game_stats[
                (self.game_stats.player_id == player_id) & (self.game_stats.season <= contract_season)]
            game_stats = game_stats.drop(columns=['player_id', 'player', 'season', 'player_season', 'position']).values
            num_time_steps = game_stats.shape[0]

            # As long as the player exists and has contract years
            # Add in their transformed info
            if num_time_steps > 0:
                x1[i, -num_time_steps:, :] = self.game_scaler.transform(game_stats)

            # extra info should include d, lw, rw, c, handedness, signing_age, status,
            # length (if predicting for cap hit)
            status = self.contract_stats.loc[idx, 'signing_status']
            age = self.contract_stats.loc[idx, 'signing_age']
            length = self.contract_stats.loc[idx, 'length']
            defense = self.player_stats.loc[self.player_stats.player_id == player_id, 'defense'].values[0]
            center = self.player_stats.loc[self.player_stats.player_id == player_id, 'center'].values[0]
            right_wing = self.player_stats.loc[self.player_stats.player_id == player_id, 'right_wing'].values[0]
            left_wing = self.player_stats.loc[self.player_stats.player_id == player_id, 'left_wing'].values[0]
            hand = self.player_stats.loc[self.player_stats.player_id == player_id, 'handed'].values[0]

            # make sure to transform the static data
            if self.prediction_type == 'cap_hit':
                age_length_transformed = self.static_scaler.transform(np.array([[age, length]]))
                age = age_length_transformed[0, 0]
                length = age_length_transformed[0, 1]
                x2[i, :] = np.array([status, age, length, defense, center, right_wing, left_wing, hand])
            else:
                age = self.static_scaler.transform(np.array([[age]]))[0, 0]
                x2[i, :] = np.array([status, age, defense, center, right_wing, left_wing, hand])

            # get the target: pct of cap OR length
            if self.prediction_type == 'cap_hit':
                pct_of_cap = self.contract_stats.loc[idx, 'cap_hit_pct']
                y.append(pct_of_cap)

            else:
                if length > 8:
                    length = 8
                length -= 1
                y.append(length)

        y = np.array(y).reshape(-1, 1)

        # Convert the length to categorical if its the target
        if self.prediction_type == 'length':
            y = to_categorical(y, num_classes=8)

        return [x1, x2], y

    def get_single_player(self, player_name, signing_date, signing_status, contract_length=5):
        """
        Return stats that can be used to guess a new contract for a single player.
        Some info isn't readily available in the database, so needs to be put in by hand
        :param player_name: name of player. Hopefully in database.
        :param signing_date: When the contract is expected to be signed. Tuple in form (day, month, year)
        :param signing_status: Whether they are an RFA (0) or UFA (1)
        :param contract_length: Estimated length of contract. (int, 1-8)
        :return: [game_stats, static_stats]
        """
        # first try to get the player from the player_stats df
        player_name = player_name.upper().replace(' ', '.')
        try:
            player_id = self.player_stats[self.player_stats.player == player_name].reset_index().loc[0, 'player_id']
        except KeyError:
            print('Player not found.')
            return None

        dob = self.player_stats[self.player_stats.player_id == player_id].reset_index().loc[0, 'date_of_birth']
        birth_list = dob.strip('()').split(', ')
        date, month, year = [int(val) for val in birth_list]
        contract_date, contract_month, contract_year = signing_date
        signing_age = contract_year - year

        # get the last year that should be included in the game stats
        if (date + month * 1000) > (contract_date + contract_month * 1000):
            signing_age -= 1

        # Figure out which years to include in the prediction
        if contract_month <= 6:
            contract_year -= 2
        else:
            contract_year -= 1
        # this is the last year that should be INCLUDED in the game stats
        contract_season = int(str(contract_year) + str(contract_year + 1))

        game_stats = self.game_stats[
            (self.game_stats.player_id == player_id) & (self.game_stats.season <= contract_season)]
        game_stats = game_stats.drop(columns=['player_id', 'player', 'season', 'player_season', 'position']).values
        # transform the data
        game_stats = self.game_scaler.transform(game_stats)
        # add a batch dimension
        game_stats = game_stats[np.newaxis, ...]

        # rescale the ['signing_age', 'length'] variables
        to_scale = np.array([[signing_age, contract_length]])
        scaled = self.static_scaler.transform(to_scale)
        signing_age = scaled[0, 0]
        contract_length = scaled[0, 1]

        # extra info should include d, lw, rw, c, handedness, signing_age, status, length
        defense = self.player_stats.loc[self.player_stats.player_id == player_id, 'defense'].values[0]
        center = self.player_stats.loc[self.player_stats.player_id == player_id, 'center'].values[0]
        right_wing = self.player_stats.loc[self.player_stats.player_id == player_id, 'right_wing'].values[0]
        left_wing = self.player_stats.loc[self.player_stats.player_id == player_id, 'left_wing'].values[0]
        hand = self.player_stats.loc[self.player_stats.player_id == player_id, 'handed'].values[0]
        if self.prediction_type == 'cap_hit':
            static_stats = np.array([signing_status,
                                     signing_age,
                                     contract_length,
                                     defense,
                                     center,
                                     right_wing,
                                     left_wing,
                                     hand])
        else:
            static_stats = np.array([signing_status,
                                     signing_age,
                                     defense,
                                     center,
                                     right_wing,
                                     left_wing,
                                     hand])
        static_stats = static_stats[np.newaxis, ...]

        return [game_stats, static_stats]


class Engine:
    """
    Training Engine for the CapHit or Length Calculation
    Performs training and evaluation
    """

    def __init__(self, model, data_generator, optimizer, prediction_type):
        """
        Initialize model
        :param model: model to fit
        :param data_generator: data to fit
        :param optimizer: optimizer to use
        :param prediction_type: type of model to build (cap_hit or length)
        """
        self.history = None
        self.data_generator = data_generator
        self.prediction_type = prediction_type

        self.model = model
        self.optimizer = optimizer

        if self.prediction_type == 'cap_hit':
            self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        else:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=self.optimizer,
                               metrics=[categorical_accuracy])

    def model_summary(self, summary_type=0, save_file=None):
        """
        Show summaries of the model
        :param summary_type: 0 for standard keras model summary, 1 for a visual depiction of the network architecture
        :param save_file: If you want to save it, include a filename for the image
        :return: the summary
        """
        if summary_type == 0:
            return self.model.summary()
        else:
            if save_file is None:
                return SVG(model_to_dot(self.model).create(prog='dot', format='svg'))
            else:
                plot_model(self.model, to_file=save_file)
                return SVG(model_to_dot(self.model).create(prog='dot', format='svg'))

    def load_model(self, model_file):
        """
        Load a model from memory
        :param model_file: location of h5 file (without the extension)
        :return: nothing
        """
        self.model = load_model(model_file+'.h5')
        # also load the scalers that should be associated with the model
        game_scaler_file = model_file + '_gs.sav'
        self.data_generator.game_scaler = joblib.load(game_scaler_file)

        static_scaler_file = model_file + '_ss.sav'
        self.data_generator.static_scaler = joblib.load(static_scaler_file)

    def save_model(self, model_file):
        """
        Save model to h5 file
        :param model_file: Where to save file (without the extension)
        :return: nothing
        """
        self.model.save(model_file+'.h5')
        # also save the scalers so that we scale the same way as the model was trained on
        game_scaler_file = model_file + '_gs.sav'
        joblib.dump(self.data_generator.game_scaler, game_scaler_file)

        static_scaler_file = model_file + '_ss.sav'
        joblib.dump(self.data_generator.static_scaler, static_scaler_file)

    def fit(self, epochs=10, tensorboard=False, early_stopping=False, verbose=2):
        """
        Fit the model using the given dataset
        :param epochs: number of epochs to run for
        :param tensorboard: whether or not to use the TB callback
        :param early_stopping: whether or not to use early stopping
        :param verbose: how much to print out each epoch
        :return:
        """
        val_x, val_y = self.data_generator.data_generation(self.data_generator.val_indices)
        callbacks = []

        # Check for callbacks to use
        if tensorboard:
            log_dir = new_run_log_dir(self.prediction_type+'_logs')
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
        if early_stopping:
            callbacks.append(EarlyStopping(patience=3, restore_best_weights=True))

        self.history = self.model.fit_generator(self.data_generator,
                                                epochs=epochs,
                                                validation_data=(val_x, val_y),
                                                verbose=verbose,
                                                callbacks=callbacks)

    def plot_history(self, validation=False, save_file=None, display=True):
        """
        plot the history obtained from the latest fit
        :param validation:
        :param save_file: if you want to save it, include a save file name as well
        :param display: is you want to save but not display set to False
        :return:
        """
        if self.history is not None:
            plot_history(self.history, validation, save_file, display)
        else:
            print("Need to do a fit first. Call engine.fit()")

    def predict(self, player_name, signing_date, signing_status, length=5, verbose=True):
        """
        Predict the salary for a new player
        :param player_name: "First Last"
        :param length: expected length of contract
        :param signing_date: in tuple form (day, month, year)
        :param signing_status: 0 for RFA, 1 for UFA
        :param verbose: print out the result or not
        :return: cap hit, cap hit pct
        """
        x = self.data_generator.get_single_player(player_name, signing_date, signing_status, length)

        if self.prediction_type == 'cap_hit':
            cap_hit_pct = self.model.predict(x)[0][0]
            cap_hit = cap_hit_pct*CURRENT_CAP

            if verbose:
                print("Current Cap:           {:0.3f} M".format(CURRENT_CAP/1e6))
                print("Total Contract Value:  {:0.3f} M".format(cap_hit*length/1e6))
                print("Expected Cap Hit:      {:0.3f} M".format(cap_hit/1e6))
                print("Percentage of Cap:     {:0.3f} %".format(cap_hit_pct*100))

            return cap_hit, cap_hit_pct

        else:
            lengths = self.model.predict(x)[0]

            if verbose:
                max_length = 0
                max_prob = 0
                for ii, length in enumerate(lengths):
                    print("Length: {:d} ** Probability: {:0.3f}".format(ii + 1, length))
                    if length > max_prob:
                        max_prob = length
                        max_length = ii + 1

                print("\nMost likely contract length: ", max_length)

            return lengths
