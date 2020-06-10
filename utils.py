import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import Path
import pickle
import os


def write_pickle(path, data):
    """Write pickle file

    Save for reproducibility

    Args:
        path: The path we want to write the pickle file at
        data: The data we want to save in the pickle file
    """
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(path):
    """Read pickle file

    Make sure we don't overwrite our batches if we are validating and testing

    Args:
        path: The path we want to check the batches

    return: Data that we already stored in the pickle file
    """
    path = Path(path)
    data = None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_chem_dataset(k_shot, cross_validation=True, meta_batch_size=32, num_batches=100, verbose=False):
    """Load in the chemistry data for training

    "I'm limited by the technology of my time."
    Ideally the model would choose from a Uniformly distributed pool of unlabeled reactions
    Then we would run that reaction in the lab and give it back to the model
    The issue is the labs are closed, so we have to restrict the model to the reactions drawn
    from uniform distributions that we have labels for
    The list below is a list of inchi keys for amines that have a reaction drawn from a uniform
    distribution with a successful outcome (no point in amines that have no successful reaction)

    Args:
        k_shot:             An integer. The number of unseen classes in the dataset
        params:             A dictionary. The dictionary that is initialized with parameters.
                            Use the key "cross_validate" in the dictionary to separate
                            loading training data and testing data
        meta_batch_size:    An integer. The batch size for meta learning, default is 32
        num_batches:        An integer. The batch size for training, default is 100
        verbose:            A boolean that gives information about
                            the number of features to train on is

    return:
        amine_left_out_batches:         A dictionary of batches with structure:
                                        key is amine left out,
                                        value has following hierarchy
                                        batches -> x_t, y_t, x_v, y_v -> meta_batch_size number of amines ->
                                        k_shot number of reactions -> each reaction has some number of features
        amine_cross_validate_samples:   A dictionary of batches with structure:
                                        key is amine which the data is for,
                                        value has the following hierarchy
                                        x_s, y_s, x_q, y_q -> k_shot number of reactions ->
                                        each reaction has some number of features
        amine_test_samples:             A dictionary that has the same structure as
                                        amine_cross_validate_samples
        counts:                         A dictionary to record the number of
                                        successful and failed reactions in the format of
                                        {"total": [# of failed reactions, # of successful reactions]}
    """
    viable_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                     'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                     'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
                     'LCTUISCIGMWMAT-UHFFFAOYSA-N',
                     'JERSPYRKVMAEJY-UHFFFAOYSA-N',
                     'JMXLWMIFDJCGBV-UHFFFAOYSA-N',
                     'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
                     'WGYRINYTHSORGH-UHFFFAOYSA-N',
                     'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                     'VNAAUNTYIONOHR-UHFFFAOYSA-N',
                     'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                     'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                     'XFYICZOIWSBQSK-UHFFFAOYSA-N',
                     'UMDDLGMCNFAZDX-UHFFFAOYSA-O',
                     'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                     'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                     'CALQKRVFTWDYDG-UHFFFAOYSA-N',
                     'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                     'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                     'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']

    # Set up various strings corresponding to headers
    distribution_header = '_raw_modelname'
    amine_header = '_rxn_organic-inchikey'
    score_header = '_out_crystalscore'
    name_header = 'name'
    to_exclude = [score_header, amine_header, name_header, distribution_header]
    path = './data/0050.perovskitedata_DRP.csv'

    # Successful reaction is defined as having a crystal score of...
    SUCCESS = 4

    # Get amine and distribution counts for the data
    df = pd.read_csv(path)

    # Set up the 0/1 labels and drop non-uniformly distributed reactions

    df = df[df[distribution_header].str.contains('Uniform')]
    df = df[df[amine_header].isin(viable_amines)]

    if verbose:
        print('There should be 1661 reactions here, shape is', df.shape)
    df[score_header] = [1 if val == SUCCESS else 0 for val in df[score_header].values]
    amines = df[amine_header].unique().tolist()

    # Hold out 4 amines for testing, the other 16 are fair game for the cross validation
    # I basically picked these randomly since I have no idea which inchi key corresponds to what
    hold_out_amines = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                       'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                       'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                       'JMXLWMIFDJCGBV-UHFFFAOYSA-N']

    if cross_validation:

        amines = [a for a in amines if a not in hold_out_amines]

        # Used to set up our weighted loss function
        counts = {}
        all_train = df[df[amine_header].isin(amines)]
        print('Number of reactions in training set', all_train.shape[0])
        all_train_success = all_train[all_train[score_header] == 1]
        print('Number of successful reactions in the training set',
              all_train_success.shape[0])

        # [Number of failed reactions, number of successful reactions]
        counts['total'] = [all_train.shape[0] -
                           all_train_success.shape[0], all_train_success.shape[0]]

        amine_left_out_batches = {}
        amine_cross_validate_samples = {}

        for amine in amines:
            # Since we are doing cross validation, create a training set without each amine
            print("Generating batches for amine", amine)
            available_amines = [a for a in amines if a != amine]

            all_train = df[df[amine_header].isin(available_amines)]
            print(
                f'Number of reactions in training set holding out {amine}', all_train.shape[0])
            all_train_success = all_train[all_train[score_header] == 1]
            print(
                f'Number of successful reactions in training set holding out {amine}', all_train_success.shape[0])

            counts[amine] = [all_train.shape[0] -
                             all_train_success.shape[0], all_train_success.shape[0]]
            batches = []
            for _ in range(num_batches):
                # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
                batch = generate_batch(df, meta_batch_size,
                                       available_amines, to_exclude, k_shot)
                batches.append(batch)

            amine_left_out_batches[amine] = batches
            # print("hey this is {}".format(batches))

            # Now set up the cross validation data
            X = df[df[amine_header] == amine]
            y = X[score_header].values
            X = X.drop(to_exclude, axis=1).values
            cross_valid = generate_valid_test_batch(X, y, k_shot)

            amine_cross_validate_samples[amine] = cross_valid

        print('Generating testing batches for training')
        amine_test_samples = load_test_samples(hold_out_amines, df, to_exclude, k_shot, amine_header, score_header)

        if verbose:
            print('Number of features to train on is',
                  len(df.columns) - len(to_exclude))

        return amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts

    else:
        print('Holding out', hold_out_amines)

        available_amines = [a for a in amines if a not in hold_out_amines]
        # Used to set up our weighted loss function
        counts = {}
        all_train = df[df[amine_header].isin(available_amines)]
        print('Number of reactions in training set', all_train.shape[0])
        all_train_success = all_train[all_train[score_header] == 1]
        print('Number of successful reactions in the training set',
              all_train_success.shape[0])

        counts['total'] = [all_train.shape[0] -
                           all_train_success.shape[0], all_train_success.shape[0]]

        batches = []
        print('Generating training batches')
        for _ in range(num_batches):
            # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
            batch = generate_batch(df, meta_batch_size,
                                   available_amines, to_exclude, k_shot)
            batches.append(batch)

        print('Generating testing batches for testing! DO NOT RUN IF YOU SEE THIS LINE!')
        amine_test_samples = load_test_samples(hold_out_amines, df, to_exclude, k_shot, amine_header, score_header)

        if verbose:
            print('Number of features to train on is',
                  len(df.columns) - len(to_exclude))

        return batches, amine_test_samples, counts


def generate_batch(df, meta_batch_size, available_amines, to_exclude, k_shot, amine_header='_rxn_organic-inchikey',
                   score_header='_out_crystalscore'):
    """Generate the batch for training amines

    Args:
        df:                 The data frame of the amines data
        meta_batch_size:    An integer. Batch size for meta learning
        available_amines:   A list. The list of amines that we are generating batches on
        to_exclude:         A list. The columns in the dataset that we need to drop
        k_shot:             An integer. The number of unseen classes in the dataset
        amine_header:       The header of the amine list in the data frame,
                            default = '_rxn_organic-inchikey'
        score_header:       The header of the score header in the data frame,
                            default = '_out_crystalscore'

    return: A list of the batch with
    training and validation features and labels in numpy arrays.
    The format is [[training_feature],[training_label],[validation_feature],[validation_label]]
    """
    x_t, y_t, x_v, y_v = [], [], [], []

    for _ in range(meta_batch_size):
        # Grab the tasks
        X = df[df[amine_header] == np.random.choice(available_amines)]

        y = X[score_header].values

        # Drop these columns from the dataset
        X = X.drop(to_exclude, axis=1).values

        # Standardize features since they are not yet standardized in the dataset
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
        qry = np.random.choice(X.shape[0], size=k_shot, replace=False)

        x_t.append(X[spt])
        y_t.append(y[spt])
        x_v.append(X[qry])
        y_v.append(y[qry])

    return [np.array(x_t), np.array(y_t), np.array(x_v), np.array(y_v)]


def generate_valid_test_batch(X, y, k_shot):
    """Generate the batches for the amine used for cross validation or testing

    Args:
        X:      Dataframe. The features of the chosen amine in the dataset
        y:      Dataframe. The labels of the chosen amine in the dataset
        k_shot: An integer. The number of unseen classes in the dataset

    return: A list of the features and labels for the amine
    """
    spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
    qry = [i for i in range(len(X)) if i not in spt]
    if len(qry) <= 5:
        print("Warning: minimal testing data for meta-learn assessment")

    x_s = X[spt]
    y_s = y[spt]
    x_q = X[qry]
    y_q = y[qry]

    scaler = StandardScaler()
    scaler.fit(x_s)

    x_s = scaler.transform(x_s)
    x_q = scaler.transform(x_q)

    return [x_s, y_s, x_q, y_q]


def load_test_samples(hold_out_amines, df, to_exclude, k_shot, amine_header, score_header):
    """This is a function used for loading testing samples specifically

    Args:
        hold_out_amines:    The list of all holdout amines that are used for testing.
                            DO NOT TOUCH!
        df:                 The data frame of the amines data
        to_exclude:         A list. The columns in the dataset that we need to drop
        k_shot:             An integer. The number of unseen classes in the dataset
        amine_header:       The header of the amine list in the data frame.
        score_header:       The header of the score header in the data frame.

    return: A dictionary that contains the test sample amines' batches
    """
    amine_test_samples = {}
    for a in hold_out_amines:
        # grab task
        X = df[df[amine_header] == a]

        y = X[score_header].values
        X = X.drop(to_exclude, axis=1).values
        test_sample = generate_valid_test_batch(X, y, k_shot)

        amine_test_samples[a] = test_sample
    return amine_test_samples

def save_model(params, amine=None):
    """This is to save models

    Create specific folders and store the model in the folder

    Args:
        params: A dictionary of the initialized parameters
        amine:  The specific amine that we want to store models for.
                Default is None

    return: The path for dst_folder
    """
    dst_folder_root = '.'
    dst_folder = ""
    if amine is not None and amine in params["training_batches"]:
        dst_folder = '{0:s}/PLATIPUS_few_shot/PLATIPUS_{1:s}_{2:d}way_{3:d}shot_{4:s}'.format(
            dst_folder_root,
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class'],
            amine
        )
    elif amine is not None and amine in params["validation_batches"]:
        dst_folder = '{0:s}/PLATIPUS_few_shot/PLATIPUS_{1:s}_{2:d}way_{3:d}shot_{4:s}'.format(
            dst_folder_root,
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class'],
            amine
        )
        return dst_folder
    else:
        dst_folder = '{0:s}/PLATIPUS_few_shot/PLATIPUS_{1:s}_{2:d}way_{3:d}shot'.format(
            dst_folder_root,
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class']
        )
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print('No folder for storage found')
        print(f'Make folder to store meta-parameters at')
    else:
        print(
            'Found existing folder. Meta-parameters will be stored at')
    print(dst_folder)
    return dst_folder


if __name__ == "__main__":
    params = {}
    params["cross_validate"] = True
    load_chem_dataset(5, params, meta_batch_size=32, num_batches=100, verbose=True)
