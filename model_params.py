from pathlib import Path

common_params = {
    'datasource': 'drp_chem',
    'cross_validate': True,
    'verbose': False,
    'train_flag': True,
    'gpu_id': 1,
    'test_data': True,  # TODO: redundant with full_dataset?
    'meta': False,
    'full_dataset': True,
    'fine_tuning': False,
    'with_historical_data': True,   # Train models with historical data of other amines
    'with_k': False,     # Train the model with k additional amine-specific experiments
    'train_size': 10,   # k after pretrain
    'active_learning': False,
    'active_learning_iter': 10,   # x before active learning
    'stats_path': Path('./results/cv_statistics.pkl'),
    'cv_stats_overwrite': True,     # TODO: change this to false when running all models
    'save_model': False
}

knn_params = {
    'neighbors': 2,
    'config': None,
    'model_name': 'KNN'
}

svm_params = {
    'config': None,
    'model_name': 'SVM'
}

randomforest_params = {
    'config': None,
    'model_name': 'Random_Forest'
}

logisticregression_params = {
    'config': None,
    'model_name': 'Logistic_Regression'
}

decisiontree_params = {
    'config': None,
    'model_name': 'Decision_Tree'
}

gradientboosting_params = {
    'config': None,
    'model_name': 'Gradient_Boosting'
}

meta_params = {
    'k_shot': 10,
    'n_way': 2,
    'inner_lr': 1e-3,
    'meta_lr': 1e-3,
    'pred_lr': 1e-1,
    'meta_batch_size': 10,
    'Lt': 1,
    'Lv': 100,
    'num_inner_updates': 10,
    'kl_reweight': 0.0001,
    'p_dropout_base': 0.4,
    'num_epochs_save': 2,
    'num_batches': 250,
    'model_name': 'Platipus'
}

meta_train = {
    'num_epochs': 4,
    'resume_epoch': 0,
    'train_flag': True
}

meta_test = {
    'num_epochs': 0,
    'resume_epoch': 4,
    'train_flag': False
}
