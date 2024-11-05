import os
import random
import json
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from imblearn.over_sampling import SMOTE


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch baseline models Training')

    # Required parameters
    parser.add_argument("--data_folder", default='./data', type=str,
                        help="Path to data folder.")
    parser.add_argument("--runs", default=10, type=int,
                        help="Number of runs for training.")
    args = parser.parse_args()
    args.model_path = "./save/binary_svm_lr_models"
    args.model_filename = 'binary_svm_lr_model'
    args.save_folder = os.path.join(args.model_path, args.model_filename)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    return args


if __name__ == "__main__":

    __args = get_args()

    n_runs = __args.runs
    seeds = np.random.randint(10000, 90000, size=n_runs)

    results = []
    raw_results = {'svc': [], 'lr': []}

    class_dict = {'UNCLEAR': 0, 'HIGH': 0, 'LOW': 1}
    # load data
    train_file = os.path.join(__args.data_folder, 'rob_dataset_train.json')
    val_file = os.path.join(__args.data_folder, 'rob_dataset_test.json')
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(val_file, 'r') as f:
        val_data = json.load(f)

    # merge train and val data
    train_data.extend(val_data)
    df = pd.DataFrame(train_data)

    # get data
    # train_questions = [train_data[i]['question']+' '+train_data[i]['answers']['text'][0] for i in range(len(train_data))]
    # train_labels = [train_data[i]['label'] for i in range(len(train_data))]
    #
    # val_questions = [val_data[i]['question']+' '+val_data[i]['answers']['text'][0] for i in range(len(val_data))]
    # val_labels = [val_data[i]['label'] for i in range(len(val_data))]

    # one hot encoding train labels
    # train_labels = np.array([class_dict[i] for i in train_labels])
    # val_labels = np.array([class_dict[i] for i in val_labels])


    best_f1_svc = None
    best_preds_svc = None
    best_f1_lr = None
    best_preds_lr = None

    for seed in seeds:
        random.seed(int(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_data, val_data = train_test_split(df, test_size=0.2, stratify=df['question'], random_state=seed)

        train_questions = [row['question'] + ' ' + row['answers']['text'][0] for _, row in train_data.iterrows()]
        train_labels = [class_dict[row['label']] for _, row in train_data.iterrows()]

        ids = [row['id'] for _, row in val_data.iterrows()]
        val_questions = [row['question'] + ' ' + row['answers']['text'][0] for _, row in val_data.iterrows()]
        val_labels = [class_dict[row['label']] for _, row in val_data.iterrows()]

        # idx = np.random.permutation(len(train_questions))
        # train_questions = [train_questions[i] for i in idx]
        # train_labels = train_labels[idx]

        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000, binary=True)

        X = vectorizer.fit_transform(train_questions)
        X_val = vectorizer.transform(val_questions)

        # train
        clf_svc = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto',
                                                                     random_state=seed,
                                                                     class_weight={0: 1, 1: 0.482},
                                                                     probability=True))

        clf_svc.fit(X, train_labels)
        # predict
        y_pred_svc = clf_svc.predict(X_val)
        auc_svc = roc_auc_score(val_labels, clf_svc.predict_proba(X_val)[:, 1])
        accuracy_svc = accuracy_score(val_labels, y_pred_svc)
        f1_svc = f1_score(val_labels, y_pred_svc, zero_division=1)
        precision_svc = precision_score(val_labels, y_pred_svc, zero_division=1)
        recall_svc = recall_score(val_labels, y_pred_svc, zero_division=1)

        if best_f1_svc is None:
            best_f1_svc = f1_svc
            best_preds_svc = y_pred_svc
        elif f1_svc > best_f1_svc:
            best_f1_svc = f1_svc
            best_preds_svc = y_pred_svc

        _run_svc_ = {"Model": "SVC", "AUC": auc_svc, "Accuracy": accuracy_svc, "F1": f1_svc, "Precision": precision_svc, "Recall": recall_svc}

        results.append(_run_svc_)
        raw_results['svc'].append({
            'id': ids,
            'seed': [seed.astype(str)] * len(val_labels),
            'preds': y_pred_svc.tolist(),
            'labels': val_labels,
            'probs': clf_svc.predict_proba(X_val)[:, 1].tolist()
        })

        clf_lr = LogisticRegression(random_state=seed,
                                    class_weight={0: 1, 1: 0.482},
                                    solver='liblinear')
        clf_lr.fit(X, train_labels)
        y_pred_lr = clf_lr.predict(X_val)
        auc_lr = roc_auc_score(val_labels, clf_lr.predict_proba(X_val)[:, 1])
        accuracy_lr = accuracy_score(val_labels, y_pred_lr)
        f1_lr = f1_score(val_labels, y_pred_lr, zero_division=1)
        precision_lr = precision_score(val_labels, y_pred_lr, zero_division=1)
        recall_lr = recall_score(val_labels, y_pred_lr, zero_division=1)

        if best_f1_lr is None:
            best_f1_lr = f1_lr
            best_preds_lr = y_pred_lr
        elif f1_lr > best_f1_lr:
            best_f1_lr = f1_lr
            best_preds_lr = y_pred_lr

        _run_lr_ = {"Model": "Logistic Regression", "AUC": auc_lr, "Accuracy": accuracy_lr, "F1": f1_lr, "Precision": precision_lr, "Recall": recall_lr}
        results.append(_run_lr_)
        raw_results['lr'].append({
            'id': ids,
            'seed': [seed.astype(str)] * len(val_labels),
            'preds': y_pred_lr.tolist(),
            'labels': val_labels,
            'probs': clf_lr.predict_proba(X_val)[:, 1].tolist()
        })

    with open(os.path.join(__args.save_folder, 'raw_results.json'), 'w') as f:
        json.dump(raw_results, f)

    best_predictions = {
        "SVC": best_preds_svc.tolist(),
        "SVC_probs": clf_svc.predict_proba(X_val)[:, 1].tolist(),
        "Logistic Regression": best_preds_lr.tolist(),
        "Logistic Regression_probs": clf_lr.predict_proba(X_val)[:, 1].tolist(),
        "Val Labels": val_labels
    }
    with open(os.path.join(__args.save_folder, 'best_predictions.json'), 'w') as f:
        json.dump(best_predictions, f)

    with open(os.path.join(__args.save_folder, 'svc_lr_results.json'), 'w') as f:
        json.dump(results, f)
    svc_lr_results = pd.DataFrame(results)
    svc_lr_results.to_csv(os.path.join(__args.save_folder, 'svc_lr_results.csv'), index=False)

    #plot results for best predictions
    print(classification_report(val_labels, best_preds_svc))
    # confusion matrix
    print(confusion_matrix(val_labels, best_preds_svc))

    fpr1, tpr1, _ = roc_curve(val_labels, clf_svc.predict_proba(X_val)[:, 1].tolist())
    roc1 = roc_auc_score(val_labels, clf_svc.predict_proba(X_val)[:, 1].tolist())

    print(classification_report(val_labels, best_preds_lr))
    # confusion matrix
    print(confusion_matrix(val_labels, best_preds_lr))

    fpr2, tpr2, _ = roc_curve(val_labels, clf_lr.predict_proba(X_val)[:, 1].tolist())
    roc2 = roc_auc_score(val_labels, clf_lr.predict_proba(X_val)[:, 1].tolist())

    plt.figure()
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='SVM - ROC curve (area = %0.2f)' % roc1)
    plt.plot(fpr2, tpr2, color='blue', lw=2, label='Logistic Regression - ROC curve (area = %0.2f)' % roc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.show()

    print("Done")
