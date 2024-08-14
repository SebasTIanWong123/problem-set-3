'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''
    micro_tp = micro_fp = micro_fn = 0

    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    for genre in genre_list:
        tp = genre_tp_counts[genre]
        fp = genre_fp_counts[genre]
        fn = genre_true_counts[genre] - tp
    
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)
        
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    
    # These are the micrometrics.
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list
   

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    from sklearn.metrics import precision_recall_fscore_support

    #preparing the true and predicted labels 
    pred_rows = []
    true_rows = []

    for index, row in model_pred_df.itterws():
        actual_genres = eval(row['actual genres'])
        predicted_genre = row['predicted']
        pred_rows.append(predicted_genre)
        true_rows.extend(actual_genres)

    true_rows = [genre_list.index(genre) for genre in true_rows]
    pred_rows = [genre_list.index(genre) for genre in pred_rows]

    macro_prec, macro_rec, macro_f1, _=precision_recall_fscore_support(true_rows, pred_rows, average='macro', zero_division=0)
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(true_rows, pred_rows, average='micro', zero_division=0)

    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1