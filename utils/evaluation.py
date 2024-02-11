import numpy as np
from sklearn.metrics import accuracy_score


def calculate_SLC_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true_bool = y_true.argmax(-1)
    else:
        y_true_bool = y_true

    if len(y_pred.shape) > 1:
        y_pred_bool = y_pred.argmax(-1)
    else:
        y_pred_bool = y_pred


    res = {'ACC': accuracy_score(y_true_bool, y_pred_bool)}

    return res

def display_results(results, metrics=['ACC'], logger=None):
    if logger is not None:
        logger.critical('{0:>10}'.format("Label") + ' '.join(['%10s'] * len(metrics)) % tuple([m for m in metrics]))
        logger.critical('{0:>10}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))
    else:
        print('{0:>20}'.format("Label") + ' '.join(['%10s'] * len(metrics)) % tuple([m for m in metrics]))
        print('{0:>20}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))

    return [results[m] for m in metrics]