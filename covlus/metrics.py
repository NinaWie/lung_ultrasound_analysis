from sklearn.metrics import confusion_matrix


def sensitivity_specificity(y_true, y_pred):
    """compute sensitivity and specificity"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity, specificity
