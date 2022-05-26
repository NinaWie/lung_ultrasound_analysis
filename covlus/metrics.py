from sklearn.metrics import confusion_matrix, accuracy_score


def sensitivity_specificity(y_true, y_pred, title):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    acc = accuracy_score(y_true, y_pred)
    print(
        f"Accuracy {title}", round(acc, 2), "sensitvity",
        round(sensitivity, 2), "specificity", round(specificity, 2)
    )
    return sensitivity, specificity