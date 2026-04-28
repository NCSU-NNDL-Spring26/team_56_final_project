from sklearn.metrics import classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(confusion_matrix(y_true, y_pred))
