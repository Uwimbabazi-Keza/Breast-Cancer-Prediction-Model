from sklearn.metrics import confusion_matrix, f1_score, classification_report

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict_classes(x_test)
  
    c = confusion_matrix(y_test, y_pred)
    specificity = c[0, 0] / (c[0, 0] + c[0, 1])
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
    
    return c, specificity, f1, report
