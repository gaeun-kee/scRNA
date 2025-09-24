import numpy as np
from math import sqrt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

# -----------------------------
# AUROC with 95% CI
# -----------------------------
def roc_auc_ci(y_true, y_score, positive=1):
    """
    ROC-AUC와 95% CI 계산
    """
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1 * N2))
    lower = max(0, AUC - 1.96 * SE_AUC)
    upper = min(1, AUC + 1.96 * SE_AUC)

    print(f"AUROC (95% CI): {AUC:.3f} [{lower:.3f}, {upper:.3f}]")
    return AUC, (lower, upper)


# -----------------------------
# AUPRC with 95% CI
# -----------------------------
def roc_auprc_ci(y_true, y_score, positive=1):
    """
    AUPRC와 95% CI 계산
    """
    AUPRC = average_precision_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUPRC / (2 - AUPRC)
    Q2 = 2 * AUPRC**2 / (1 + AUPRC)
    SE_AUPRC = sqrt((AUPRC*(1 - AUPRC) + (N1 - 1)*(Q1 - AUPRC**2) + (N2 - 1)*(Q2 - AUPRC**2)) / (N1 * N2))
    lower = max(0, AUPRC - 1.96 * SE_AUPRC)
    upper = min(1, AUPRC + 1.96 * SE_AUPRC)

    print(f"AUPRC (95% CI): {AUPRC:.3f} [{lower:.3f}, {upper:.3f}]")
    return AUPRC, (lower, upper)


# -----------------------------
# 공통 평가 함수
# -----------------------------
def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    모델 평가: ROC-AUC, AUPRC, confusion matrix, classification report
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    auroc, auroc_ci = roc_auc_ci(y_true, y_pred_proba)
    auprc, auprc_ci = roc_auprc_ci(y_true, y_pred_proba)

    report = classification_report(y_true, y_pred, labels=[0, 1])
    confusion = confusion_matrix(y_true, y_pred)

    print(report)
    print("Confusion Matrix:\n", confusion)

    return {
        "auroc": (auroc, auroc_ci),
        "auprc": (auprc, auprc_ci),
        "report": report,
        "confusion_matrix": confusion
    }


# -----------------------------
# ROC & PR Curve Plot
# -----------------------------
def plot_curves(y_true, y_pred_proba):
    """
    ROC Curve와 Precision-Recall Curve 시각화
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax[0])
    ax[0].set_title("ROC Curve")

    PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, ax=ax[1])
    ax[1].set_title("Precision-Recall Curve")

    plt.show()

