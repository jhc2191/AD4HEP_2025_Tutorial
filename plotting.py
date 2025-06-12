import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from scipy.stats import wasserstein_distance as emd
from sklearn.metrics import (
    auc,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))



def plot_loss(loss, path):
    """Plot loss function"""
    plt.figure(figsize=(5, 4))
    plt.semilogy(loss["train"], label="Train")
    plt.semilogy(loss["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss per batch")
    plt.grid()
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def plot_roc(y_true, y_pred_student, y_pred_teacher, save_path):
    teacher_fpr, teacher_tpr, _ = roc_curve(y_true + 0.0, -y_pred_teacher)
    student_fpr, student_tpr, _ = roc_curve(y_true + 0.0, -y_pred_student)
    teacher_auc = np.round(roc_auc_score(y_true + 0.0, -y_pred_teacher), 3)
    student_auc = np.round(roc_auc_score(y_true + 0.0, -y_pred_student), 3)

    plt.plot(
        teacher_fpr,
        teacher_tpr,
        label="Teacher, AUC={0:.3f}".format(teacher_auc),
        zorder=4,
    )
    plt.plot(
        student_fpr,
        student_tpr,
        label="Student, AUC={0:.3f}".format(student_auc),
        zorder=4,
    )
    plt.plot(
        [0, 0.2, 0.5, 0.7, 1],
        [0, 0.2, 0.5, 0.7, 1],
        linestyle="--",
        color="gray",
        zorder=4,
    )
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(zorder=1)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def plot_prc(y_true, y_pred_student, y_pred_teacher, save_path):
    teacher_precision, teacher_recall, _ = precision_recall_curve(
        y_true + 0.0, -y_pred_teacher, pos_label=1
    )
    teacher_ap = average_precision_score(y_true + 0.0, -y_pred_teacher)
    student_precision, student_recall, _ = precision_recall_curve(
        y_true + 0.0, -y_pred_student, pos_label=1
    )
    student_ap = average_precision_score(y_true + 0.0, -y_pred_student)

    plt.plot(
        teacher_recall,
        teacher_precision,
        label="Teacher, AP={0:.3f}".format(teacher_ap),
        zorder=4,
    )
    plt.plot(
        student_recall,
        student_precision,
        label="Student, AP={0:.3f}".format(student_ap),
        zorder=4,
    )
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(zorder=1)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()


def plot_dist(y_true, y_pred_student, y_pred_teacher, save_path):
    # Print score distance
    teacher_anomaly_score = np.array(y_pred_teacher[~y_true])
    student_anomaly_score = np.array(y_pred_student[~y_true])
    teacher_normal_score = np.array(y_pred_teacher[y_true])
    student_normal_score = np.array(y_pred_student[y_true])
    distance_anomaly = emd(student_anomaly_score, teacher_anomaly_score)
    distance_normal = emd(student_normal_score, teacher_normal_score)

    maximum = np.max([np.max(y_pred_teacher), np.max(y_pred_student)])
    bins = np.arange(0, maximum, maximum / 50)

    plt.hist(
        np.array(y_pred_teacher[~y_true]),
        bins=bins,
        density=True,
        edgecolor="#B71C1C",
        facecolor="none",
        histtype="step",
        label="Teacher, Anomaly",
    )
    plt.hist(
        np.array(y_pred_student[~y_true]),
        bins=bins,
        density=True,
        edgecolor="#EF5350",
        facecolor="none",
        histtype="step",
        label="Student, EMD: {0:.3f}".format(distance_anomaly),
    )
    plt.hist(
        np.array(y_pred_teacher[y_true]),
        bins=bins,
        density=True,
        edgecolor="#66BB6A",
        facecolor="none",
        histtype="step",
        label="Teacher, Normal",
    )
    plt.hist(
        np.array(y_pred_student[y_true]),
        bins=bins,
        density=True,
        edgecolor="#1B5E20",
        facecolor="none",
        histtype="step",
        label="Student, EMD: {0:.3f}".format(distance_normal),
    )
    plt.xlabel("Anomaly score")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()



def rocData(y, predict_test, labels):
    df = pd.DataFrame()

    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:, i]
        df[label + '_pred'] = predict_test[:, i]

        fpr[label], tpr[label], threshold = roc_curve(df[label], df[label + '_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
    return fpr, tpr, auc1


def plotRoc(fpr, tpr, auc, labels, linestyle, legend=True):
    for _i, label in enumerate(labels):
        plt.plot(
            tpr[label],
            fpr[label],
            label='{} tagger, AUC = {:.1f}%'.format(label.replace('j_', ''), auc[label] * 100.0),
            linestyle=linestyle,
        )
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001, 1)
    plt.grid(True)
    if legend:
        plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90, 'hls4ml', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)


def makeRoc(y, predict_test, labels, linestyle='-', legend=True):
    if 'j_index' in labels:
        labels.remove('j_index')

    fpr, tpr, auc1 = rocData(y, predict_test, labels)
    plotRoc(fpr, tpr, auc1, labels, linestyle, legend=legend)
    return predict_test