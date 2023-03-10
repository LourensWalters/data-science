import matplotlib.pyplot as plt
import itertools
import numpy as np
import seaborn as sns
from sklearn.metrics import auc
from numpy import set_printoptions


def plot_confusion_matrix(cm, classes, normalise=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Creates a plot for the specified confusion matrix object and calculates relevant accuracy measures. Note, unlike the scikit-learn version of this function, this function only has two values i.e. True or False. 'True' denotes sum of rows (True label) and 'False' no normalisation i.e. actual counts.

    Parameters:
        cm:         Dataset containing actual and expected (predicted) values.
        classes:    List of class values predicted (greater than 2 list members for multi-class classification).
        normalise:  Determines how to calculate percentages in confusion matrix: 'true': sum of rows (True label), 'false': no normalisation i.e. counts
        title:      Title of confusion matrix plot
        cmap:       Colormap recognized by matplotlib. str or matplotlib Colormap, default=’viridis’

    Output:
        none

    """

    # Add Normalisation option
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=18)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

    fp_label = 'false positive'
    fp = cm[0][1]
    fn_label = 'false negative'
    fn = cm[1][0]
    tp_label = 'true positive'
    tp = cm[1][1]
    tn_label = 'true negative'
    tn = cm[0][0]

    tpr_label = 'sensitivity'
    tpr = round(tp / (tp + fn), 3)
    tnr_label = 'specificity'
    tnr = round(tn / (tn + fp), 3)
    ppv_label = 'precision'
    ppv = round(tp / (tp + fp), 3)
    npv_label = 'npv'
    npv = round(tn / (tn + fn), 3)
    fpr_label = 'fpr'
    fpr = round(fp / (fp + tn), 3)
    fnr_label = 'fnr'
    fnr = round(fn / (tp + fn), 3)
    fdr_label = 'fdr'
    fdr = round(fp / (tp + fp), 3)

    acc_score = round((tp + tn) / (tp + fp + tn + fn), 3)

#    print('\naccuracy:\t\t\t{:.2f}  \nprecision:\t\t\t{:.2f} \nsensitivity:\t\t\t{:.2f}'.format(acc_score, ppv, tpr))
#    print('\nspecificity:\t\t\t{:.2f} \nnegative predictive value:\t{:.2f}'.format(tnr, npv))
#    print('\nfalse positive rate:\t\t{:.2f}  \nfalse negative rate:\t\t{:.2f} \nfalse discovery rate:\t\t{:.2f}'.format(fpr, fnr,
#                                                                                                            fdr))


def plot_roc_curve(fpr, tpr, title="Receiver operating characteristic (ROC) Curve"):
    """
    Creates a plot for the specified roc curve object.

    Parameters:
        fpr:        Series containing True Positive Rate.
        tpr:        Series containing False Positive Rate.
        title:      Title of ROC plot

    Output:
        none
    """

    # Visualization for ROC curve
    print('AUC: {:.2f}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10, 8))
    lw = 2
    _ = plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve');
    _ = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--');
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i / 20.0 for i in range(21)])
    plt.xticks([i / 20.0 for i in range(21)])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(loc="lower right", fontsize=18)
    _ = plt.show();


def plot_feature_importance_log(fit, features):
    """
    Creates a plot for the logistic regression feature importance object.

    Parameters:
        fit:        Fitted logistic regression model object.
        features:   List of feature names.

    Output:
        none

    """

    set_printoptions(precision=2)

    # Summarize selected features
    scores = -np.log10(fit.pvalues)

    importances = np.array(scores)
    feature_list = features
    sorted_ID = np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

#    for i, v in enumerate(reverse_importances):
#        print('Feature: %20s\tScore:\t%.2f' % (reverse_features[i], v))

    # Plot feature importance
    _ = plt.figure(figsize=[10, 10]);
    _ = plt.xticks(rotation='horizontal', fontsize=20)
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.yticks(fontsize=20)
    _ = plt.show();


def plot_feature_importance_dec(fit, features):
    """
    Creates a plot for the random forest feature importance object.

    Parameters:
        fit:        List of feature importance for random forest/ decision tree.
        features:   List of feature names.

    Output:
        none

    """

    set_printoptions(precision=2)

    # Summarize selected features
    scores = fit
    # scores /= scores.max()

    importances = np.array(scores)
    feature_list = features
    sorted_ID = np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

#    for i, v in enumerate(reverse_importances):
#        print('Feature: %20s\tScore:\t%.2f' % (reverse_features[i], v))

    # Plot feature importance
    _ = plt.figure(figsize=[10, 10]);
    _ = plt.xticks(rotation='horizontal', fontsize=20)
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.yticks(fontsize=20)
    _ = plt.show();


def plot_feature_importance(fit, features):
    """
    Creates a plot for the Select K-best feature importance object.

    Parameters:
        fit:        Fitted logistic regression model object.
        features:   List of feature names.

    Output:
        none

    """

    set_printoptions(precision=2)

    # Summarize selected features
    scores = -np.log10(fit.pvalues_)

    importances = np.array(scores)
    feature_list = features
    sorted_ID = np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

#    for i, v in enumerate(reverse_importances):
#        print('Feature: %20s\tScore:\t%.2f' % (reverse_features[i], v))

    # Plot feature importance
    _ = plt.figure(figsize=[10, 10]);
    _ = plt.xticks(rotation='horizontal', fontsize=20)
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.yticks(fontsize=20)
    _ = plt.show();


def plot_age(df, axes, single_plot=True):
    """
    Creates a continuous density plot of disease status (true or false) by age.
    """

    if single_plot:
        sns.kdeplot(data=df.loc[(df['ca_disease'] == 0), 'age'], shade=True, label='Disease False')
        sns.kdeplot(data=df.loc[(df['ca_disease'] == 1), 'age'], shade=True, label='Disease True')
        plt.xlabel('Age', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15)
        plt.show()

    else:
        facet_grid = sns.FacetGrid(df, hue='ca_disease')
        _ = facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0]);

        legend_labels = ['disease false', 'disease true']
        for t, l in zip(axes[0].get_legend().texts, legend_labels):
            t.set_text(l)
            axes[0].set(xlabel='age', ylabel='density')

        avg = df[["age", "ca_disease"]].groupby(['age'], as_index=False).mean();
        _ = sns.barplot(x='age', y='ca_disease', data=avg, ax=axes[1]);
        _ = axes[1].set(xlabel='age', ylabel='disease probability');


def plot_categorical(attribute, labels, ax_index, df, axes):
    """
    Plot categorical variables.

    Parameters:
        attribute:        Boolean value indicating whether to plot categorical (True) or continuous data (False).
        labels:           List of label values for plotting categorical variables.
        ax_index:         List of label values for plotting categorical variables.
        df:               Dataframe containing data to plot.
        axes:             Plotting axes.

    Output:
        none
    """


    sns.countplot(x=attribute, data=df, ax=axes[ax_index][0])
    sns.countplot(x='ca_disease', hue=attribute, data=df, ax=axes[ax_index][1])
    avg = df[[attribute, 'ca_disease']].groupby([attribute], as_index=False).mean()
    _ = sns.barplot(x=attribute, y='ca_disease', hue=attribute, data=avg, ax=axes[ax_index][2])

    for t, l in zip(axes[ax_index][1].get_legend().texts, labels):
        t.set_text(l)
    for t, l in zip(axes[ax_index][2].get_legend().texts, labels):
        t.set_text(l)


def plot_continuous(attribute, xlabel, ax_index, df, axes):
    """
    Plot continuous variables.

    Parameters:
        attribute:        Boolean value indicating whether to plot categorical (True) or continuous data (False).
        xlabel:           List of label values for plotting continuous variables.
        ax_index:         List of label values for plotting continuous variables.
        df:               Dataframe containing data to plot.
        axes:             Plotting axes.

    Output:
        none
    """

    if (ax_index == 5):
        return
    _ = sns.distplot(df[[attribute]], ax=axes[ax_index][0]);
    _ = axes[ax_index][0].set(xlabel=xlabel, ylabel='density');
    axes[ax_index][0].xaxis.label.set_size(24)
    axes[ax_index][0].yaxis.label.set_size(24)
    axes[ax_index][0].tick_params('y', labelsize=20);
    axes[ax_index][0].tick_params('x', labelsize=20);
    _ = sns.violinplot(x='ca_disease', y=attribute, data=df, ax=axes[ax_index][1]);
    axes[ax_index][1].xaxis.label.set_size(24)
    axes[ax_index][1].yaxis.label.set_size(24)
    axes[ax_index][1].tick_params('y', labelsize=20);
    axes[ax_index][1].tick_params('x', labelsize=20);
    plt.tight_layout()


def plot_var(isCategorical, categorical, continuous, df, axes):
    """
    Plot either categorical or continuous variables.

    Parameters:
        isCategorical:         Boolean value indicating whether to plot categorical (True) or continuous data (False).
        categorical:           List of label values for plotting categorical variables.
        continuous:            List of label values for plotting continuous variables.
        df:                    Dataframe containing data to plot.
        axes:                  Plotting axes.

    Output:
        none
    """
    if isCategorical:
        [plot_categorical(x[0], x[1], i, df, axes) for i, x in enumerate(categorical)]
    else:
        [plot_continuous(x[0], x[1], i, df, axes) for i, x in enumerate(continuous)]


def main():
    from sklearn.metrics import confusion_matrix
    """
    main function - does all the work
    """
    # parse arguments
    cnf_matrix = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])

    # generate plots
    plot_confusion_matrix(cnf_matrix, classes=[0, 1], normalise=True)


if __name__ == "__main__":
    # call main
    main()
