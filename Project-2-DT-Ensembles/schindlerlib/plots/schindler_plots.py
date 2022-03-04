import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Jack's confusion matrix plotter
def plot_confusion_matrix(cm, class_labels):
    """Pretty prints a confusion matrix as a figure

    Args:
        cm:  A confusion matrix for example
        [[245, 5 ], 
         [ 34, 245]]
         
        class_labels: The list of class labels to be plotted on x-y axis

    Rerturns:
        Just plots the confusion matrix.
    """
    
    df_cm = pd.DataFrame(cm, index = [i for i in class_labels],
                  columns = [i for i in class_labels])
    sns.set(font_scale=1)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("Real label")
    plt.show()



def missing_plot(df: pd.DataFrame, title: str):
    missingvalues = df.isnull().sum(axis=0)/df.shape[0]*100
    missingvalues.sort_values(inplace=True, ascending=False)

    plt.figure().set_size_inches(12, 6)
    ax = sns.barplot(x= missingvalues.index, y= missingvalues)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90, ha='right')
    #ax.axhline(0.2); ax.axhline(0.4); ax.axhline(0.6); ax.axhline(0.8)
    ax.set_title('NaN data in ' + title + ' dataset as proportion of total observations')
    ax.set_xlabel('Column')
    ax.set_ylabel('%')

    return plt.show()



def make_nan_fig(df, title):
  nan_df = df.isnull().sum()/len(df)*100
  nan_df = nan_df.sort_values(ascending = False)

  nan_df_fig = plt.bar(nan_df.index, nan_df.values)
  plt.xticks(rotation='vertical', fontsize = 12)
  plt.ylabel('% NaN', fontsize = 12)
  plt.title('NaN data in ' + title + ' dataset as proportion of total observations')
  fig = plt.gcf()
  fig.set_size_inches(18, 5)
  plt.show()

  return nan_df_fig



def make_feat_importance_plot(coefs, feature_names):
  coef_abs = np.absolute(coefs)
  coef_plot = coef_abs.transpose().flatten()
  coef_plot_df = pd.DataFrame({'feature_name': feature_names, 'coef': coef_plot})
  coef_plot_df = coef_plot_df.sort_values('coef', ascending = False)

  plt.bar(x = coef_plot_df['feature_name'], height = coef_plot_df['coef'])
  plt.xticks(rotation='vertical', fontsize = 12)

  fig = plt.gcf()
  fig.set_size_inches(25, 6)

  return plt.show()



def make_roc_plot(fpr, tpr, roc_auc):
  formatter = "{0:.3f}"
  plt.plot(fpr, tpr, label='ROC curve (area =' + formatter.format(roc_auc) + ')')

  plt.plot([0, 1], [0, 1], 'k--') 
  plt.title('Receiver Operating Characteristic')
  plt.xlabel('False Positive Rate or (1 - Specifity)')
  plt.ylabel('True Positive Rate or (Sensitivity)')
  plt.legend(loc="lower right")

  fig = plt.gcf()
  fig.set_size_inches(14, 8)

  return plt.show()



def make_scree_plot(array):
  plt.plot(np.cumsum(array.explained_variance_ratio_ * 100))
  plt.xlabel("Number of components (Dimensions)")
  plt.ylabel("Explained variance (%)")
  plt.xticks(np.arange(0, 400, step=10), rotation = 'vertical')
  plt.grid(visible = True)

  fig = plt.gcf()
  fig.set_size_inches(25, 5)

  return plt.show()



def make_nan_fig(df, title):
  nan_df = df.isnull().sum()/len(df)*100
  nan_df = nan_df.sort_values(ascending = False)

  nan_df_fig = plt.bar(nan_df.index, nan_df.values)
  plt.xticks(rotation='vertical', fontsize = 12)
  plt.ylabel('% NaN', fontsize = 12)
  #plt.hlines(y=40, xmin= 0, xmax = 40, linestyle = 'dashed')
  plt.title('NaN data in ' + title + ' dataset as proportion of total observations')
  fig = plt.gcf()
  fig.set_size_inches(18, 5)
  plt.show()

  return nan_df_fig



def p_hat_plot(p: np.ndarray):
  p = pd.Series(p)

  fig = plt.figure(figsize = (5,3))
  ax = fig.gca()

  return p.hist(ax = ax)

