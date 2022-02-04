import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta


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


## Correction factor
def reweight(pi,q1=0.5,r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w


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
  plt.show()
  
  
def make_scree_plot(array):
  plt.plot(np.cumsum(array.explained_variance_ratio_ * 100))
  plt.xlabel("Number of components (Dimensions)")
  plt.ylabel("Explained variance (%)")
  plt.xticks(np.arange(0, 105, step=5))
  plt.grid(visible = True)

  fig = plt.gcf()
  fig.set_size_inches(18, 5)

  return plt.show()


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


def reweight_proba_multi(pi, q, r):
  pi_rw = pi.copy()
  tot = np.dot(pi, (np.array(q)/r))
  for i in range(len(q)):
    pi_rw[:,i] = (pi[:,i] * q[i] / r) / tot
  return pi_rw


def make_multi_point_pred(array):
  df = pd.DataFrame(array)

  N_rows = len(df.index) 
  point_pred_lst = []

  for i in range(N_rows):
    temp = df.iloc[i,:].idxmax(axis = 0)
    temp += 1
    point_pred_lst.append(temp)

  return pd.Series(point_pred_lst, index = df.index)


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


def construct_age_admit(df: pd.DataFrame):
  # Add empty column for admittime and date of birth
  df['real_admittime'] = np.nan
  df['real_dob'] = np.nan

  # Realistic admittime formatting
  for i in range(len(df)):
    admit_datetime = datetime.strptime(df['ADMITTIME'].iloc[i], '%Y-%m-%d %H:%M:%S')
    delta = timedelta(days = df['Diff'].iloc[i])
    real_admit = admit_datetime + delta
    df['real_admittime'].iloc[i] = real_admit

  # Realistic date of birth formatting
  for i in range(len(df)):
    dob_datetime = datetime.strptime(df['DOB'].iloc[i], '%Y-%m-%d %H:%M:%S')
    delta = timedelta(days = df['Diff'].iloc[i])
    real_dob = dob_datetime + delta
    df['real_dob'].iloc[i] = real_dob
  
  # Drop previous datetime variables
  df = df.drop(['ADMITTIME', 'DOB'], axis = 1)

  # Add empty column for new variable age at admission
  df['age_at_admin'] = np.nan

  # Age-at-Admission is the difference between date of admission and date of birth:
  for i in range(len(df)):
    df['age_at_admin'].iloc[i] = df['real_admittime'].iloc[i] - df['real_dob'].iloc[i]
    df['age_at_admin'].iloc[i] = df['age_at_admin'].iloc[i].days / 365.25

  # Drop temp admintime and dob variables
  df = df.drop(['real_admittime', 'real_dob', 'Diff'], axis = 1)
  
  return df


def p_hat_plot(p: np.ndarray):
  p = pd.Series(p)

  fig = plt.figure(figsize = (5,3))
  ax = fig.gca()

  return p.hist(ax = ax)