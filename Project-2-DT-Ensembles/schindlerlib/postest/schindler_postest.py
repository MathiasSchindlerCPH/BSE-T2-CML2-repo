import pandas as pd
import numpy as np



## Correction factor
def reweight(pi,q1=0.5,r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w



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

