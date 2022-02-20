import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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

