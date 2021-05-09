from IPython.display import clear_output
import requests
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer, KNNImputer


def run(df):

    # Clean data
    df = clean_df_for_demo(df)
    hobbies = df.columns
    X_train = df

    # Fit model
    print('training model...')
    #X_train = normalize_scale(X_train)
    #X_train = normalize(X_train)
    imp = KNNImputer(n_neighbors=5)
    imp.fit(X_train)

    # Query, predict
    done = False
    while not done:
        likes_dislikes = ask_likes_dislikes(hobbies)
        user_df = get_user(hobbies, likes_dislikes)
        preds_df = predict(imp, user_df)
        output_preds(preds_df, likes_dislikes)
        doagain = input('\nAgain? ')
        if 'n' in doagain or 'N' in doagain:
            done = True

def clean_df_for_demo(df):
    col_cutoff = df.columns.tolist().index('Ageing')
    drop_cols = df.columns[col_cutoff-1:]
    df.drop(drop_cols, axis=1, inplace=True)
    df.rename({col:col.replace(',','/') for col in df.columns}, axis=1, inplace=True)
    imp = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(data=imp.fit_transform(df), columns=df.columns)
    return df

def normalize_scale(X):
    sc = MinMaxScaler((1,5))
    X = sc.fit_transform(normalize(X))
    return X

def ask_likes_dislikes(choices):
  clear_output()
  done = False
  while not done:
    likes = input(f'Enter a few things you like to do from the following list, separated by commas. \n\n {choices}\n')
    dislikes = input(f'\nNow enter a few things you would dislike to do from that list.')
    likes = [item.strip() for item in likes.split(',')]
    dislikes = [item.strip() for item in dislikes.split(',')]
    missing = [item for item in dislikes if item not in choices] + [item for item in likes if item not in choices]
    if missing:
      clear_output()
      print(f"**Couldn't find {missing}. This is a janky demo: please spell things exactly as provided, separated by commas!")
    else:
      done = True
  clear_output()
  return likes, dislikes

def get_user(hobbies, likes_dislikes):
  likes, dislikes = likes_dislikes
  user_df = pd.DataFrame(np.nan, index=[0], columns=hobbies)
  for item in likes:
    user_df.at[0, item] = 5
  for item in dislikes:
    user_df.at[0, item] = 1
  return user_df

def predict(model, user_df):
  preds = model.transform(user_df)
  res = pd.DataFrame(data=np.round_(preds, 1), columns=user_df.columns)
  #res = pd.DataFrame({col:pred for col, pred in zip(user_df.columns, preds)})
#  res = sorted(list(zip(user_df.columns, preds)), key=lambda x:x[1], reverse=True)
  return res

def output_preds(preds_df, likes_dislikes):
  #hobbies_scores = predict(model, user_df)
  likes, dislikes = likes_dislikes
  hobbies_scores = sorted(list(zip(preds_df.columns, preds_df.iloc[0,:])), key=lambda x:x[1], reverse=True)
  hobbies_scores = [(hobby, score) for hobby, score in hobbies_scores if hobby not in likes and hobby not in dislikes]
  print(f'You liked: {likes}.')
  print(f'You disliked {dislikes}\n')
  print("Predicted hobbies, in decending order of how much you like 'em':\n")
  for hobby, score in hobbies_scores:
    print(hobby, round(score, 1))
