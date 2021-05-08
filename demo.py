def get_hates_likes(choices):
  clear_output()
  done = False
  while not done:
    hates = input(f'Enter a few things you HATE from the following list, separated by commas. \n\n {X_train.columns}\n')
    likes = input(f'\nNow enter a few things you LIKE ')
    hates = [item.strip() for item in hates.split(',')]
    likes = [item.strip() for item in likes.split(',')]
    missing = [item for item in hates if item not in choices] + [item for item in likes if item not in choices]
    if missing:
      clear_output()
      print(f"**Couldn't find {missing}. This is a janky demo: please spell things exactly as provided, separated by commas!")
    else:
      done = True
  clear_output()
  return hates, likes

def get_user(hobbies):
  hates, likes = get_hates_likes(choices=hobbies)
  user_df = pd.DataFrame(np.nan, index=[0], columns=hobbies)
  for item in hates:
    user_df.at[0, item] = 5
  for item in likes:
    user_df.at[0, item] = 5
  return user_df

def predict(model, user_df):
  preds = model.transform(user_df)[0]
  res = sorted(list(zip(user.columns, preds)), key=lambda x:x[1], reverse=True)
  return res

def predict_output(model, user_df):
  hobbies_scores = predict(model, user_df)
  print("Predicted hobbies, in decending order of how much you like 'em':\n")
  for hobby, score in hobbies_scores:
    print(hobby, round(score, 1))
