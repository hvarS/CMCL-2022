import pandas as pd
import glob


FEATURES_NAMES = ['FFDAvg', 'FFDStd', 'TRTAvg', 'TRTStd']

all_predictions = [pd.read_csv(f) for f in glob.glob('predictions*.csv')]
for pred in all_predictions:
    pred.drop(columns = ['langText'],inplace = True)

# all_predictions = pd.concat(all_predictions)
# mean_df = all_predictions.groupby(['language','sentence_id', 'word_id', 'word']).mean().reset_index()

# df_num = mean_df[FEATURES_NAMES]
# df_num[df_num < 0] = 0
# df_num[df_num > 100] = 100

all_predictions[0].to_csv('pred_result.csv', index=False)