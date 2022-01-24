import numpy as np

def evaluate(predict_df, truth_df):
  """Compute MAE for each of the 4 variables."""
  mae_FFDAvg = np.abs(predict_df['FFDAvg'] - truth_df['FFDAvg']).mean()
  mae_FFDStd = np.abs(predict_df['FFDStd'] - truth_df['FFDStd']).mean()
  mae_TRTAvg = np.abs(predict_df['TRTAvg'] - truth_df['TRTAvg']).mean()
  mae_TRTStd = np.abs(predict_df['TRTStd'] - truth_df['TRTStd']).mean()
  mae_overall = (mae_FFDAvg + mae_FFDStd + mae_TRTAvg + mae_TRTStd) / 4

  print(f'MAE for FFDAvg: {mae_FFDAvg}')
  print(f'MAE for FFDStd: {mae_FFDStd}')
  print(f'MAE for TRTAvg: {mae_TRTAvg}')
  print(f'MAE for TRTStd: {mae_TRTStd}')
  print(f'Overall MAE: {mae_overall}')
  return mae_overall
  
