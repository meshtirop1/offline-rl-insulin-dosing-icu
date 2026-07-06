# Behavior-cloning baseline

train=239896 val=50314 test=50397 rows

## Route classification (majority-class balanced weighting)

Majority-class baseline (always predict `NONE`), on test:

```
                 precision    recall  f1-score   support

BOLUS_INYECTION       0.00      0.00      0.00     11612
     BOLUS_PUSH       0.00      0.00      0.00       553
       INFUSION       0.00      0.00      0.00      3147
           NONE       0.70      1.00      0.82     35085

       accuracy                           0.70     50397
      macro avg       0.17      0.25      0.21     50397
   weighted avg       0.48      0.70      0.57     50397
```

Trained HistGradientBoostingClassifier, on val:

```
                 precision    recall  f1-score   support

BOLUS_INYECTION       0.52      0.71      0.60     12390
     BOLUS_PUSH       0.07      0.42      0.12       519
       INFUSION       0.52      0.78      0.63      2721
           NONE       0.90      0.69      0.78     34684

       accuracy                           0.69     50314
      macro avg       0.51      0.65      0.53     50314
   weighted avg       0.78      0.69      0.72     50314
```

Trained HistGradientBoostingClassifier, on test:

```
                 precision    recall  f1-score   support

BOLUS_INYECTION       0.52      0.72      0.60     11612
     BOLUS_PUSH       0.08      0.41      0.13       553
       INFUSION       0.53      0.78      0.63      3147
           NONE       0.91      0.69      0.79     35085

       accuracy                           0.70     50397
      macro avg       0.51      0.65      0.54     50397
   weighted avg       0.79      0.70      0.73     50397
```

## Insulin-type classification (rows with route != NONE only)

n_dosed: train=74611 val=15630 test=15312

Trained HistGradientBoostingClassifier, on test:

```
              precision    recall  f1-score   support

Intermediate       0.12      0.44      0.19       554
        Long       0.19      0.51      0.27      1039
       Short       0.97      0.73      0.83     13719

    accuracy                           0.71     15312
   macro avg       0.42      0.56      0.43     15312
weighted avg       0.88      0.71      0.77     15312
```

## Dose regression (units; trained on log1p(dose), rows with route != NONE)

val:  MAE=5.89  RMSE=11.67

test: MAE=5.94  RMSE=11.57

naive median-dose baseline, test: MAE=7.70  RMSE=14.59

## Feature list used for all three models

['last_glc', 'glc_slope', 'n_glc_readings_in_bin', 'hours_since_last_glc', 'hours_since_last_dose', 'dose_last_24h', 'los_icu_days', 'first_icu_stay', 'bin_index']
