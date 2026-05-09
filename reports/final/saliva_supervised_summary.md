# Final supervised saliva reports

This report compares baseline runs with train-only augmentation runs for saliva datasets.

Positive ΔPR-AUC, ΔRecall and ΔF1 indicate improvement. Negative ΔBrier and ΔECE indicate better calibration.

## COVID saliva

| dataset      | model   |   delta_pr_auc |   delta_recall |   delta_f1 |   delta_specificity |   delta_brier |   delta_ece |
|:-------------|:--------|---------------:|---------------:|-----------:|--------------------:|--------------:|------------:|
| covid_saliva | lda     |         0.0062 |         0.004  |    -0.0006 |             -0.0126 |       -0.0045 |     -0.0058 |
| covid_saliva | logreg  |         0.0103 |         0.053  |     0.0264 |             -0.0272 |       -0.0088 |     -0.0013 |
| covid_saliva | plsda   |         0.0013 |         0.0033 |    -0.001  |             -0.0101 |       -0.0005 |      0.0021 |
| covid_saliva | svm_lin |         0.0108 |         0.0213 |     0.0062 |             -0.0259 |       -0.0046 |     -0.0015 |
| covid_saliva | svm_rbf |         0.005  |         0.0368 |     0.0254 |             -0.0097 |       -0.0038 |      0.0012 |

## Diabetes saliva

| dataset         | model   |   delta_pr_auc |   delta_recall |   delta_f1 |   delta_specificity |   delta_brier |   delta_ece |
|:----------------|:--------|---------------:|---------------:|-----------:|--------------------:|--------------:|------------:|
| diabetes_saliva | lda     |        -0.0026 |         0.0058 |     0.006  |              0.0063 |       -0.0048 |     -0.0012 |
| diabetes_saliva | logreg  |         0.0094 |         0.034  |     0.0133 |             -0.0131 |       -0.0129 |     -0.0043 |
| diabetes_saliva | plsda   |         0.0009 |        -0.0091 |    -0.0007 |              0.0136 |       -0.0007 |     -0.0017 |
| diabetes_saliva | svm_rbf |         0.048  |         0.0832 |     0.0595 |              0.0368 |       -0.0486 |     -0.0304 |

## Combined saliva summary

| dataset         | model   |   delta_pr_auc |   delta_recall |   delta_f1 |   delta_specificity |   delta_brier |   delta_ece |
|:----------------|:--------|---------------:|---------------:|-----------:|--------------------:|--------------:|------------:|
| covid_saliva    | lda     |         0.0062 |         0.004  |    -0.0006 |             -0.0126 |       -0.0045 |     -0.0058 |
| covid_saliva    | logreg  |         0.0103 |         0.053  |     0.0264 |             -0.0272 |       -0.0088 |     -0.0013 |
| covid_saliva    | plsda   |         0.0013 |         0.0033 |    -0.001  |             -0.0101 |       -0.0005 |      0.0021 |
| covid_saliva    | svm_lin |         0.0108 |         0.0213 |     0.0062 |             -0.0259 |       -0.0046 |     -0.0015 |
| covid_saliva    | svm_rbf |         0.005  |         0.0368 |     0.0254 |             -0.0097 |       -0.0038 |      0.0012 |
| diabetes_saliva | lda     |        -0.0026 |         0.0058 |     0.006  |              0.0063 |       -0.0048 |     -0.0012 |
| diabetes_saliva | logreg  |         0.0094 |         0.034  |     0.0133 |             -0.0131 |       -0.0129 |     -0.0043 |
| diabetes_saliva | plsda   |         0.0009 |        -0.0091 |    -0.0007 |              0.0136 |       -0.0007 |     -0.0017 |
| diabetes_saliva | svm_rbf |         0.048  |         0.0832 |     0.0595 |              0.0368 |       -0.0486 |     -0.0304 |
