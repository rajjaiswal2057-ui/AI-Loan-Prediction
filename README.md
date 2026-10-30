# AI Loan Prediction

Loan approval prediction and financial risk assessment using machine learning.

The classifier is Logistic Regression trained with batch gradient descent, written from scratch in NumPy.
No scikit-learn model, scaler or metric is used. The confusion matrix, ROC curve and AUC are also coded by hand.

## Live demo
Coming soon.

## How to run
```bash
pip install -r requirements.txt
pip install -r requirements-train.txt   # only needed to re-run the two scripts below (adds matplotlib)
python analysis/eda.py        # dataset audit, writes reports/eda.json and figures
python training/train.py      # trains the model, writes model/saved_model/model.json and reports/results.json
python app/app.py             # then open http://127.0.0.1:5000
python tests/test_project.py  # sanity tests (model, metrics, API validation)
```
The trained model is already included, so `python app/app.py` works straight away.

## Folders
| Folder | What it contains |
|---|---|
| `ml/logistic_regression.py` | sigmoid, binary cross-entropy, gradient descent, L2 |
| `ml/preprocessing.py` | cleaning, encoding, standard scaler, stratified split |
| `ml/metrics.py` | confusion matrix, precision, recall, F1, ROC, AUC |
| `ml/risk_engine.py` | prediction, risk score, explanations, what-if |
| `training/train.py` | training, L2 study, feature ablation, fit diagnosis |
| `analysis/eda.py` | exploratory data analysis |
| `tests/` | sanity tests for the model, metrics and API input checks |
| `app/` | Flask backend and the HTML/CSS/JS front end (Chart.js included, works offline) |
| `reports/` | figures, results.json, eda.json |

## Results (test set, 642 rows)
Accuracy 92.37%, precision 93.32%, recall 94.49%, F1 93.90%, ROC-AUC 0.9761.
Confusion matrix: TN 216, FP 27, FN 22, TP 377. Train accuracy 92.14% vs test 92.37%, so the model is well fitted.
Settings: learning rate 0.1, 2000 epochs, L2 lambda 0.001, threshold 0.5.

## Risk score
`Risk = 100 x [0.45(1-P) + 0.25(1-Credit) + 0.15*Burden + 0.10(1-Assets) + 0.05(1-Income)]`
Bands: 0-30 Low, 31-60 Medium, 61-100 High. This is a score made for this project, not an official credit score.
The details are in `ml/risk_engine.py` and the report.

## Things to keep in mind
* In this dataset approval follows the CIBIL score almost like a cut-off (about 550). The other features add very little.
* Two ratio features (loan-to-income, assets-to-loan) were tried and gave no gain, so the model uses the 11 raw features.
* Loan-to-income only ranges from 1.5x to 4.0x in the data. The app warns when an input is outside the training range.
* This is a learning project and must not be used for real lending decisions.
