# DSLR

42 Data Science x Logistic Regression project.

## Requirements

- Python 3.9+
- See requirements.txt for Python packages

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Usage (normal)

Train (batch gradient descent):

```bash
python3 src/Logistic_Regression/logreg_train.py datasets/dataset_train.csv
```

Predict:

```bash
python3 src/Logistic_Regression/logreg_predict.py datasets/dataset_test.csv output/weights.json
```

Test accuracy:

```bash
python3 tester.py datasets/dataset_train.csv houses.csv
```

## Usage (bonus: SGD)

Train with stochastic gradient descent:

```bash
python3 src/Logistic_Regression/logreg_train.py datasets/dataset_train.csv --sgd
```

Predict and test are the same as normal:

```bash
python3 src/Logistic_Regression/logreg_predict.py datasets/dataset_test.csv output/weights.json
python3 tester.py datasets/dataset_train.csv houses.csv
```
