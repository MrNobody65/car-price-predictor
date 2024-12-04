# car-price-predictor

## Installation
* Create a python environment (using `conda` or `python -m`).
* Install necessary libraries:
```sh
pip install -r requirements.txt
```

## Model
Evaluating different models based on R-squared to get the best model: **Decision Tree**.

## Evaluation metrics
* **MAE**: Measures the average size of errors, showing how far predictions are from actual values on average.
```math
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|
```
* **MSE**: Calculates the average of squared errors, penalizing large mistakes more than small ones.
```math
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
```
* **RMSE**: Shows the average error size, in the same units as the target, with extra focus on big errors.
```math
\text{RMSE} = \sqrt{\text{MSE}}
```
* **MAPE**: Shows the average error as a percentage of actual values, making it easy to compare across datasets.
```math
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{\hat{y}_i - y_i}{y_i} \right| \times 100
```
* **R-squared**: Explains how much of the target's variation is captured by the model, with higher values meaning better fit.
```math
R^2 = 1 - \frac{\sum_{i=1}^{n} (\hat{y}_i - y_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
```
**Where:**
* $n$: Number of observations.
* $\hat{y}_i$: Predicted value for observation $i$.
* $y_i$: Actual value for observation $i$.
* $\bar{y}$: Mean of actual values $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$.

| Regression Model        | MAE             | MSE             | RMSE            | MAPE            | R-square            |
|-------------------------|-----------------|-----------------|-----------------|-----------------|---------------|
| Linear Regression       | 1.2326e+17      | 8.7462e+38      | 2.9574e+19      | 9.4814e+10      | -3.3609e+21   |
| K-Neighbor Regression   | 2.1484e+7       | 8.7898e+15      | 9.3754e+7       | 11.0963         | 0.9662        |
| Decision Tree           | 1.5396e+7       | 5.2015e+15      | 7.2122e+7       | 6.9632          | 0.9800        |
| Random Forest           | 1.6488e+7       | 5.3190e+15      | 7.2932e+7       | 8.1182          | 0.9796        |
| Gradient Boost Regressor| 1.6696e+8       | 9.6327e+16      | 3.1037e+8       | 76.7775         | 0.6298        |
| Ada Boost Regressor     | 9.5352e+8       | 2.2747e+18      | 1.5082e+9       | 373.8944        | -7.7408       |
| Cat Boost Regressor     | 7.1621e+7       | 2.0027e+16      | 1.4152e+8       | 42.5228         | 0.9230        |
| XGBoost Regressor       | 8.3035e+7       | 3.4893e+16      | 1.8679e+8       | 46.1225         | 0.8659        |

In this work, I choose R-squared because the values of this metric are easy for developers to observe (not too large) and effectively interpret the capability of the models with highly distributed data like this.

Observing other metrics also indicates that **Decision Tree** is the best model.

## RESTful API
I use **FastAPI** to build the API. To run the app:
```sh
uvicorn app:app --reload
```
Go to <u>127:0:0:1:8000/docs</u> to test the API.
