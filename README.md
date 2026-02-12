# Cleaner version of the code for the soft vertical equity constrained mass appraisal models.

## Installation of the required packages

```
pip install -r requirements.txt
```

## Running the code

You can edit the parameters and basic settings in the `main.py` file. Parameters for LightGBM are defined in `model_parameters.yaml`. Then, to run the code, simply execute:
```
python main.py
```
This will train four models: 
- Linear Regression (LR)
- LightGBM (LGBM)
- LightGBM with a soft vertical equity constraint on $Cov(ratios, prices)$ (LGBSmoothPenalty)
- LightGBM with a soft vertical equity constraint on an upper bound of $Cov(ratios, prices)$ using triangle inequality (LGBCovPenalty)

 The evaluation is done in different Rolling-Origin fols in this file, and its printing the results to the console. You can also modify the code to save the results to a file or to perform additional analyses as needed.

