
# Model Fit in Linear Regression - Lab

## Introduction
In this lab, you'll learn how to evaluate your model results and you'll learn how to select the appropriate features using stepwise selection.

## Objectives
You will be able to:
* Use stepwise selection methods to determine the most important features for a model
* Use recursive feature elimination to determine the most important features for a model

## The Ames Housing Data once more


```python
import pandas as pd
import numpy as np

ames = pd.read_csv('ames.csv')

continuous = ['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
categoricals = ['BldgType', 'KitchenQual', 'SaleType', 'MSZoning', 'Street', 'Neighborhood']

ames_cont = ames[continuous]

# log features
log_names = [f'{column}_log' for column in ames_cont.columns]

ames_log = np.log(ames_cont)
ames_log.columns = log_names

# normalize (subract mean and divide by std)

def normalize(feature):
    return (feature - feature.mean()) / feature.std()

ames_log_norm = ames_log.apply(normalize)

# one hot encode categoricals
ames_ohe = pd.get_dummies(ames[categoricals], prefix=categoricals, drop_first=True)

preprocessed = pd.concat([ames_log_norm, ames_ohe], axis=1)
```

## Perform stepwise selection

The function for stepwise selection is copied below. Use this provided function on your preprocessed Ames Housing data.


```python
import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ 
    Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
```


```python
X = preprocessed.drop('SalePrice_log', axis=1)
y = preprocessed['SalePrice_log']

result = stepwise_selection(X, y, verbose = True)
print('resulting features:')
print(result)
```

    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)


    Add  GrLivArea_log                  with p-value 1.59847e-243
    Add  KitchenQual_TA                 with p-value 1.56401e-67
    Add  1stFlrSF_log                   with p-value 7.00069e-48
    Add  KitchenQual_Fa                 with p-value 1.70471e-37
    Add  Neighborhood_OldTown           with p-value 3.20105e-23
    Add  KitchenQual_Gd                 with p-value 4.12635e-21
    Add  Neighborhood_Edwards           with p-value 9.05184e-17
    Add  Neighborhood_IDOTRR            with p-value 1.10068e-18
    Add  LotArea_log                    with p-value 1.71728e-13
    Add  Neighborhood_NridgHt           with p-value 7.05633e-12
    Add  BldgType_Duplex                with p-value 4.30647e-11
    Add  Neighborhood_NAmes             with p-value 2.25803e-09
    Add  Neighborhood_SWISU             with p-value 5.40743e-09
    Add  Neighborhood_BrkSide           with p-value 8.79638e-10
    Add  Neighborhood_Sawyer            with p-value 6.92011e-09
    Add  Neighborhood_NoRidge           with p-value 5.87105e-08
    Add  Neighborhood_Somerst           with p-value 3.00722e-08
    Add  Neighborhood_StoneBr           with p-value 6.58621e-10
    Add  Neighborhood_MeadowV           with p-value 2.26069e-05
    Add  SaleType_New                   with p-value 0.000485363
    Add  SaleType_WD                    with p-value 0.00253157
    Add  Neighborhood_BrDale            with p-value 0.00374541
    Add  MSZoning_RM                    with p-value 8.29694e-05
    Add  MSZoning_RL                    with p-value 0.00170469
    Add  MSZoning_FV                    with p-value 0.00114668
    Add  MSZoning_RH                    with p-value 3.95797e-05
    Add  Neighborhood_NWAmes            with p-value 0.00346099
    Drop SaleType_WD                    with p-value 0.0554448


    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:46: FutureWarning: 
    The current behaviour of 'Series.argmax' is deprecated, use 'idxmax'
    instead.
    The behavior of 'argmax' will be corrected to return the positional
    maximum in the future. For now, use 'series.values.argmax' or
    'np.argmax(np.array(values))' to get the position of the maximum
    row.


    Add  Neighborhood_Mitchel           with p-value 0.00994666
    Drop Neighborhood_Somerst           with p-value 0.0500753
    Add  Neighborhood_SawyerW           with p-value 0.00427685
    resulting features:
    ['GrLivArea_log', 'KitchenQual_TA', '1stFlrSF_log', 'KitchenQual_Fa', 'Neighborhood_OldTown', 'KitchenQual_Gd', 'Neighborhood_Edwards', 'Neighborhood_IDOTRR', 'LotArea_log', 'Neighborhood_NridgHt', 'BldgType_Duplex', 'Neighborhood_NAmes', 'Neighborhood_SWISU', 'Neighborhood_BrkSide', 'Neighborhood_Sawyer', 'Neighborhood_NoRidge', 'Neighborhood_StoneBr', 'Neighborhood_MeadowV', 'SaleType_New', 'Neighborhood_BrDale', 'MSZoning_RM', 'MSZoning_RL', 'MSZoning_FV', 'MSZoning_RH', 'Neighborhood_NWAmes', 'Neighborhood_Mitchel', 'Neighborhood_SawyerW']


### Build the final model again in Statsmodels


```python
import statsmodels.api as sm
X_fin = X[result]
X_with_intercept = sm.add_constant(X_fin)
model = sm.OLS(y,X_with_intercept).fit()
model.summary()
```

    /Users/lore.dirick/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>SalePrice_log</td>  <th>  R-squared:         </th> <td>   0.835</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.832</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   269.0</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 16 Apr 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>13:16:34</td>     <th>  Log-Likelihood:    </th> <td> -754.40</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1460</td>      <th>  AIC:               </th> <td>   1565.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1432</td>      <th>  BIC:               </th> <td>   1713.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    27</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                <td>   -0.2174</td> <td>    0.164</td> <td>   -1.323</td> <td> 0.186</td> <td>   -0.540</td> <td>    0.105</td>
</tr>
<tr>
  <th>GrLivArea_log</th>        <td>    0.3694</td> <td>    0.015</td> <td>   24.477</td> <td> 0.000</td> <td>    0.340</td> <td>    0.399</td>
</tr>
<tr>
  <th>KitchenQual_TA</th>       <td>   -0.7020</td> <td>    0.055</td> <td>  -12.859</td> <td> 0.000</td> <td>   -0.809</td> <td>   -0.595</td>
</tr>
<tr>
  <th>1stFlrSF_log</th>         <td>    0.1445</td> <td>    0.015</td> <td>    9.645</td> <td> 0.000</td> <td>    0.115</td> <td>    0.174</td>
</tr>
<tr>
  <th>KitchenQual_Fa</th>       <td>   -1.0372</td> <td>    0.087</td> <td>  -11.864</td> <td> 0.000</td> <td>   -1.209</td> <td>   -0.866</td>
</tr>
<tr>
  <th>Neighborhood_OldTown</th> <td>   -0.8625</td> <td>    0.063</td> <td>  -13.615</td> <td> 0.000</td> <td>   -0.987</td> <td>   -0.738</td>
</tr>
<tr>
  <th>KitchenQual_Gd</th>       <td>   -0.4021</td> <td>    0.050</td> <td>   -8.046</td> <td> 0.000</td> <td>   -0.500</td> <td>   -0.304</td>
</tr>
<tr>
  <th>Neighborhood_Edwards</th> <td>   -0.7019</td> <td>    0.048</td> <td>  -14.530</td> <td> 0.000</td> <td>   -0.797</td> <td>   -0.607</td>
</tr>
<tr>
  <th>Neighborhood_IDOTRR</th>  <td>   -0.8583</td> <td>    0.097</td> <td>   -8.855</td> <td> 0.000</td> <td>   -1.048</td> <td>   -0.668</td>
</tr>
<tr>
  <th>LotArea_log</th>          <td>    0.1096</td> <td>    0.015</td> <td>    7.387</td> <td> 0.000</td> <td>    0.081</td> <td>    0.139</td>
</tr>
<tr>
  <th>Neighborhood_NridgHt</th> <td>    0.3854</td> <td>    0.057</td> <td>    6.809</td> <td> 0.000</td> <td>    0.274</td> <td>    0.496</td>
</tr>
<tr>
  <th>BldgType_Duplex</th>      <td>   -0.4073</td> <td>    0.061</td> <td>   -6.678</td> <td> 0.000</td> <td>   -0.527</td> <td>   -0.288</td>
</tr>
<tr>
  <th>Neighborhood_NAmes</th>   <td>   -0.3763</td> <td>    0.038</td> <td>   -9.981</td> <td> 0.000</td> <td>   -0.450</td> <td>   -0.302</td>
</tr>
<tr>
  <th>Neighborhood_SWISU</th>   <td>   -0.6263</td> <td>    0.089</td> <td>   -7.020</td> <td> 0.000</td> <td>   -0.801</td> <td>   -0.451</td>
</tr>
<tr>
  <th>Neighborhood_BrkSide</th> <td>   -0.5641</td> <td>    0.066</td> <td>   -8.493</td> <td> 0.000</td> <td>   -0.694</td> <td>   -0.434</td>
</tr>
<tr>
  <th>Neighborhood_Sawyer</th>  <td>   -0.4026</td> <td>    0.055</td> <td>   -7.342</td> <td> 0.000</td> <td>   -0.510</td> <td>   -0.295</td>
</tr>
<tr>
  <th>Neighborhood_NoRidge</th> <td>    0.4347</td> <td>    0.070</td> <td>    6.221</td> <td> 0.000</td> <td>    0.298</td> <td>    0.572</td>
</tr>
<tr>
  <th>Neighborhood_StoneBr</th> <td>    0.4538</td> <td>    0.087</td> <td>    5.226</td> <td> 0.000</td> <td>    0.283</td> <td>    0.624</td>
</tr>
<tr>
  <th>Neighborhood_MeadowV</th> <td>   -0.6622</td> <td>    0.118</td> <td>   -5.592</td> <td> 0.000</td> <td>   -0.895</td> <td>   -0.430</td>
</tr>
<tr>
  <th>SaleType_New</th>         <td>    0.1483</td> <td>    0.044</td> <td>    3.388</td> <td> 0.001</td> <td>    0.062</td> <td>    0.234</td>
</tr>
<tr>
  <th>Neighborhood_BrDale</th>  <td>   -0.4733</td> <td>    0.123</td> <td>   -3.839</td> <td> 0.000</td> <td>   -0.715</td> <td>   -0.231</td>
</tr>
<tr>
  <th>MSZoning_RM</th>          <td>    1.0820</td> <td>    0.147</td> <td>    7.363</td> <td> 0.000</td> <td>    0.794</td> <td>    1.370</td>
</tr>
<tr>
  <th>MSZoning_RL</th>          <td>    0.9916</td> <td>    0.156</td> <td>    6.356</td> <td> 0.000</td> <td>    0.686</td> <td>    1.298</td>
</tr>
<tr>
  <th>MSZoning_FV</th>          <td>    1.2052</td> <td>    0.165</td> <td>    7.284</td> <td> 0.000</td> <td>    0.881</td> <td>    1.530</td>
</tr>
<tr>
  <th>MSZoning_RH</th>          <td>    0.8503</td> <td>    0.189</td> <td>    4.490</td> <td> 0.000</td> <td>    0.479</td> <td>    1.222</td>
</tr>
<tr>
  <th>Neighborhood_NWAmes</th>  <td>   -0.2055</td> <td>    0.054</td> <td>   -3.837</td> <td> 0.000</td> <td>   -0.311</td> <td>   -0.100</td>
</tr>
<tr>
  <th>Neighborhood_Mitchel</th> <td>   -0.1943</td> <td>    0.065</td> <td>   -3.004</td> <td> 0.003</td> <td>   -0.321</td> <td>   -0.067</td>
</tr>
<tr>
  <th>Neighborhood_SawyerW</th> <td>   -0.1666</td> <td>    0.058</td> <td>   -2.862</td> <td> 0.004</td> <td>   -0.281</td> <td>   -0.052</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>295.535</td> <th>  Durbin-Watson:     </th> <td>   1.965</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1270.571</td> 
</tr>
<tr>
  <th>Skew:</th>          <td>-0.903</td>  <th>  Prob(JB):          </th> <td>1.26e-276</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.198</td>  <th>  Cond. No.          </th> <td>    48.7</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Use Feature ranking with recursive feature elimination

Use feature ranking to select the 5 most important features


```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select = 5)
selector = selector.fit(X, y.values.ravel()) # convert y to 1d np array to prevent DataConversionWarning
selector.support_ 
```




    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
            True,  True,  True,  True, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False,  True, False, False, False, False, False, False, False,
           False, False])



Fit the linear regression model again using the 5 selected columns


```python
selected_columns = X.columns[selector.support_ ]
linreg.fit(X[selected_columns],y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



Now, predict $\hat y$ using your model. You can use `.predict()` in scikit-learn. 


```python
yhat = linreg.predict(X[selected_columns])
```

Now, using the formulas of R-squared and adjusted R-squared below, and your Python/numpy knowledge, compute them and contrast them with the R-squared and adjusted R-squared in your statsmodels output using stepwise selection. Which of the two models would you prefer?

$SS_{residual} = \sum (y - \hat{y})^2 $

$SS_{total} = \sum (y - \bar{y})^2 $

$R^2 = 1- \dfrac{SS_{residual}}{SS_{total}}$

$R^2_{adj}= 1-(1-R^2)\dfrac{n-1}{n-p-1}$


```python
SS_Residual = np.sum((y-yhat)**2)
SS_Total = np.sum((y-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X[selected_columns].shape[1]-1)
```


```python
r_squared
```




    0.23943418177114217




```python
adjusted_r_squared
```




    0.2368187559863112



## Level up (Optional)

- Perform variable selection using forward selection, using this resource: https://planspace.org/20150423-forward_selection_with_statsmodels/. Note that this time features are added based on the adjusted R-squared!
- Tweak the code in the `stepwise_selection()` function written above to just perform forward selection based on the p-value 

## Summary
Great! You practiced your feature selection skills by applying stepwise selection and recursive feature elimination to the Ames Housing dataset! 
