# Credit card fraud project

This is a small project on a credit card imbalanced dataset obtained through Kaggle. The dataset can be found on the following link:
https://www.kaggle.com/mlg-ulb/creditcardfraud

More details about the data can also be found on the link above, but we provide a short summary next. This dataset consists of various credit card transactions made by European cardholders. For privacy reasons, the data is provided in the form of PCA components (essentially a combination of the original variables that ensures that the covariance matrix is diagonal), except for two columns, Time and Amount, which, respectively, track the time at which a transaction have occurred and the associated amount. The last column, Class, contains labels 0 -- for legitimate transactions --, and 1 -- for fraudulent transactions.

The goal of the project is to develop and deploy a model that is able to correctly classify an unknown transaction as either legitimate or fraudulent. The challenge comes from the data imbalance. About 500 transactions of approximately 300.000 in total are fraudulent, corresponding to a ratio of about 0.0017.

In this project, we train a **random forest** model for dealing with the imbalance.

The **explore_data.ipynb** file is a jupyter notebook with some exploratory data analysis that can be looked at more carefully.
One of its major conclusions is that, of the about 500 transactions, only about 1/3 of it are responsible for 95% of all the amount loss due to the frauds, which suggests a possible strategy for improving results. We observed some improvement by restricting ourselves to a limited set of fraudulent transactions, but have yet to quantify if the effect is statistically significant.
In this file, we also tried various approaches for imbalanced data: **oversampling with SMOTE**, **random undersampling**, a **combined** approach, and **assigning higher weights to class 1**. 
We concluded the following
**oversampling and combined do not seem to lead to substantial improvements; they could be there but we need a more accurate statistical investigation;**
**undersampling hurts the precision metric too much and therefore is not appropriate;**
**the weight assignment method seems to be the less costly and resulted in small improvements.**

The **model.py** file uses the knowledge gathered with the exploratory analysis to train the random forest.
I am in the process of tweaking it a bit, and have yet to account for the missing 2/3 of the fraudulent transactions (which are the majority, but account for only 5% of losses).

The next step in this project is to develop a **flask** app for deploying the model. We will be including that in the next few days.

