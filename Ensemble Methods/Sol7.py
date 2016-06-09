
import pandas as pd
import numpy as np
import pylab as pl
from numpy import genfromtxt, savetxt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report

pd.options.mode.chained_assignment = None  # default='warn'

# Tuning parameters
NoOfEstimators = 200
NoJobs = 1

train_df = pd.read_csv("cs-training.csv")
test_df = pd.read_csv("cs-test.csv")

train_df = train_df[train_df.age != 0] #outlier

train_df.NumberOfDependents = train_df.NumberOfDependents.fillna(0)
test_df.NumberOfDependents = test_df.NumberOfDependents.fillna(0)
train_df.MonthlyIncome = train_df.MonthlyIncome.fillna(6670)
test_df.MonthlyIncome = test_df.MonthlyIncome.fillna(6670)

features = np.array([	'RevolvingUtilizationOfUnsecuredLines',
		    'DebtRatio','MonthlyIncome','NumberOfDependents',
		    'NumberOfOpenCreditLinesAndLoans',
		    'NumberOfTime30-59DaysPastDueNotWorse',
		    'NumberOfTime60-89DaysPastDueNotWorse',
		    'NumberOfTimes90DaysLate',
		    'NumberRealEstateLoansOrLines',
		    'age'
		    ])

def rf_predictedValue():
    print '----------RandomForest----------'
    rf_clf = RandomForestClassifier(n_estimators = NoOfEstimators, n_jobs = NoJobs)
    rf_clf.fit(train_df[features], train_df['SeriousDlqin2yrs'])
    rf_predictedValue = rf_clf.predict_proba(test_df[features])
    print 'Feature Importance = %s' % rf_clf.feature_importances_
    return rf_predictedValue[:,1]

def ef_predictedValue():
    print '----------ExtraForest----------'
    ef_clf = ExtraTreesClassifier(n_estimators = NoOfEstimators, n_jobs = NoJobs)
    ef_clf.fit(train_df[features], train_df['SeriousDlqin2yrs'])
    ef_predictedValue = ef_clf.predict_proba(test_df[features])
    print 'Feature Importance = %s' % ef_clf.feature_importances_
    return ef_predictedValue[:,1]

def ab_predictedValue():
    print '----------AdaBoost----------'
    ab_clf = AdaBoostClassifier(n_estimators = NoOfEstimators)
    ab_clf.fit(train_df[features], train_df['SeriousDlqin2yrs'])
    ab_predictedValue = ab_clf.predict_proba(test_df[features])
    print 'Feature Importance = %s' % ab_clf.feature_importances_
    return ab_predictedValue[:,1]

def gb_predictedValue():
    print '----------GradientBoosting----------'
    gb_clf = GradientBoostingClassifier(n_estimators = NoOfEstimators)
    gb_clf.fit(train_df[features], train_df['SeriousDlqin2yrs'])
    gb_predictedValue = gb_clf.predict_proba(test_df[features])
    print 'Feature Importance = %s' % gb_clf.feature_importances_
    return gb_predictedValue[:,1]


def main():
    results = np.vstack((rf_predictedValue(), ef_predictedValue(), ab_predictedValue(), gb_predictedValue())).T
    avg_results = np.mean(results, axis=1)

    predicted_probs = [[index + 1, x] for index, x in enumerate(avg_results)]

    savetxt('submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', header='Id,Probability', comments = '')
    savetxt('results.csv', results, delimiter=',', fmt='%f,%f,%f,%f', header='rf,ef,ab,gb', comments = '')

if __name__ == "__main__":
    main()

