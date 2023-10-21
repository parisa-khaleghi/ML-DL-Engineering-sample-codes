import sys
import datetime

sys.path.append('/Users/parisakhaleghi/Desktop/Coding/assist_projects/python/git_commit')

from git_commit import commit

commit( '/Users/parisakhaleghi/Desktop/Coding/ML-Engineering-sample-codes',
        '/Users/parisakhaleghi/Desktop/Coding/ML-Engineering-sample-codes/ML_with_python/7_logistic_regression_binary/exercise.ipynb',
        'update: group data set in employees who left and those who remain',
        datetime.datetime.now().year,
        datetime.datetime.now().month-1,
        datetime.datetime.now().day,
        datetime.datetime.now().hour,
        datetime.datetime.now().minute
        )