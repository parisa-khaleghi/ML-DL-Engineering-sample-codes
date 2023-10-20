import pandas as pd
import numpy as np
import os

df = pd.read_csv(os.getcwd() + "/ML with python/3- Gradient Descent and Cost Function/test_scores.csv")
from IPython.display import display
import math

x = np.array(df.math)
y = np.array(df.cs)

def gradient_descent(x, y):

    m_current = b_current = 0
    n = len(x)
    
    learning_rate =   0.0002 # 31.604511334602297
    cost = cost_next = 0

    for i in range(1000000):
        y_predicted = m_current * x + b_current
        cost_next = (1/n) * sum ([val**2 for val in (y-y_predicted)])

        if math.isclose(cost, cost_next, rel_tol=1e-20):
            print("cost {}, cost_next {}, iteration {}, ==".format(cost, cost_next, i))
            break;
        cost = cost_next
        d_m  = -(2/n) * sum(x * (y - y_predicted))
        d_b  = -(2/n) * sum(y - y_predicted)
        m_current = m_current - learning_rate * d_m
        b_current = b_current - learning_rate * d_b
        # print("m {}, b {}, cost {}, iteration {}".format(m_current, b_current, cost, i))
        print("cost {}, iteration {}".format(cost, i))
        
gradient_descent(x, y)