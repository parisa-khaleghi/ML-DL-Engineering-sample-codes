import numpy as np

def gradient_decent(x, y):
    m_current = b_current = 0
    
    # define the number of iterations (steps)
    iterations = 1000

    n = len(x)

    # we can choose learning_rate as 0.01 or even 0.000001
    learning_rate = 0.001 
    for i in range(iterations):
        y_predicted = m_current * x + b_current
        d_m  = -(2/n) * sum(x * (y - y_predicted))
        d_b  = -(2/n) * sum(y - y_predicted)
        m_current = m_current - learning_rate * d_m
        b_current = b_current - learning_rate * d_b
        print("m {}, b {}, iteration {}".format(m_current, b_current, i))
        



# use numpy array instead of python list for convinience
# and it's faster
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_decent(x, y)