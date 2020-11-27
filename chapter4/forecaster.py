import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def insert_success(p, sample_size, ci):
    p = p / sample_size * 100
    p = p + (random.randrange(-ci,ci) if ci else 0)
    p = p / 100
    return True if random.random() < p else False


def calc_ci(sample_size):
    expected_efficiency = 0.1
    return int(1.96*math.sqrt(expected_efficiency*((1-expected_efficiency)/sample_size))*100)


def forecast(successes_one, successes_two, attempts):
    attempt_count, attempts = 0, []
    confidence_interval = calc_ci(sample_size)
    simulations = 100
    while len(attempts) < simulations:
        attempt_count += 1
        success_one = insert_success(successes_one, sample_size, confidence_interval) 
        success_two = insert_success(successes_two, sample_size, confidence_interval)
        if success_one and success_two:
            attempts.append(attempt_count)
            attempt_count = 0
    df = pd.DataFrame({'attempts': attempts})
    ax = sns.boxplot(data=df, x='attempts')
    plt.show()

    print('Average:', np.average(df.attempts))
    print('Median:', np.median(df.attempts))


#Experimental variables
successes_one = 2
successes_two = 2
sample_size = 10


#Forecast. Prints the average number of attempts required and the median (50th percentile).
forecast(successes_one, successes_two, sample_size)
