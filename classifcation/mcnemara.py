import baseline
import multinomial_regression
import naive_bayes
import nicotine_df
import pandas as pd
from mlxtend.evaluate import mcnemar
import numpy as np

actual = nicotine_df.mcnemra_test_set[nicotine_df.predicates]['Nicotine'].to_list()
baseline = baseline.baseline().tolist()
multinomial = multinomial_regression.multinomial().tolist()
naivebayes = naive_bayes.naive_bayes().tolist()
baseline_correct_incorrect = [0,0]
multinomial_correct_incorrect = [0,0]
naivebayes_correct_incorrect = [0,0]
for i in range(len(actual)):
    if actual[i] != baseline[i]:
        baseline_correct_incorrect[1] += 1
    else:
        baseline_correct_incorrect[0] += 1
    if actual[i] != multinomial[i]:
        multinomial_correct_incorrect[1] += 1
    else:
        multinomial_correct_incorrect[0] += 1
    if actual[i] != naivebayes[i]:
        naivebayes_correct_incorrect[1] += 1
    else:
        naivebayes_correct_incorrect[0] += 1

print(baseline_correct_incorrect)
print(multinomial_correct_incorrect)
print(naivebayes_correct_incorrect)




print("Comparing baseline and multinomial")
print(mcnemar(np.asarray([np.asarray(baseline_correct_incorrect),np.asarray(multinomial_correct_incorrect)])))

print("Comparing baseline and bayes")
print(mcnemar(np.asarray([np.asarray(baseline_correct_incorrect),np.asarray(naivebayes_correct_incorrect)])))


print("Comparing multinomial and bayes")
print(mcnemar(np.asarray([np.asarray(multinomial_correct_incorrect),np.asarray(naivebayes_correct_incorrect)])))
