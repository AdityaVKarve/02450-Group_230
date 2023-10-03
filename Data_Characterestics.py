import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy import stats

dataset = pd.read_csv('./Dataset/drug_consumption_cleaned_normalised.csv')



def get_normal_plot(x):
    mean = np.mean(x)
    stdev = np.std(x)
    pdf = stats.norm.pdf(x.sort_values(),mean, stdev)

    return [x.sort_values(),pdf]

def get_normal_hist(x):
    sorted = x.sort_values()
    unique_vals = set(sorted)
    unique_val_count = len(unique_vals)
    return [unique_vals, sorted, unique_val_count]

print("Dataset columns:")
print(dataset.columns)
print("______________________________")


#Check dataset attributes
for c in dataset.columns:
    print(c)
    print("_____________________________________________")
    print("Mean:{}".format(np.mean(dataset[c])))
    print("Median:{}".format(np.median(dataset[c])))
    print("Mode:{}".format(stats.mode(dataset[c]).mode[0]))
    print("Standard Deviation:{}".format(np.std(dataset[c])))
    print("_____________________________________________")


#Plot personality traits-> The 5 continuous attributes
plt.subplot(211)
n_norm_x, n_norm_y = get_normal_plot(dataset['Neuroticism (N-Score)'])
e_norm_x, e_norm_y = get_normal_plot(dataset['Extroversion (E-Score)'])
o_norm_x, o_norm_y = get_normal_plot(dataset['Openness (O-Score)'])
a_norm_x, a_norm_y = get_normal_plot(dataset['Agreeableness (A-Score)'])
c_norm_x, c_norm_y = get_normal_plot(dataset['Conscientiousness (C-Score)'])

plt.subplot(2, 3, 1)
fig = plt.plot(n_norm_x, n_norm_y)
plt.xlabel('Neuroticism (N-Score)')
plt.ylabel('Probability')

plt.subplot(2, 3, 2)
fig = plt.plot(e_norm_x, e_norm_y)
plt.xlabel('Extroversion (E-Score)')
plt.ylabel('Probability')

plt.subplot(2, 3, 3)
fig = plt.plot(o_norm_x, o_norm_y)
plt.xlabel('Openness (O-Score)')
plt.ylabel('Probability')

plt.subplot(2, 3, 4)
fig = plt.plot(a_norm_x, a_norm_y)
plt.xlabel('Agreeableness (A-Score)')
plt.ylabel('Probability')

plt.subplot(2, 3, 5)
fig = plt.plot(c_norm_x, c_norm_y)
plt.xlabel('Conscientiousness (C-Score)')
plt.ylabel('Probability')
plt.show()

#Plot personality traits-> The 2 discrete attributes
i_x, i_y, i_count = get_normal_hist(dataset['Impulsiveness (BIS-11)'])
s_x, s_y, s_count = get_normal_hist(dataset['Sensation (SS)'])

plt.subplot(2,1,1)
fig = plt.hist(i_y, bins = i_count, alpha=0.6, color='b')
plt.xlabel('Impulsiveness (BIS-11)')
plt.ylabel('Count')

plt.subplot(2,1,2)
fig = plt.hist(s_y, bins = s_count, alpha=0.6, color='g')
plt.xlabel('Sensation (SS)')
plt.ylabel('Count')
plt.show()


#Plot drug abuse-> Alcohol to Chocolate
al_x, al_y, al_count = get_normal_hist(dataset['Alcohol'])
am_x, am_y, am_count = get_normal_hist(dataset['Amphetamines'])
amy_x, amy_y, amy_count = get_normal_hist(dataset['Amyl nitrites'])
b_x, b_y, b_count = get_normal_hist(dataset['Benzos'])
cf_x, cf_y, cf_count = get_normal_hist(dataset['Caffeine'])
ch_x, ch_y, ch_count = get_normal_hist(dataset['Chocolate'])

plt.subplot(2,3,1)
fig = plt.hist(al_y, bins = al_count, alpha=0.6, color='m')
plt.xlabel('Alcohol')
plt.ylabel('Count')

plt.subplot(2,3,2)
fig = plt.hist(am_y, bins = am_count, alpha=0.6, color='b')
plt.xlabel('Amphetamines')
plt.ylabel('Count')

plt.subplot(2,3,3)
fig = plt.hist(amy_y, bins = amy_count, alpha=0.6, color='g')
plt.xlabel('Amyl Nitrites')
plt.ylabel('Count')

plt.subplot(2,3,4)
fig = plt.hist(b_y, bins = b_count, alpha=0.6, color='y')
plt.xlabel('Benzos')
plt.ylabel('Count')

plt.subplot(2,3,5)
fig = plt.hist(cf_y, bins = cf_count, alpha=0.6, color='c')
plt.xlabel('Caffeine')
plt.ylabel('Count')

plt.subplot(2,3,6)
fig = plt.hist(ch_y, bins = ch_count, alpha=0.6, color='k')
plt.xlabel('Chocolate')
plt.ylabel('Count')
plt.show()

#Plot drug abuse-> Cocaine to Legal highs
co_x, co_y, co_count = get_normal_hist(dataset['Cocaine'])
cr_x, cr_y, cr_count = get_normal_hist(dataset['Crack'])
ec_x, ec_y, ec_count = get_normal_hist(dataset['Ecstasy'])
h_x, h_y, h_count = get_normal_hist(dataset['Heroin'])
k_x, k_y, k_count = get_normal_hist(dataset['Ketamine'])
lh_x, lh_y, lh_count = get_normal_hist(dataset['Legal Highs'])

plt.subplot(2,3,1)
fig = plt.hist(co_y, bins = co_count, alpha=0.6, color='m')
plt.xlabel('Cocaine')
plt.ylabel('Count')

plt.subplot(2,3,2)
fig = plt.hist(cr_y, bins = cr_count, alpha=0.6, color='b')
plt.xlabel('Crack')
plt.ylabel('Count')

plt.subplot(2,3,3)
fig = plt.hist(ec_y, bins = ec_count, alpha=0.6, color='g')
plt.xlabel('Ecstasy')
plt.ylabel('Count')

plt.subplot(2,3,4)
fig = plt.hist(h_y, bins = h_count, alpha=0.6, color='y')
plt.xlabel('Heroin')
plt.ylabel('Count')

plt.subplot(2,3,5)
fig = plt.hist(k_y, bins = k_count, alpha=0.6, color='c')
plt.xlabel('Ketamine')
plt.ylabel('Count')

plt.subplot(2,3,6)
fig = plt.hist(lh_y, bins = lh_count, alpha=0.6, color='k')
plt.xlabel('Legal Highs')
plt.ylabel('Count')

plt.show()

#Plot drug abuse-> LSD to Inhalants
l_x, l_y, l_count = get_normal_hist(dataset['LSD'])
me_x, me_y, me_count = get_normal_hist(dataset['Methamphetamine'])
mm_x, mm_y, mm_count = get_normal_hist(dataset['Magic Mushrooms'])
ni_x, ni_y, ni_count = get_normal_hist(dataset['Nicotine'])
se_x, se_y, se_count = get_normal_hist(dataset['Semer'])
in_x, in_y, in_count = get_normal_hist(dataset['Inhalants/Volatiles'])

plt.subplot(2,3,1)
fig = plt.hist(l_y, bins = l_count, alpha=0.6, color='m')
plt.xlabel('LSD')
plt.ylabel('Count')

plt.subplot(2,3,2)
fig = plt.hist(me_y, bins = me_count, alpha=0.6, color='b')
plt.xlabel('Methamphetamine')
plt.ylabel('Count')

plt.subplot(2,3,3)
fig = plt.hist(mm_y, bins = mm_count, alpha=0.6, color='g')
plt.xlabel('Magic Mushrooms')
plt.ylabel('Count')

plt.subplot(2,3,4)
fig = plt.hist(ni_y, bins = ni_count, alpha=0.6, color='y')
plt.xlabel('Nicotine')
plt.ylabel('Count')

plt.subplot(2,3,5)
fig = plt.hist(se_y, bins = se_count, alpha=0.6, color='c')
plt.xlabel('Semer')
plt.ylabel('Count')

plt.subplot(2,3,6)
fig = plt.hist(in_y, bins = in_count, alpha=0.6, color='k')
plt.xlabel('Inhalants/Volatiles')
plt.ylabel('Count')

plt.show()

