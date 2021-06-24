import pandas as pd
import seaborn as sb
from PreProcessing import import_eeg
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#%%

metadata = pd.read_excel('Metadata_train.xlsx')[['Age', 'Gender', 'Epilepsy type', 'Diagnosis', 'Sleep state']]
metadata = metadata[metadata['Sleep state'] != 'sleep']
metadata = metadata[metadata['Diagnosis'] != 'undetermined']
metadata = metadata[metadata['Epilepsy type'] != 'undetermined']
print(len(metadata))

plt.figure()
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(21,4))
fig.tight_layout(pad=4.0)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1.2)

sb.color_palette("Paired")

ageP = sb.histplot(data=metadata['Age'], bins=10, ax=axs[0])
axs[0].set_title('Age distribution', fontsize=16)
ageP.set_xticklabels(ageP.get_xticks(), size=14)
ageP.set_yticklabels(ageP.get_yticks(), size=15)
axs[0].set_xlabel('Age (years)', fontsize=15)
axs[0].yaxis.label.set_size(15)
axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


genderP = sb.histplot(data=metadata['Gender'], ax=axs[1])
axs[1].set_title('Gender distribution', fontsize=16)
genderP.set_xticklabels(['Female', 'Male'], size=15)
genderP.set_yticklabels(genderP.get_yticks(), size=15)
axs[1].xaxis.label.set_size(15)
axs[1].yaxis.label.set_size(15)
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=20)

xl1 = ['generalized idiopathic', 'focal symptomatic', 'cryptogenic', 'focal cryptogenic']
xl2 = ['cardiovascular', 'epileptic seizure', 'other', 'provoked seizure', 'psychogenic', 'vagal syncope']

metadata = metadata[~metadata['Epilepsy type'].isnull()]
typesE = sb.histplot(data=metadata['Epilepsy type'], ax=axs[2])
axs[2].set_xticklabels(xl1, rotation=30, ha='right', fontsize=15)
axs[2].set_title('Epilepsy Type distribution', fontsize=16)
typesE.set_yticklabels(typesE.get_yticks(), size=15)
axs[2].set_xlabel('')

axs[2].xaxis.label.set_size(15)
axs[2].yaxis.label.set_size(15)
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


# fig.suptitle("Dataset's data distribution", va='center', fontsize=15)
# fig.align_labels()

#%Pie Chart

metadata = pd.read_excel('Metadata_train.xlsx')[['Age', 'Gender', 'Epilepsy type', 'Diagnosis', 'Sleep state']]
metadata = metadata[metadata['Sleep state'] != 'sleep']
metadata = metadata[metadata['Diagnosis'] != 'undetermined']
metadata = metadata[metadata['Epilepsy type'] != 'undetermined']
print(len(metadata))
# plt.style.use('ggplot')
# fig, ax = plt.subplots(figsize=(5,5))
# fig.tight_layout(pad=2.0)
x = metadata['Diagnosis'].value_counts()
diagDpatches, texts, autotexts = axs[3].pie(x=x, autopct="%.1f%%", explode=[0.15]*6, labels=x.keys(), pctdistance=0.75, startangle=-80, radius=1.2)
[ _.set_fontsize(16) for _ in texts ]
[ _.set_fontsize(14) for _ in autotexts ]
autotexts[-1].set_fontsize(0)
autotexts[-2].set_fontsize(0)
axs[3].set_title('Diagnosis Labels Distribution', fontsize=16)
axs[3].xaxis.label.set_size(20)
axs[3].yaxis.label.set_size(20)

#%% Age and Gender percentages
metadata = pd.read_excel('Metadata_train.xlsx')[['Age', 'Gender', 'Epilepsy type', 'Diagnosis', 'Sleep state']]
metadata = metadata[metadata['Sleep state'] != 'sleep']
metadata = metadata[metadata['Diagnosis'] != 'undetermined']
metadata = metadata[metadata['Epilepsy type'] != 'undetermined']

print(len(metadata))

males = sum(metadata['Gender'] == 'male')
females = sum(metadata['Gender'] == 'female')
males = males / (males + females) * 100
females = 100 - males

epileptic = sum(metadata['Diagnosis'] == 'epileptic seizure')
non_epileptic = sum(metadata['Diagnosis'] != 'epileptic seizure')

epileptic = epileptic / (epileptic + non_epileptic) * 100
focal = sum(metadata['Epilepsy type'] != 'focal symptomatic')

# Now let's look only at epileptic
focal_symptomatic = sum(metadata['Epilepsy type'] == 'focal symptomatic')
focal = focal_symptomatic /(focal_symptomatic + non_epileptic) * 100
non_focal = 100 - focal

#%%
metadata = pd.read_excel('Metadata_train.xlsx')[['Age', 'Gender', 'Diagnosis', 'Epilepsy type', 'Sleep state']]
metadata = metadata[metadata['Sleep state'] != 'sleep']

m_gender = metadata[~ metadata['Epilepsy type'].isnull()][['Gender', 'Diagnosis', 'Epilepsy type']]
m_gender = m_gender[m_gender['Diagnosis'] == 'epileptic seizure'][['Gender', 'Epilepsy type']]
m_gender['Epilepsy type'][m_gender['Epilepsy type'] != 'focal symptomatic'] = 'other epilepsy type'

pct_gender = (m_gender.groupby(['Epilepsy type','Gender']).size() / m_gender.groupby(['Gender']).size()).reset_index().rename({0:'percent'}, axis=1)

m_age = metadata[~ metadata['Epilepsy type'].isnull()][['Age', 'Diagnosis', 'Epilepsy type']]
m_age = m_age[m_age['Diagnosis'] == 'epileptic seizure'][['Age', 'Epilepsy type']]
m_age['Epilepsy type'][m_age['Epilepsy type'] != 'focal symptomatic'] = 'other epilepsy type'
m_age['Age'][m_age['Age'] >= 50] = 'old'
m_age['Age'][m_age['Age'] != 'old'] = 'young'

pct_age = (m_age.groupby(['Epilepsy type','Age']).size() / m_age.groupby(['Age']).size()).reset_index().rename({0:'percent'}, axis=1)

pct_gender.rename(columns={'Gender': 'Subset'}, inplace=True)
pct_age.rename(columns={'Age': 'Subset'}, inplace=True)

pct = pd.DataFrame([['focal symptomatic', 'all', 0.714], ['other epilepsy type', 'all', 0.286]], columns=['Epilepsy type', 'Subset', 'percent'])

pcts = pd.concat([pct, pct_age, pct_gender], axis=0)

sb.barplot(data=pcts, x='Subset', y='percent', hue='Epilepsy type')
plt.title('Subset-based epilepsy type distribution')


