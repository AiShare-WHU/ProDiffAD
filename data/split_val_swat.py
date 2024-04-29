#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#normal0 = pd.read_excel('SWaT_Dataset_Normal_v0.xlsx')
normal1 = pd.read_excel('SWaT_Dataset_Normal_v1.xlsx')
attack = pd.read_excel('SWaT_Dataset_Attack_v0.xlsx')


# In[3]:


import os
#if not os.path.isfile(directory + 'train.csv'):
#normal0.to_csv('train.csv', index=None, header=None)
normal1.to_csv('train1.csv', index=None, header=None)
attack.to_csv('test.csv', index=None, header=None)


# In[79]:


test = pd.read_csv('test.csv')
train1 = pd.read_csv('train1.csv')
#train = pd.read_csv('train.csv')


# In[51]:


train.shape


# In[80]:


train1.shape


# In[81]:


test.shape


# In[82]:


train1['Normal/Attack'] = [0 if x == 'Normal' else 1 for x in train1['Normal/Attack']]
test['Normal/Attack'] = [0 if x == 'Normal' else 1 for x in test['Normal/Attack']]


# In[83]:


test['Normal/Attack'].value_counts()


# In[84]:


train1['Normal/Attack'].value_counts()


# In[85]:


# trim column names
train1 = train1.rename(columns=lambda x: x.strip())
test = test.rename(columns=lambda x: x.strip())


# In[86]:


test_labels = test['Normal/Attack']
test_labels


# In[87]:


def search_ratio(test_labels, val_len):
    test = test_labels[val_len:]
    val = test_labels[:val_len]
    test_ratio = (np.sum(test) /test.shape[0]) * 100
    val_ratio = (np.sum(val) / val.shape[0]) * 100
    print(f'val ratio: {val_ratio}')
    print(f'test ratio: {test_ratio}')
    print('----')
    return val_ratio, test_ratio


# In[88]:


vr, tr = search_ratio(test_labels=test_labels.to_numpy(), val_len=int(0.2 * test.shape[0]))
vr, tr = search_ratio(test_labels=test_labels.to_numpy(), val_len=int(0.1 * test.shape[0]))
vr, tr = search_ratio(test_labels=test_labels.to_numpy(), val_len=int(0.3 * test.shape[0]))


# In[89]:


val_len = int(0.1 * test.shape[0])
val_len


# In[90]:


validation = test[:val_len]
validation.shape


# In[91]:


validation['Normal/Attack'].value_counts()


# In[92]:


validation_labels = validation['Normal/Attack']
validation_labels


# In[93]:


test_clipped = test[val_len:]
test_clipped.shape


# In[94]:


test_labels_clipped = test_labels[val_len:]
test_labels_clipped.shape


# In[95]:


plt.rcParams['figure.figsize'] = 30, 2
plt.plot(test_clipped.to_numpy()[:, 4])
plt.fill_between(np.arange(test_labels_clipped.to_numpy().shape[0]), test_labels_clipped.to_numpy(), color='red', alpha=0.7, linestyle='dashed', linewidth=0.3)


# In[96]:


plt.plot(validation.to_numpy()[:, 4])
plt.fill_between(np.arange(validation_labels.to_numpy().shape[0]), validation_labels.to_numpy(), color='red', alpha=0.7, linestyle='dashed', linewidth=0.3)


# In[37]:


plt.plot(train_values[:, 4])


# In[97]:


# we don't need timestamps or training labels
train_dropped = train1.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
test_dropped = test_clipped.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
validation_dropped = validation.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)


# In[98]:


train_dropped.head()


# In[99]:


test_dropped.head()


# In[100]:


validation_dropped.head()


# In[101]:


# Transform all columns into float64
for i in list(train_dropped):
    train_dropped[i]=train_dropped[i].apply(lambda x: str(x).replace("," , "."))
train_dropped = train_dropped.astype(float)

for i in list(test_dropped):
    test_dropped[i]=test_dropped[i].apply(lambda x: str(x).replace("," , "."))
test_dropped = test_dropped.astype(float)

for i in list(validation_dropped):
    validation_dropped[i]=validation_dropped[i].apply(lambda x: str(x).replace("," , "."))
validation_dropped = validation_dropped.astype(float)


# In[57]:


train_values = train_dropped.values
test_values = test_dropped.values
validation_values = validation_dropped.values


# In[102]:


print(f'Train min: {train_values.min()}')
print(f'Train max: {train_values.max()}')
print('---')
print(f'Test min: {test_values.min()}')
print(f'Test max: {test_values.max()}')
print('---')
print(f'Validation min: {validation_values.min()}')
print(f'Validation max: {validation_values.max()}')


# In[32]:


test_values.min(axis=0)


# In[30]:


train_values.shape


# In[103]:


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

def scale_data(train, test, validation):
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True).fit(train)
    #scaler = MaxAbsScaler().fit(train)

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    validation_scaled = scaler.transform(validation)

    # train_scaled = scaler.fit_transform(train)
    # validation_scaled = scaler.fit_transform(validation)
    # test_scaled = scaler.fit_transform(test)

    return train_scaled, test_scaled, validation_scaled


# In[104]:


train_norm, test_norm, validation_norm = scale_data(train_values, test_values, validation_values)


# In[105]:


for i in range(51):
    print(f'-----Dim {i}----')
    print(f'TRAIN original: {train_values[:, i].min()}, {train_values[:, i].max()}')
    print(f'TRAIN norm:    {train_norm[:, i].min()}, {train_norm[:, i].max()}')
    print(f'TEST original: {test_values[:, i].min()}, {test_values[:, i].max()}')
    print(f'TEST norm:    {test_norm[:, i].min()}, {test_norm[:, i].max()}')
    print(f'VAL original: {validation_values[:, i].min()}, {validation_values[:, i].max()}')
    print(f'VAL norm:    {validation_norm[:, i].min()}, {validation_norm[:, i].max()}')


# In[106]:


print(train_norm.min(), train_norm.max())
print(test_norm.min(), test_norm.max())
print(validation_norm.min(), validation_norm.max())
print(train_norm.shape, test_norm.shape, validation_norm.shape)


# In[107]:


test_labels_clipped = test_labels_clipped.to_numpy()
test_labels_reshaped = np.zeros_like(test_norm)

for idx in range(0, len(test_labels_clipped)):
    if test_labels_clipped[idx]:
        # labels_reshaped.shape[1] == 51 aka num_feats
        test_labels_reshaped[idx][0:test_labels_reshaped.shape[1]] = 1


# In[108]:


validation_labels_reshaped = np.zeros_like(validation_norm)

for idx in range(0, len(validation_labels)):
    if validation_labels[idx]:
        # labels_reshaped.shape[1] == 51 aka num_feats
        validation_labels_reshaped[idx][0:validation_labels_reshaped.shape[1]] = 1


# In[109]:


print(test_labels_reshaped.shape, validation_labels_reshaped.shape)


# In[110]:


np.save('labels.npy', test_labels_reshaped)
np.save('labels_validation.npy', validation_labels_reshaped)
np.save('train.npy', train_norm)
np.save('test.npy', test_norm)
np.save('validation.npy', validation_norm)


# In[111]:


load_test = np.load('../processed/SWAT_big/validation.npy')


# In[81]:


import numpy as np
labels = np.load('../processed/SWAT_big/labels.npy')
labels_validation = np.load('../processed/SWAT_big/labels_validation.npy')
labels = (np.sum(labels, axis=1) >= 1) + 0
labels_validation = (np.sum(labels_validation, axis=1) >= 1) + 0
# labels.shape[0]


# In[82]:


((labels.sum() + labels_validation.sum()) / (labels.shape[0] + labels_validation.shape[0])) * 100


# In[55]:


labels.sum()


# In[61]:


(labels.sum() / labels.shape[0]) 


# In[113]:


load_test.shape


# In[ ]:




