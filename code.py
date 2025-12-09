# Predict Eras Tour Outfits
# Camille Phaneuf-Hadd (cphaneuf@g.harvard.edu)

# ------------------------------ project overview -----------------------------

# --- RESEARCH QUESTION ---
# Did Taylor Swift wear predictable combinations of outfits during the Eras Tour?

# --- DATA DESCRIPTION ---
# Data adapted from the following Swiftie resources:
# Outfit data https://docs.google.com/spreadsheets/d/1WZyhckHAwOosHGA65h5dp5SHL5aoUMiyYXcY1k5MYUM/edit?gid=174092590#gid=174092590
# Benchmarking data https://docs.google.com/spreadsheets/d/1uvVEEqZsUbWCb61vSmpJRGfwtwoxbU70hkKgJiramHQ/edit?gid=1500331289#gid=1500331289

# ------------------------------ import packages ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import GridSearchCV 

# -------------------------- import and examine data --------------------------

# set working directory
wd = '/Users/camillephaneuf/Desktop/ANDL/G3Courses/Psy2085/eras_outfits/' # absolute path for Camille's use
os.chdir(wd)

# read in data
df = pd.read_csv("data.csv")

# set random number for use throughout project
rand = 13 # iykyk

# examine data structure
print(df.columns)
print(df.head(10))

# delete Unnamed: 28 and Resources column
cols_to_drop = df.columns[-2:].tolist() # get last 2 column names
df.drop(columns = cols_to_drop, inplace = True) # drop last 2 columns in place
print(df.columns)

# change data types; all columns should be categorical except for:
    # Date as date
    # Night, Standing Ovation as int
    # Koi Fish Guitar as binary
print(df.dtypes)
df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')
df['Koi Fish Guitar'] = df['Koi Fish Guitar'].map({'Yes': 1, 'No': 0}).astype('Int8')
exceptions = {'Date', 'Night', 'Standing Ovation', 'Koi Fish Guitar'}
for col in df.columns:
    if col not in exceptions:
        df[col] = df[col].astype('category')
print(df.dtypes)

# set plotting colors corresponding to album covers
Fear_col = "#CF9D2D"
SN_col = "#6D1AAD"
Red_col = "#B70202"
NEN_col = "#3DD5CE"
Rep_col = "#000000"
Love_col = "#FDA3DA"
Folk_col = "#A9A9A9"
Ever_col = "#EDE3C9"
Mid_col = "#282276"
TTPD_col= 'white'

# ensure no duplicate Dates
len(df['Date'].unique()) == len(df['Date']) # True -- check!
df[df.duplicated(subset=['Date'])] # empty -- check!

# Day barplot
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['Day'].value_counts().reindex(day_order)
day_counts.plot(kind = 'bar', color = 'gray', edgecolor = 'black', xlabel = 'Day', ylabel = 'Count')

# City barplot
city_order = df.drop_duplicates(subset = 'City', keep = 'first').sort_values('Date')['City']
city_counts = df['City'].value_counts().reindex(city_order)
ax_city = city_counts.plot(kind = 'bar', color = 'gray', edgecolor = 'black', xlabel = 'City', ylabel = 'Count')
plt.xticks(rotation = 90, fontsize = 4)
plt.show()

# Country barplot
country_order = df.drop_duplicates(subset = 'Country', keep = 'first').sort_values('Date')['Country']
country_counts = df['Country'].value_counts().reindex(country_order)
fig, ax_con = plt.subplots()
bars_con = ax_con.bar(country_counts.index, country_counts.values, color = 'gray', edgecolor = 'black')
for bar, country in zip(bars_con, country_counts.index):
    if country in ['USA']:
        bar.set_hatch('//////') # dense diagonal lines
        bar.set_facecolor('gray') # makes hatch visible
        bar.set_edgecolor('black')
ax_con.set_xlabel('Country')
ax_con.set_ylabel('Count')
plt.xticks(rotation = 90) # rotate x-axis labels
plt.show()

# Region barplot
region_order = df.drop_duplicates(subset = 'Region', keep = 'first').sort_values('Date')['Region']
region_counts = df['Region'].value_counts().reindex(region_order)
fig, ax_reg = plt.subplots()
bars_reg = ax_reg.bar(region_counts.index, region_counts.values, color = 'gray', edgecolor = 'black')
for bar, region in zip(bars_reg, region_counts.index):
    if region in ['Southwest', 'West', 'Southeast', 'Northeast', 'Midwest']:
        bar.set_hatch('//////') # dense diagonal lines
        bar.set_facecolor('gray') # makes hatch visible
        bar.set_edgecolor('black')
ax_reg.set_xlabel('Region')
ax_reg.set_ylabel('Count')
plt.xticks(rotation = 90) # rotate x-axis labels
plt.show()

# Night barplot
night_counts = df['Night'].value_counts()
night_counts.plot(kind = 'bar', color = 'gray', edgecolor = 'black', xlabel = 'Night', ylabel = 'Count')

# define era barplots
def plot_era_counts(column, brcolor, edcolor):
    '''
    Plots a bar chart of value counts for a given column.

    Parameters
    ----------
    column: str
        Name of the column to plot.
    '''
    counts = df[column].value_counts()
    counts.plot(kind = 'bar', color = brcolor, edgecolor = edcolor)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation = 90)
    plt.show()
    
# era = Love
plot_era_counts('Lover Bodysuit', Love_col, 'black')
plot_era_counts('The Man Jacket', Love_col, 'black')
plot_era_counts('Lover Mic #1', Love_col, 'black')
plot_era_counts('Lover Guitar', Love_col, 'black')
plot_era_counts('Lover Mic #2', Love_col, 'black')

# era = Fearless
plot_era_counts('Fearless Dress', Fear_col, 'black')
plot_era_counts('Fearless Mic', Fear_col, 'black')

# era = Red
plot_era_counts('Red Shirt', Red_col, 'black')
plot_era_counts('Red Mic #1', Red_col, 'black')
plot_era_counts('Red Romper', Red_col, 'black')
plot_era_counts('Red Guitar', Red_col, 'black')
plot_era_counts('Red Mic #2', Red_col, 'black')

# WANEGBT Chant (Kam) barplot
chant_order = df.sort_values('Date')['WANEGBT Chant (Kam)']
chant_counts = df['WANEGBT Chant (Kam)'].value_counts().reindex(chant_order)
usa_chants = df.loc[df['Country'] == 'USA', 'WANEGBT Chant (Kam)'].unique()
fig, ax_cnt = plt.subplots()
bars_cnt = ax_cnt.bar(chant_counts.index, chant_counts.values, color = Red_col, edgecolor = 'black')
for bar, chant in zip(bars_cnt, chant_counts.index):
    if chant in usa_chants:
        bar.set_hatch('//////') # dense diagonal lines
        bar.set_facecolor(Red_col) # makes hatch visible
        bar.set_edgecolor('black')
ax_cnt.set_xlabel('WANEGBT Chant (Kam)')
ax_cnt.set_ylabel('Count')
plt.xticks(rotation = 90, fontsize = 4) # rotate x-axis labels
plt.show()

# era = Speak Now
plot_era_counts('Speak Now Gown', SN_col, 'black')
plot_era_counts('Speak Now Mic', SN_col, 'black')

# Koi Fish Guitar barplot
koi_counts = df['Koi Fish Guitar'].value_counts()
koi_counts.plot(kind = 'bar', color = SN_col, edgecolor = 'black')
plt.xlabel('Koi Fish Guitar')
plt.ylabel('Count')
plt.xticks([0, 1], ["No", "Yes"], rotation = 90)
plt.show()

# era = Reputation
plot_era_counts('Reputation Jumpsuit', Rep_col, 'black')
plot_era_counts('Reputation Mic', Rep_col, 'black')

# era = Indie Sister
plot_era_counts('Evermore Dress', Ever_col, Folk_col)
plot_era_counts('Indie Sister Cloak', Ever_col, Folk_col)
plot_era_counts('Indie Sister Mic', Ever_col, Folk_col)

# Standing Ovation histogram
max_val = df['Standing Ovation'].max()
bins = np.arange(0, max_val, 20)
plt.hist(df['Standing Ovation'], bins = bins, color = Ever_col, edgecolor = Folk_col)
plt.xlabel('Standing Ovation')
plt.ylabel('Count')
plt.xticks(rotation = 90)
plt.show()

# era = 1989
plot_era_counts('1989 Match', NEN_col, 'black') 
plot_era_counts('1989 Mic', NEN_col, 'black') 

# 1989 outfit subplots
cols_1989 = ['1989 Top', '1989 Skirt', 'Left 1989 Boot', 'Right 1989 Boot']
fig, axs_1989 = plt.subplots(2, 2) # 2 rows, 2 columns
c = 0
for i in range(2):
    for j in range(2):
        column = cols_1989[c]
        counts = df[column].value_counts()
        counts.plot(kind = 'bar', ax = axs_1989[i, j], color = NEN_col, edgecolor = 'black')
        axs_1989[i, j].set_xlabel(column)
        axs_1989[i, j].set_ylabel('Count')
        axs_1989[i, j].tick_params(axis = 'x', labelrotation = 90)
        c += 1
plt.tight_layout()
plt.show()

# era = TTPD
plot_era_counts('TTPD Dress', TTPD_col, 'black')
plot_era_counts('TTPD Gloves', TTPD_col, 'black')
plot_era_counts('TTPD Choker', TTPD_col, 'black')
plot_era_counts('TTPD Mic #1', TTPD_col, 'black')
plot_era_counts('Broken Heart Set', TTPD_col, 'black')
plot_era_counts('Broken Heart Jacket', TTPD_col, 'black')
plot_era_counts('TTPD Mic #2', TTPD_col, 'black')

# era = Surprise
plot_era_counts('Surprise Song Dress', 'grey', 'black')

# era = Midnights
plot_era_counts('Midnights Fur', Mid_col, 'black')
plot_era_counts('Midnights Shirt', Mid_col, 'black')
plot_era_counts('Midnights Bodysuit', Mid_col, 'black')
plot_era_counts('Karma Jacket', Mid_col, 'black')
plot_era_counts('Midnights Mic', Mid_col, 'black')

# -------------------- multinomial logistic regression fit --------------------

# RQ #1a: how well can we predict the Lover Bodysuit (i.e., the 1st outfit of the show)
#         from day, city, country, region, and night?

# set X and y
X_cat = df[['Day', 'City', 'Country', 'Region', 'Night']]
X = pd.get_dummies(X_cat, drop_first = True) # dummy-code nominal variables
y = df['Lover Bodysuit']
print(X_cat.head()) # features as categories
print(X.head()) # featues as ordinal or dummy-coded numerics (needed for LogisticRegression)
print(y.head()) # target

# create train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rand) # 80-20 split
print(X_train)
y_train.value_counts()
print(X_test)
y_test.value_counts()

# "squash" ordinal variables into [0, 1] (nominal variables stay 0, 1)
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
print(X_train_scaled)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns, index = X_test.index)
print(X_test_scaled)

# fit the model on training data
lover_model1 = LogisticRegression(max_iter = 1000)           
lover_fit1 = lover_model1.fit(X_train, y_train)

# predictions on test data
print(lover_fit1.classes_)
pred_probs = lover_fit1.predict_proba(X_test)
print(pred_probs) # probabilistic predictions for each class
print(X_test.filter(regex = '^(Night|City_)').head())
# e.g., for Night 1 in Zurich, the Lover Bodysuit has the highest probability of being Blue & Gold
pred_labels = lover_fit1.predict(X_test) # prediction
pred_labels # predictions for each show in X_test

# model evaluation
labels = np.unique(y_test)
conf_mat = metrics.confusion_matrix(y_test, pred_labels, labels = labels)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(pd.DataFrame(conf_mat, index = labels, columns = labels)) # observed values in the rows, predicted values in the columns
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')
# our classifications are poor:
    # 0/3 correct for All Pink
    # 0/8 correct for Blue & Gold
    # 2/14 correct for Pink & Blue
    # 0/4 correct for Pink & Orange
    # 0/1 correct for Purple Tassels

# heatmap (source all at once)
plt.figure(figsize = (12,8))
sns.set(font_scale = 1.4)
ax = sns.heatmap(conf_mat, cmap = 'Greys', annot = True, fmt = 'd', square = True, xticklabels = labels , yticklabels = labels)
ax.set(xlabel = 'Predicted', ylabel = 'Actual')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(axis = 'both', labelsize = 10)
plt.show()

# evaluation metrics
lover_acc1 = lover_fit1.score(X_test, y_test)
print(f'Accuracy: {lover_acc1:.3f}')
target_labels = labels.tolist()
print(metrics.classification_report(y_true = y_test, y_pred = pred_labels, target_names = target_labels))
pd.Series(pred_labels).value_counts() 
# accuracy = proportion of all predictions that were correct (.067)
# precision = when the model assigns a label, how often that label is correct
# recall = how often the model correctly identifies instances of that label
# f1-score = "mean" of precision and recall
# support = frequencies of the observed labels

'''
Not surprisingly, knowing information like the night or the city does not help us 
predict the show opening outfit: the Lover Bodysuit. Let's engineer some features 
for our next model -- i.e., the number of shows since outfits were last worn.
'''

# RQ #1b: how well can we predict the Lover Bodysuit (i.e., the 1st outfit of the show)
#         from outfit delays AND day, city, country, region, and night?

# reminder of original data structure
print(df.columns)
print(df.head(10))

# columns to calculate "delay" for
exclude = {'Date', 'Day', 'City', 'Country', 'Region', 'Night',
           'WANEGBT Chant (Kam)', 'Standing Ovation'}
delay_cols = [c for c in df.columns if c not in exclude]

# make sure data are sorted chronologically
df = df.sort_values('Date').reset_index(drop = True)

# for each outfit column, compute delay since last seen
pos = pd.Series(df.index, index = df.index)
for col in delay_cols:
    # difference in index between current row and previous occurrence of the same value
    delay = pos.groupby(df[col], dropna = False).diff()
    # store as integers (first occurrences [NAs] are coded as -1)
    df[f"{col} Delay"] = delay.fillna(-1).astype(int)
    
# set new X and y
X_cat = df.filter(regex = '^(Day|City|Country|Region|Night|.*Delay)$')
X = pd.get_dummies(X_cat, drop_first = True) # dummy-code nominal variables
y = df['Lover Bodysuit']
print(X_cat.head()) # features as categories
print(X.head()) # featues as ordinal or dummy-coded numerics (needed for LogisticRegression)
print(y.head()) # target

# create train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rand) # 80-20 split
print(X_train)
y_train.value_counts()
print(X_test)
y_test.value_counts()

# "squash" ordinal variables into [0, 1] (nominal variables stay 0, 1)
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
print(X_train_scaled)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns, index = X_test.index)
print(X_test_scaled)

# fit the model on training data
lover_model2 = LogisticRegression(max_iter = 1000)           
lover_fit2 = lover_model2.fit(X_train, y_train)

# predictions on test data
print(lover_fit2.classes_)
pred_probs = lover_fit2.predict_proba(X_test)
print(pred_probs) # probabilistic predictions for each class
print(X_test.filter(regex = '^(Night|City_)').head())
# e.g., for Night 1 in Zurich, the Lover Bodysuit has the highest probability of being Blue & Gold (no change from the 1st model)
pred_labels = lover_fit2.predict(X_test) # prediction
pred_labels # predictions for each show in X_test

# model evaluation
labels = np.unique(y_test)
conf_mat = metrics.confusion_matrix(y_test, pred_labels, labels = labels)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(pd.DataFrame(conf_mat, index = labels, columns = labels)) # observed values in the rows, predicted values in the columns
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')
# our classifications are poor:
    # 0/5 correct for All Pink
    # 1/4 correct for Blue & Gold
    # 8/17 correct for Pink & Blue
    # 1/2 correct for Pink & Orange
    # 1/2 correct for Purple Tassels

# heatmap (source all at once)
plt.figure(figsize = (12,8))
sns.set(font_scale = 1.4)
ax = sns.heatmap(conf_mat, cmap = 'Greys', annot = True, fmt = 'd', square = True, xticklabels = labels , yticklabels = labels)
ax.set(xlabel = 'Predicted', ylabel = 'Actual')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(axis = 'both', labelsize = 10)
plt.show()

# evaluation metrics
lover_acc2 = lover_fit2.score(X_test, y_test)
print(f'Accuracy: {lover_acc2:.3f}')
target_labels = labels.tolist()
print(metrics.classification_report(y_true = y_test, y_pred = pred_labels, target_names = target_labels))
pd.Series(pred_labels).value_counts()
# accuracy = proportion of all predictions that were correct (.367)
# precision = when the model assigns a label, how often that label is correct
# recall = how often the model correctly identifies instances of that label
# f1-score = "mean" of precision and recall
# suppport = frequencies of the observed labels

'''
Including the number of shows since outfits were last worn really improved our model!
Right now, we have just over a 1 in 3 shot of knowing what Lover Bodysuit Taylor will
wear at a given show. Next, let's see which features were most predictive.
'''

# ----------- multinomial logistic regression fit with penalization -----------

# fit the model on training data using a L1 (lasso) penalty
lover_model2_l1 = LogisticRegression(max_iter = 10000, penalty ='l1', solver = 'liblinear') # use faster solver           
lover_fit2_l1 = lover_model2_l1.fit(X_train, y_train)

# predictions on test data
print(lover_fit2_l1.classes_)
pred_probs = lover_fit2_l1.predict_proba(X_test)
print(pred_probs) # probabilistic predictions for each class
print(X_test.filter(regex = '^(Night|City_)').head())
# e.g., for Night 1 in Zurich, the Lover Bodysuit has the highest probability of being Blue & Gold (no change from the 2nd model)
pred_labels = lover_fit2_l1.predict(X_test) # prediction
pred_labels # predictions for each show in X_test

# model evaluation
labels = np.unique(y_test)
conf_mat = metrics.confusion_matrix(y_test, pred_labels, labels = labels)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(pd.DataFrame(conf_mat, index = labels, columns = labels)) # observed values in the rows, predicted values in the columns
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')
# our classifications are poor:
    # 0/3 correct for All Pink
    # 1/5 correct for Blue & Gold
    # 6/17 correct for Pink & Blue
    # 2/4 correct for Pink & Orange
    # 1/1 correct for Purple Tassels

# heatmap (source all at once)
plt.figure(figsize = (12,8))
sns.set(font_scale = 1.4)
ax = sns.heatmap(conf_mat, cmap = 'Greys', annot = True, fmt = 'd', square = True, xticklabels = labels , yticklabels = labels)
ax.set(xlabel = 'Predicted', ylabel = 'Actual')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(axis = 'both', labelsize = 10)
plt.show()

# evaluation metrics
lover_acc2_l1 = lover_fit2_l1.score(X_test, y_test)
print(f'Accuracy: {lover_acc2_l1:.3f}')
target_labels = labels.tolist()
print(metrics.classification_report(y_true = y_test, y_pred = pred_labels, target_names = target_labels))
pd.Series(pred_labels).value_counts()
# accuracy = proportion of all predictions that were correct (.333)
# precision = when the model assigns a label, how often that label is correct
# recall = how often the model correctly identifies instances of that label
# f1-score = "mean" of precision and recall
# suppport = frequencies of the observed labels

'''
The L1 (lasso) penalty allows parameters to shrink to 0. It slightly decreased our
accuracy. This provides early evidence against our model being over-parameterized.
'''

# fit the model on training data using a L2 (ridge) penalty
lover_model2_l2 = LogisticRegression(max_iter = 10000, penalty ='l2', solver = 'newton-cholesky') # use faster solver           
lover_fit2_l2 = lover_model2_l2.fit(X_train, y_train)

# predictions on test data
print(lover_fit2_l2.classes_)
pred_probs = lover_fit2_l2.predict_proba(X_test)
print(pred_probs) # probabilistic predictions for each class
print(X_test.filter(regex = '^(Night|City_)').head())
# e.g., for Night 1 in Zurich, the Lover Bodysuit has the highest probability of being Blue & Gold (no change from the 2nd model)
pred_labels = lover_fit2_l2.predict(X_test) # prediction
pred_labels # predictions for each show in X_test

# model evaluation
labels = np.unique(y_test)
conf_mat = metrics.confusion_matrix(y_test, pred_labels, labels = labels)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(pd.DataFrame(conf_mat, index = labels, columns = labels)) # observed values in the rows, predicted values in the columns
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')
# our classifications are poor:
    # 0/5 correct for All Pink
    # 1/4 correct for Blue & Gold
    # 7/17 correct for Pink & Blue
    # 1/3 correct for Pink & Orange
    # 1/1 correct for Purple Tassels

# heatmap (source all at once)
plt.figure(figsize = (12,8))
sns.set(font_scale = 1.4)
ax = sns.heatmap(conf_mat, cmap = 'Greys', annot = True, fmt = 'd', square = True, xticklabels = labels , yticklabels = labels)
ax.set(xlabel = 'Predicted', ylabel = 'Actual')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(axis = 'both', labelsize = 10)
plt.show()

# evaluation metrics
lover_acc2_l2 = lover_fit2_l2.score(X_test, y_test)
print(f'Accuracy: {lover_acc2_l2:.3f}')
target_labels = labels.tolist()
print(metrics.classification_report(y_true = y_test, y_pred = pred_labels, target_names = target_labels))
pd.Series(pred_labels).value_counts()
# accuracy = proportion of all predictions that were correct (.333)
# precision = when the model assigns a label, how often that label is correct
# recall = how often the model correctly identifies instances of that label
# f1-score = "mean" of precision and recall
# suppport = frequencies of the observed labels

'''
The L2 (ridge) penalty allows parameters to shrink, but not all the way to 0. It slightly
decreased our accuracy. Again, the penalization models provide early evidence against
over-parameterization.
'''

# -------------------- multinomial logistic regression fit --------------------

# RQ #2a: how well can we predict the Fearless Dress (i.e., the 2nd outfit of the show)
#         from the Lover outfit AND outfit delays AND day, city, country, region, and night?

# reminder of original data structure
print(df.columns)
print(df.head(10))
    
# set new X and y
X_cat = df.filter(regex = '^(Day|City|Country|Region|Night|.*Delay)$')
new_cols = ['Lover Bodysuit', 'The Man Jacket', 'Lover Mic #1', 'Lover Guitar', 'Lover Mic #2']
X_cat_new = df.filter(new_cols)
X_cat = pd.concat([X_cat, X_cat_new], axis = 1)
print(X_cat.columns) # got them all!
X = pd.get_dummies(X_cat, drop_first = True) # dummy-code nominal variables
y = df['Fearless Dress']
print(X_cat.head()) # features as categories
print(X.head()) # featues as ordinal or dummy-coded numerics (needed for LogisticRegression)
print(y.head()) # target

# create train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rand) # 80-20 split
print(X_train)
y_train.value_counts()
print(X_test)
y_test.value_counts()

# "squash" ordinal variables into [0, 1] (nominal variables stay 0, 1)
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
print(X_train_scaled)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns, index = X_test.index)
print(X_test_scaled)

# fit the model on training data
fearless_model = LogisticRegression(max_iter = 10000) # needed to up iterations for model convergence           
fearless_fit = fearless_model.fit(X_train, y_train)

# predictions on test data
print(fearless_fit.classes_)
pred_probs = fearless_fit.predict_proba(X_test)
print(pred_probs) # probabilistic predictions for each class
print(X_test.filter(regex = '^(Night|City_)').head())
# e.g., for Night 1 in Zurich, the Fearless Dress has the highest probability of being Long Gold
pred_labels = fearless_fit.predict(X_test) # prediction
pred_labels # predictions for each show in X_test

# model evaluation
labels = np.unique(y_test)
conf_mat = metrics.confusion_matrix(y_test, pred_labels, labels = labels)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(pd.DataFrame(conf_mat, index = labels, columns = labels)) # observed values in the rows, predicted values in the columns
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')
# our classifications are poor:
    # 0/1 correct for Blue/Silver
    # 1/1 correct for Gold/Black Tiger
    # 1/3 correct for Long Gold
    # 1/9 correct for Long Silver
    # 2/10 correct for Short Fringe
    # 2/6 correct for Silver/Black/Gold

# heatmap (source all at once)
plt.figure(figsize = (12,8))
sns.set(font_scale = 1.4)
ax = sns.heatmap(conf_mat, cmap = 'Greys', annot = True, fmt = 'd', square = True, xticklabels = labels , yticklabels = labels)
ax.set(xlabel = 'Predicted', ylabel = 'Actual')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(axis = 'both', labelsize = 10)
plt.show()

# evaluation metrics
fearless_acc = fearless_fit.score(X_test, y_test)
print(f'Accuracy: {fearless_acc:.3f}')
target_labels = labels.tolist()
print(metrics.classification_report(y_true = y_test, y_pred = pred_labels, target_names = target_labels))
pd.Series(pred_labels).value_counts()
# accuracy = proportion of all predictions that were correct (.233)
# precision = when the model assigns a label, how often that label is correct
# recall = how often the model correctly identifies instances of that label
# f1-score = "mean" of precision and recall
# suppport = frequencies of the observed labels

'''
Knowing what outfit Taylor wore for the Lover era gives us just under a 1 in 5 shot 
of knowing what Fearless Dress Taylor will wear at that show. Do our predictions improve
for subsequent eras?
'''

# ----------------------------------- to-do -----------------------------------

# do parameter tuning on fearless model
# finish implementing models where once you know 1 era's outfit combo, you predict the next era's outfit combo
# try other algorithms to make the best midnights model possible
# change midnights model training to 10-fold CV
