import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import random

#model imports

from sklearn.ensemble import Random ForestClassifier

from sklearn.neighbors import KNeighbors Classifier

from sklearn.linear_model import LogisticRegression

#processing imports

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

file_path_20_percent='../input/nslkdd/KDDTrain+_20 Percent.txt'

file path full_training_set='../input/nslkdd/KDDTrain+.txt'

file_path_test='../input/nslkdd/KDDTest+.txt'

#df=pd.read_csv(file_path_20_percent)

df=pd.read_csv(file_path_full_training_set)

test_df= pd.read_csv(file_path_test)

columns (['duration'

'protocol_type'

'service'

,'flag'

'src bytes'

'dst bytes'

'land'

,'wrong_fragment'

,'urgent'

'hot'

,'num_failed_logins'

,'logged_in'

'num_compromised'

,'root_shell'

'su_attempted'

,'num_root'

'num_file_creations'

,'num_shells'

'num_access_files'

'num_outbound_cmds'

,'is_host_login'
guest login

count

srv_count

serror_rate

srv_serror_rate

rerror rate

srv_rerror_rate

same srv rate'

'diff srv rate

srv_diff_host_rate

'dst_host_count'

dst host srv count

'dst_host_same_srv_rate

'dst_host_diff_srv_rate

'dst_host_same_sre_port_rate

'dst_host_srv_diff_host_rate

'dst_host_serror_rate

'dst_host_srv serror rate

'dst host_rerror rate

'dst_host_srv_rerror_rate

'attack'

'level'])

df.columns columns

test df.columns = columns

# sanity check

df.head()

is attack df.attack.map(lambda a: 0 if a 'normal' else 1)

test_attack test_df.attack.map(lambda a: 0 if a 'normal' else 1)

#data with attack df.join(is attack, rsuffix=_flag')

df['attack_flag] = is_attack

test df['attack flag'] = test_attack

#view the result

df.head()

dos attacks.

['apache2', 'back', 'land','neptune', 'mailbomb', 'pod','processtable', 'smurf', 'teardrop', 'udpstorm', 'worm"

probe_attacks = ['ipsweep','mscan','nmap', 'portsweep','saint','satan']

U2R['buffer_overflow','loadmdoule', 'perl','ps','rootkit','sqlattack', 'xterm']

Sybil

['ftp_write','guess_passwd","http_tunnel', 'imap', 'multihop', 'named','phf,'sendmail', 'sampgetattacks

nmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock','xsnoop']

#we will use these for plotting below

attack_labels ['Normal','DoS', 'Probe', 'U2R','Sybil']

#helper function to pass to data frame mapping

def map_attack(attack):

if attack in dos attacks:
dos_attacks map to 1

attack_type=1

elif attack in probe_attacks;

#probe attacks mapt to 2

attack type-2

elif attack in U2R:

#privilege escalation attacks map to 3

attack type=3

elif attack in Sybil:

#remote access attacks map to 4

attack_type=4

else:

#normal maps to 0

attack_type = 0

return attack_type

#map the data and join to the data set attack_mapdf.attack.apply(map_attack)

df['attack_map'] = attack_map

test_attack_map = test_df.attack.apply(map_attack)

test_df['attack_map'] = test_attack_map

#view the result

df.head()

attack vs protocol= pd.crosstab(df.attack, df.protocol_type)

attack vs protocol

def bake pies(data list,labels):

list_length len(data_list)

#setup for mapping colors

color_list sns.color_palette()

color_cycle itertools.cycle(color_list)

cdict = ()

#build the subplots

fig, axs plt.subplots(1, list_length, figsize=(18,10), tight_layout=False)

plt.subplots_adjust(wspace=1/list_length)

#loop through the data sets and build the charts

for count, data_set in enumerate(data_list):

#update our color mapt with new values

for num, value in enumerate(np.unique(data_set.index)):

if value not in cdict:

cdict[value] = next(color_cycle)

# build the wedges

wedges, texts = axs[count].pie(data_set,
colors[edict[v] for v in data_set index])

build the legend

axs[count] legend(wedges, data set index,

title "Flags",

loc="center left",

bbox_to_anchor (1, 0, 0.5, 1))

#set the title

axs[count].set_title(labels[count])

return axs

icmp_attacks attack_vs_protocol.icmp

top_attacks attack_vs_protocol.tcp

udp_attacks attack_vs_protocol.udp

#create the charts

bake_pies([icmp_attacks, tep_attacks, udp_attacks],['icmp','tep', 'udp'])

plt.show()

normal_flags df.loc[df.attack_flag=0].flag.value_counts()

attack_flags = df.loc[df.attack_flag=1].flag.value_counts()

#create the charts

flag_axs bake_pies([normal_flags, attack_flags], ['normal','attack'])

plt.show()

normal_services = df.loc[df.attack_flag 0].service.value_counts()

attack services=df.loc[df.attack_flag=1].service.value_counts()

#create the charts

service axs bake pies((normal services, attack_services], ['normalMCS', attack'])

plt.show()

features_to_encode = ['protocol_type', 'service', 'flag']

encoded = pd.get_dummies(df[features_to_encode])

test_encoded_base = pd.get_dummies(test_df[features_to_encode])

#not all of the features are in the test set, so we need to account for diffs

test_index = np.arange(len(test_df.index))

column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))

diff_df=pd.DataFrame(0, index=test_index,columns-column_diffs)

# we'll also need to reorder the columns to match, so let's get those

column_order encoded.columns.to_list()

# append the new columns

test_encoded_temp = test_encoded_base.join(diff_df)
reorder the columns

test final-test encoded temp(column order) fillna(0)

#get numeric features, we won't worry about encoding these at this point numeric_features['duration', 'src_bytes', 'dst_bytes']

#model to fit/test

to_fit encoded.join(df[numeric_features]) test_set test_final.join(test_df[numeric_features])

binary_ydf['attack_flag"]

multi_ydf['attack_map']

test_binary_y=test_df['attack_flag']

test_multi_y=test_df['attack_map']

#build the training sets

binary_train_X, binary_val_X, binary_train_y, binary_val_y=train_test_split(to_fit, binary y, test size=0.6)

multi_train_X, multi_val_X, multi_train_y, multi_val_y=train_test_split(to_fit, multi_y.

test size = 0.6)

binary model Random ForestClassifier()

binary_model.fit(binary_train_X, binary_train_y)

binary_predictions binary_model.predict(binary_val_X)

#calculate and display our base accuracty

base rf score accuracy_score(binary_predictions,binary_val_y)

base rf score

models = [

Random ForestClassifier().

LogisticRegression(max_iter=250),

KNeighbors Classifier(),

# an empty list to capture the performance of each model model_comps

# walk through the models and populate our list for model in models:

model name model. class name

accuracies cross_val_score(model, binary_train_X, binary_train_y, scoring accuracy) for count, accuracy in enumerate(accuracies):

model_comps.append((model_name, count, accuracy))

result_df=pd.DataFrame(model comps, columns=['model_name', 'count', 'accuracy']) result_df.pivot(index='count', columns='model_name', values'accuracy').boxplot(rot-45)

def add_predictions(data_set,predictions,y):

prediction_series-pd.Series(predictions, index=y.index)
#we need to add the predicted and actual outcomes to the data

predicted vs actual data sct.assign(predicted prediction series) original data

predicted vs actual assign(actualy).dropna() conf_matrix = confusion_matrix(original ix(original_data['actual"],

original_data['predicted'))

capture rows with failed predictions

base_errors = original_data[original_data['actual'] !=

original_data['predicted}}

#drop columns with no value

non_zeros base_errors.loc[:, (base_errors != 0).any(axis=0)]

#idetify the type of error

false_positives non_zeros.loc[non_zeros.actual -0]

false_negatives = non_zeros.loc[non_zeros.actual=1]

#put everything into an object

prediction_data = ('data': original_data,

'confusion_matrix': conf_matrix,

'errors': base errors,

'non zeros': non zeros,

'false_positives': false_positives,

'false_negatives': false_negatives)

return prediction_data

binary_prediction_data = add_predictions(df,

binary predictions,

binary_val_y)

# create a heatmap of the confusion matrix

sns.beatmap(data=binary_prediction_data['confusion_matrix'],

xticklabels = ['Predicted Normal', 'Predicted Attack'].

yticklabels ['Actual Normal', 'Actual Attack'],

cmap="YlGnBu",

fmt='d',

annot=True)
