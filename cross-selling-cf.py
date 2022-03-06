# %%
from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from ast import literal_eval
from surprise import *
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
import streamlit as st
from PIL import Image
from collections import defaultdict

# %%

if 'total' not in st.session_state:
    st.session_state['total'] = 300


st.header("Telco Recs - Cross Selling Packages recommender")
st.write("Recommendations predicteed using collabarative Filtering approaches")

img=Image.open("data/fm-data/main.png")

st.image(img,width=800)


# %%
user_profile = pd.read_csv("data/User_Profile_Null_Handled.csv")

st.write(user_profile.head())

# %%
user_profile["Sub_Update_Status"].value_counts()

# %%
data=user_profile.iloc[:,[1,3,11,14,18,23,25,43,48,53,59,61,34]]
#data.rename(columns={"Sub_Type":"label"},inplace=True)
data = pd.get_dummies(data, prefix=['Sub_Update'], columns=['Sub_Update_Status'])
# data.drop('Sub_Update_Status', axis=1)

# Label encode class
# le = LabelEncoder()
# data['label'] = le.fit_transform(data.Sub_Type.values)
data = data.drop(['Sub_Update_NO_INFO'], axis=1)

# %%
data.fillna(0,inplace=True)
data_dim=data.iloc[:,[1,2,3,4,5,6,7,8,9,10,12,13]]

# %% [markdown]
# ## PCA Based rating calculation

# %%
pc=PCA(n_components=12) 
pc.fit(data_dim)

# %%
#How mucb variance, captured together
pc.explained_variance_ratio_.cumsum()

# %%
### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
pca = PCA(n_components=1)
pca.fit(data_dim)
reduced_data = pca.fit_transform(data_dim)
results_df = pd.DataFrame(reduced_data,columns=['ratings'])

# %%
# applying min-max-scaler to reduced features
scaler = MinMaxScaler()
results_df[['ratings']] = scaler.fit_transform(results_df[['ratings']])

# %%
data=pd.concat([data,results_df],axis=1)

# %%
data.rename(columns={"Sub_Type":"package"}, inplace=True)
data = data[["ACCOUNT_NUM.hash","package","ratings"]]
data

# %% [markdown]
# ## Interaction Matrix

# %%
# create a histogram of all the interactions by all the users present in the dataset
def create_interaction_matrix(data):
    interactions = data.groupby('ACCOUNT_NUM.hash').count()['package']
    plt.hist(interactions,bins=20)
    plt.show()

    # create the user item matrix using the ratings dataset - Hint: try using pivot function 
    interactions_metrix = data.pivot_table(index="ACCOUNT_NUM.hash", columns="package", values="ratings",aggfunc=np.sum)
    # replace all the missing values with zero
    return interactions_metrix.fillna(0)

# %%
create_interaction_matrix(data)

# %%
#X = interactions_metrix.values.T

# %% [markdown]
# ## Function definition

# %%
def train_test_splitter(data):
    reader = Reader(rating_scale=(0, 1))
    data_model = Dataset.load_from_df(data, reader)
    return data_model,train_test_split(data_model, test_size=.20)

# %%
def get_Iu(uid):
    """Return the number of items rated by given user
    
    Args:
        uid: The raw id of the user.
    Returns:
        The number of items rated by the user.
    """
    
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """Return the number of users that have rated given item
    
    Args:
        iid: The raw id of the item.
    Returns:
        The number of users that have rated the item.
    """
    
    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:  # item was not part of the trainset
        return 0

# %%
def get_top_n(predictions, n=10):
    
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:5]
        

    return top_n

# %%
# calculate NDCG
def ndcg(y_true, y_pred, k=None, powered=False):
    def dcg(scores, k=None, powered=False):
        if k is None:
            k = scores.shape[0]
        if not powered:
            ret = scores[0]
            for i in range(1, k):
                ret += scores[i] / np.log2(i + 1)
            return ret
        else:
            ret = 0
            for i in range(k):
                ret += (2 ** scores[i] - 1) / np.log2(i + 2)
            return ret
    
    ideal_sorted_scores = np.sort(y_true)[::-1]
    ideal_dcg_score = dcg(ideal_sorted_scores, k=k, powered=powered)
    
    pred_sorted_ind = np.argsort(y_pred)[::-1]
    pred_sorted_scores = y_true[pred_sorted_ind]
    dcg_score = dcg(pred_sorted_scores, k=k, powered=powered)
    
    return dcg_score / ideal_dcg_score

def ndcg1(y_true, y_pred, k=None):
    return ndcg(y_true, y_pred, k=k, powered=False)

def ndcg2(y_true, y_pred, k=None):
    return ndcg(y_true, y_pred, k=k, powered=True)

# %%
def top_n_pred(predictions):
    top_n = get_top_n(predictions, n=3)
    #print(top_n)
    users_est = defaultdict(list)
    users_true=defaultdict(list)
    rec_for_user=defaultdict(list)
    for uid, user_ratings in top_n.items():
        users_est[uid].append([est for (_, est,_) in user_ratings])
        users_true[uid].append([true_r for (_,_,true_r) in user_ratings])
        rec_for_user[uid].append([iid for (iid,_,_) in user_ratings])
    return top_n, users_est, users_true, rec_for_user

# %%
def calc_ndcg(users_true,users_est):
    ndcg_list=[]
    for uid in top_n:
        
        for i in users_true[uid]:
            y_true=np.asarray(i)#.reshape(-1,1)
        for i in users_est[uid]:
            y_pred=np.asarray(i)#.reshape(-1,1)
        
            ndcg_list.append(ndcg1(y_true, y_pred, k=None))

    ndcg_list = [i for i in ndcg_list if str(i) != 'nan']
    ndgc_rate = np.mean(ndcg_list)
    return ndcg_list, ndgc_rate

# %% [markdown]
# ## Collobarative Recommenders

# %%
data.info()

# %%
data = data[data["ratings"] > 0]

# %%
data_model, (trainset, testset) = train_test_splitter(data)

# %% [markdown]
# ## SVD Model

# %%
# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
svd_validate = cross_validate(algo, data_model, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# %%
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# %%
top_n, users_est, users_true, rec_for_user = top_n_pred(predictions)

# %%
ndcg_list, ndgc_rate = calc_ndcg(users_true,users_est)
print("NDCG", ndgc_rate)

# %%
# Let's build a pandas dataframe with all the predictions
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])    
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)

# %%
# 10 Best predictions
best_predictions = df.sort_values(by='err')[:10]
best_predictions

# %%
df.sort_values(by='err').to_csv("data/cross-selling/scored/SVD_Scored.csv")

# %%
rmse = accuracy.rmse(predictions)
print("SVD RMSE -->",rmse)
print("SVD Accuracy -->",1-rmse)

# %%
data_triplet = data.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
data_triplet.dropna(subset=["uid"],inplace=True)
data_triplet.drop("uid", axis=1, inplace = True)
data_triplet.rename(columns={"package":"Actual_Subscription","iid":"SVD_recommendation","err":"SVD_error"}, inplace = True)

# %% [markdown]
# ## SlopeOne

# %%
# We'll use the SlopeOne algorithm.
algo = SlopeOne()

# Run 5-fold cross-validation and print results
so_validate =cross_validate(algo, data_model, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# %%
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# %%
top_n, users_est, users_true, rec_for_user = top_n_pred(predictions)

# %%
ndcg_list, ndgc_rate = calc_ndcg(users_true,users_est)
print("NDCG", ndgc_rate)

# %%
# Let's build a pandas dataframe with all the predictions
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])    
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)

# %%
# 10 Best predictions
best_predictions = df.sort_values(by='err')[:10]
best_predictions

# %%
df.sort_values(by='err').to_csv("data/cross-selling/scored/SlopeOne_Scored.csv")

# %%
rmse = accuracy.rmse(predictions)
print("SlopeOne RMSE -->",rmse)
print("SlopeOne Accuracy -->",1-rmse)

# %%
data_triplet = data_triplet.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
data_triplet.dropna(subset=["uid"],inplace=True)
data_triplet.drop("uid", axis=1, inplace = True)
data_triplet.rename(columns={"iid":"SlopeOne_recommendation","err":"SlopeOne_error"}, inplace = True)

# %% [markdown]
# ## Matrix factorization (NMF)

# %%
# We'll use the SlopeOne algorithm.
algo = NMF()

# Run 5-fold cross-validation and print results
nmf_validate =cross_validate(algo, data_model, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# %%
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# %%
top_n, users_est, users_true, rec_for_user = top_n_pred(predictions)

# %%
ndcg_list, ndgc_rate = calc_ndcg(users_true,users_est)
print("NDCG", ndgc_rate)

# %%
# Let's build a pandas dataframe with all the predictions
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])    
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)

# %%
# 10 Best predictions
best_predictions = df.sort_values(by='err')[:10]
best_predictions

# %%
df.sort_values(by='err').to_csv("data/cross-selling/scored/MF_Scored.csv")

# %%
rmse = accuracy.rmse(predictions)
print("MF RMSE -->",rmse)
print("MF Accuracy -->",1-rmse)

# %%
data_triplet = data_triplet.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
data_triplet.dropna(subset=["uid"],inplace=True)
data_triplet.drop("uid", axis=1, inplace = True)
data_triplet.rename(columns={"iid":"MF_recommendation","err":"MF_error"}, inplace = True)

# %% [markdown]
# ## KNN with Means recommender

# %%
# We'll use the SlopeOne algorithm.
algo = KNNWithMeans()

# Run 5-fold cross-validation and print results
knn_validate = cross_validate(algo, data_model, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# %%
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# %%
top_n, users_est, users_true, rec_for_user = top_n_pred(predictions)

# %%
ndcg_list, ndgc_rate = calc_ndcg(users_true,users_est)
print("NDCG", ndgc_rate)

# %%
# Let's build a pandas dataframe with all the predictions
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])    
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)

# %%
# 10 Best predictions
best_predictions = df.sort_values(by='err')[:10]
best_predictions

# %%
df.sort_values(by='err').to_csv("data/cross-selling/scored/KNNMeans_Scored.csv")

# %%
rmse = accuracy.rmse(predictions)
print("KNN Means RMSE -->",rmse)
print("KNN Means Accuracy -->",1-rmse)

# %%
data_triplet = data_triplet.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
data_triplet.dropna(subset=["uid"],inplace=True)
data_triplet.drop("uid", axis=1, inplace = True)
data_triplet.rename(columns={"iid":"KNNMeans_recommendation","err":"KNNMeans_error"}, inplace = True)

# %% [markdown]
# ## Factorization Machines

# %%


# %%


# %% [markdown]
# ## Benchmark lgorithms

# %%
benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SlopeOne(), NMF()]:
    # Perform cross validation
    results = cross_validate(algorithm, data_model, measures=['RMSE'], cv=5, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
val_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
val_df

# %%
fig,ax = plt.subplots(figsize=(13,8))
ax.plot(so_validate["test_rmse"], color='blue')
ax.plot(svd_validate["test_rmse"], color='green')
ax.plot(knn_validate["test_rmse"], color='orange')
ax.plot(nmf_validate["test_rmse"], color='red')
ax.plot(so_validate["test_mae"], linestyle='dashdot', color='blue')
ax.plot(svd_validate["test_mae"], linestyle='dashdot', color='green')
ax.plot(knn_validate["test_mae"], linestyle='dashdot', color='orange')
ax.plot(nmf_validate["test_mae"], linestyle='dashdot', color='red')
# plt.xticks(np.arange(0, 30, 0.5))
plt.title("Boradband Packages Recommender", loc="center")
plt.legend(["RMSE: SlopeOne","RMSE: SVD","RMSE: KNNwithMeans","RMSE: NMF",
           "MAE: SlopeOne","MAE: SVD","MAE: KNNwithMeans","MAE: NMF"])

# %% [markdown]
# ## Best Accuracy Model

# %%
data_triplet.mean()

# %%
# Best Accuracy model - SlopeOne

# %% [markdown]
# ## Model Stacking approach (Ensambling)

# %% [markdown]
# ## Busines Rule filtration

# %%
data_triplet = data_triplet.merge(user_profile[["ACCOUNT_NUM.hash","Sub_Update_Date","Sub_Update_Status","Sub_Update"]], on ="ACCOUNT_NUM.hash", how ="left")

# %%
# Filter results based on Package owngrades
def play_rule(rec_packages, sub_state,sub_update):
    if (not sub_update == "NO_INFO") or (not sub_update == "NO INFO"):
        prev_package = sub_update.split("->")[0].replace(" ","")
        for pack in rec_packages:
            if sub_state == "Promotion Downgrade" and pack == prev_package:
                return "ERR"

data_triplet["Err_Rec"] = data_triplet.apply(lambda x: play_rule([x["SVD_recommendation"],x["SlopeOne_recommendation"],x["MF_recommendation"],x["KNNMeans_recommendation"]],x["Sub_Update_Status"],x["Sub_Update"]), axis =1 )

# %%
data_triplet = data_triplet.drop(["Sub_Update_Date","Sub_Update_Status","Sub_Update"],axis=1)
data_triplet.head()

# %% [markdown]
# ## Store resutls

# %%
data_triplet.to_csv("data/cross-selling/scored/CF_scored_and_eval.csv")

# %%



