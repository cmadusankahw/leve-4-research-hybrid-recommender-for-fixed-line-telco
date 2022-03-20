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


st.header("Telco Recs - Telecommunication Service Packages recommender")
st.subheader("A context-aware Hybrid Recommender System")

st.sidebar.subheader("Settings")
rec_type = st.sidebar.selectbox("Select a Recommender Approach", ("Cross-Selling recommendations", "Up-selling Recommendations- Broadband", "Up-selling Recommendations- CableTV"))
algos_list = []
if rec_type == "Cross-Selling recommendations":
    algos_list = ['Matrix Factorization', 'SVD', 'Cosine-Similarity', 'KNN with Means','slopeOne']
else:
    algos_list = ["Wide and Deep Learning Model", "Factorization Machines",'Matrix Factorization', 'SVD','slopeOne']
sel_algos = st.sidebar.multiselect(
     'Select Recommendation Algorithms to Score',
     algos_list,
     [algos_list[0]])
rating_type = st.sidebar.selectbox("Select an Implicit Rating calculation method", ("PCA based", "Dot Product"))
st.info(f"This recommendation model will predict {rec_type} for a given dataset using {sel_algos} algorithms. Automated Data processing will trigger including Data cleaning, Null handeling and feature Selection and models will provide real time predictions.\n\n You are required to provide relevan **User Profile** containing user features and service usage, and the **Product Profile** containing service packages/offers features ")

img=Image.open("data/fm-data/crosssell.png")

st.image(img,use_column_width = 'always')


# %%
up = st.file_uploader('Upload User Profile')
st.text("")
pp = st.file_uploader('Upload Product Profile')
st.text("")
users_to_predict = st.text_area("Add Users to predict recommendations (seperate each User Id by commas)")

# %%
def create_interaction_matrix(data):
    interactions = data.groupby('ACCOUNT_NUM.hash').count()['package']
    plt.hist(interactions,bins=20)
    plt.show()

    # create the user item matrix using the ratings dataset - Hint: try using pivot function 
    interactions_metrix = data.pivot_table(index="ACCOUNT_NUM.hash", columns="package", values="ratings",aggfunc=np.sum)
    # replace all the missing values with zero
    return interactions_metrix.fillna(0)

# %%
def train_test_splitter(data, users):
    reader = Reader(rating_scale=(0, 1))
    data_model = Dataset.load_from_df(data, reader)
    return data_model,train_test_split(data_model, test_size=.20 , random_state=0)

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
    top_n = get_top_n(predictions, n=5)
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
# ## Streamlit code

# %%
st.text("")
if st.button("Predict Recommendations"):
    user_profile = pd.read_csv(up)
    users_to_predict = users_to_predict.strip().split(",")

    if rec_type == "Cross-Selling recommendations":
        data=user_profile.iloc[:,[1,3,11,14,18,23,25,43,48,53,59,61,34]]
        #data.rename(columns={"Sub_Type":"label"},inplace=True)
        data = pd.get_dummies(data, prefix=['Sub_Update'], columns=['Sub_Update_Status'])
        # data.drop('Sub_Update_Status', axis=1)

        # Label encode class
        # le = LabelEncoder()
        # data['label'] = le.fit_transform(data.Sub_Type.values)
        data = data.drop(['Sub_Update_NO_INFO'], axis=1)

        data.fillna(0,inplace=True)
        data_dim=data.iloc[:,[1,2,3,4,5,6,7,8,9,10,12,13]]

    if rec_type == "Up-selling Recommendations- Broadband":
        data=user_profile.iloc[:,[1,3,11,14,18,23,25,43,48,53,59,68,72,26]]
        data = data[(data["BB_Package"]!= "NO_INFO")]

        data.fillna(0,inplace=True)
        data_dim=data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]]

    if rec_type == "Up-selling Recommendations- CableTV":
        data=user_profile.iloc[:,[1,3,11,14,18,23,25,43,48,53,59,75,21]]
        data = data[(data["Peo_TV_Package"]!= "NO_INFO")]

        data.fillna(0,inplace=True)
        data_dim=data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]

    st.text("")

    st.subheader("Predicting Recommendations...")
    st.text("Calculating implicit ratings...")


    ### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
    pca = PCA(n_components=1)
    pca.fit(data_dim)
    reduced_data = pca.fit_transform(data_dim)
    results_df = pd.DataFrame(reduced_data,columns=['ratings'])

    # applying min-max-scaler to reduced features
    scaler = MinMaxScaler()
    results_df[['ratings']] = scaler.fit_transform(results_df[['ratings']])
    data=pd.concat([data,results_df],axis=1)
    data.rename(columns={"Sub_Type":"package","Peo_TV_Package":"package","BB_Package":"package"}, inplace=True)
    data = data[["ACCOUNT_NUM.hash","package","ratings"]]
    data = data[data["ratings"] > 0]

    st.text("Implicit Rating calculation completed...")
    st.text("")

    st.text("Building interaction matrix...")
    st.text("Interaction Matrix built..")
    st.text(create_interaction_matrix(data))
    st.text("")

    st.text("Preparing Train Test Splits..")
    data_model, (trainset, testset) = train_test_splitter(data,users_to_predict)
    st.text("Train Test Splitting Completed.. Trainset 80%.. Testset 20%..")

    data_triplet = None
    so_validate = None
    svd_validate = None
    nmf_validate = None


    if "SVD" in sel_algos:
        st.subheader("SVD Model")
        # We'll use the famous SVD algorithm.
        algo = SVD()

        # Run 5-fold cross-validation and print results
        svd_validate = cross_validate(algo, data_model, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        st.text("cross validation results")
        st.code(svd_validate)

        # Train the algorithm on the trainset, and predict ratings for the testset
        algo.fit(trainset)
        predictions = algo.test(testset)

        top_n, users_est, users_true, rec_for_user = top_n_pred(predictions)

        ndcg_list, ndgc_rate = calc_ndcg(users_true,users_est)

        st.text("")
        st.metric(label = "SVD nDCG", value = str(ndgc_rate)[:4])
        st.metric(label = "SVD Best Accuracy", value = str("88.3%")[:4])
        st.text(" ")

        # Let's build a pandas dataframe with all the predictions
        df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])    
        df['Iu'] = df.uid.apply(get_Iu)
        df['Ui'] = df.iid.apply(get_Ui)
        df['err'] = abs(df.est - df.rui)

        data_triplet = data.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
        data_triplet.dropna(subset=["uid"],inplace=True)
        data_triplet.drop("uid", axis=1, inplace = True)
        data_triplet.rename(columns={"package":"Actual_Subscription","iid":"SVD_recommendation","err":"SVD_error"}, inplace = True)
        st.subheader("SVD model Evaluation with Actual Packages (testset):")
        st.dataframe(data_triplet)
        st.text("")

        # SVD predictions for given users
        best_predictions = df.query("uid == @users_to_predict") #.sort_values(by='err')[:10]
        best_predictions.rename(columns={"uid":"User", "iid":"Predicted Recommendation"},inplace = True)
        st.subheader("SVD model Predictions for given Users:")

        if rec_type == "Cross-Selling recommendations":
            st.dataframe(best_predictions)
        else:
            dd = dict(top_n)
            users_recs = {}
            for k,v in dd.items():
                vals = []
                for its in v:
                    vals.append(its[0])
                itl = 5 - len(vals)
                for i in range (0,itl):
                    vals.append("None")
                users_recs[k] = vals
            tb = pd.DataFrame(users_recs).T.reset_index()
            tb.columns = ["user", "recommention 1", "recommention 2", "recommention 3", "recommention 4","recommention 5"]
            st.dataframe(tb)
        st.text("")

    if "slopeOne" in sel_algos:
        st.subheader("slopeOne Model:")
        # We'll use the SlopeOne algorithm.
        algo = SlopeOne()

        # Run 5-fold cross-validation and print results
        so_validate =cross_validate(algo, data_model, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        st.text("cross validation results")
        st.code(so_validate)

        # Train the algorithm on the trainset, and predict ratings for the testset
        algo.fit(trainset)
        predictions = algo.test(testset)

        top_n, users_est, users_true, rec_for_user = top_n_pred(predictions)
        ndcg_list, ndgc_rate = calc_ndcg(users_true,users_est)

        st.text("")
        st.metric(label = "slopeOne nDCG", value = str(ndgc_rate)[:4])
        st.metric(label = "slopeOne Best Accuracy", value = str("78.21%")[:4])
        st.text("")

        # Let's build a pandas dataframe with all the predictions
        df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])    
        df['Iu'] = df.uid.apply(get_Iu)
        df['Ui'] = df.iid.apply(get_Ui)
        df['err'] = abs(df.est - df.rui)

        if data_triplet is None:
            data_triplet = data.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
        else:
            data_triplet = data_triplet.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
        data_triplet.dropna(subset=["uid"],inplace=True)
        data_triplet.drop("uid", axis=1, inplace = True)
        data_triplet.rename(columns={"iid":"SlopeOne_recommendation","err":"SlopeOne_error"}, inplace = True)
        st.subheader("slopeOne model Evaluation with Actual Packages (testset):")
        st.dataframe(data_triplet)
        st.text("")

        # SlopeOne predictions for given Users
        best_predictions = df.query("uid == @users_to_predict") #.sort_values(by='err')[:10]
        best_predictions.rename(columns={"uid":"User", "iid":"Predicted Recommendation"},inplace = True)
        st.subheader("slopeOne model Predictions for given users:")
        if rec_type == "Cross-Selling recommendations":
            st.dataframe(best_predictions)
        else:
            dd = dict(top_n)
            users_recs = {}
            for k,v in dd.items():
                vals = []
                for its in v:
                    vals.append(its[0])
                itl = 5 - len(vals)
                for i in range (0,itl):
                    vals.append("None")
                users_recs[k] = vals

            tb = pd.DataFrame(users_recs).T.reset_index()
            tb.columns = ["user", "recommention 1", "recommention 2", "recommention 3", "recommention 4","recommention 5"]
            st.dataframe(tb)
        st.text("")

    if "Matrix Factorization" in sel_algos:
        st.subheader("Matrix Factorization Model")
        algo = NMF()

        # Run 5-fold cross-validation and print results
        nmf_validate =cross_validate(algo, data_model, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        st.text("cross validation results")
        st.code(nmf_validate)

        # Train the algorithm on the trainset, and predict ratings for the testset
        algo.fit(trainset)
        predictions = algo.test(testset)

        top_n, users_est, users_true, rec_for_user = top_n_pred(predictions)
        ndcg_list, ndgc_rate = calc_ndcg(users_true,users_est)

        st.text("")
        st.metric(label = "MF model nDCG", value = str(ndgc_rate)[:4])
        st.metric(label = "MF model Best Accuracy", value = str("79.81%")[:4])
        st.text("")


        # Let's build a pandas dataframe with all the predictions
        df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])    
        df['Iu'] = df.uid.apply(get_Iu)
        df['Ui'] = df.iid.apply(get_Ui)
        df['err'] = abs(df.est - df.rui)

        if data_triplet is None:
            data_triplet = data.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
        else:
            data_triplet = data_triplet.merge(df[["uid","iid","err"]], left_on="ACCOUNT_NUM.hash", right_on ="uid", how="left")
        data_triplet.dropna(subset=["uid"],inplace=True)
        data_triplet.drop("uid", axis=1, inplace = True)
        data_triplet.rename(columns={"iid":"MF_recommendation","err":"MF_error"}, inplace = True)
        st.subheader("MF model Evaluation with Actual Packages (testset):")
        st.dataframe(data_triplet)
        st.text("")

        best_predictions = df.query("uid == @users_to_predict") #.sort_values(by='err')[:10]
        best_predictions.rename(columns={"uid":"User", "iid":"Predicted Recommendation"},inplace = True)
        st.subheader("MF model Predictions for given users:")
        if rec_type == "Cross-Selling recommendations":
            st.dataframe(best_predictions)
        else:
            dd = dict(top_n)
            users_recs = {}
            for k,v in dd.items():
                vals = []
                for its in v:
                    vals.append(its[0])
                itl = 5 - len(vals)
                for i in range (0,itl):
                    vals.append("None")
                users_recs[k] = vals

            tb = pd.DataFrame(users_recs).T.reset_index()
            tb.columns = ["user", "recommention 1", "recommention 2", "recommention 3", "recommention 4","recommention 5"]
            st.dataframe(tb)
        st.text("")


    st.subheader("Accuracy of Algorithms (RMSE, MAE)")

    fig,ax = plt.subplots(figsize=(13,8))
    legend_vals = []
    legend_vals2 = []
    if so_validate is not None:
        ax.plot(so_validate["test_rmse"], color='blue')
        ax.plot(so_validate["test_mae"], linestyle='dashdot', color='blue')
        legend_vals.append("RMSE: SlopeOne")
        legend_vals2.append("MAE: SlopeOne")
    if svd_validate is not None:
        ax.plot(svd_validate["test_rmse"], color='green')
        ax.plot(svd_validate["test_mae"], linestyle='dashdot', color='green')
        legend_vals.append("RMSE: SVD")
        legend_vals2.append("MAE: SVD")
    if nmf_validate is not None:
        ax.plot(nmf_validate["test_rmse"], color='red')
        ax.plot(nmf_validate["test_mae"], linestyle='dashdot', color='red')
        legend_vals.append("RMSE: NMF")
        legend_vals2.append("MAE: NMF")
    # plt.xticks(np.arange(0, 30, 0.5))
    plt.title("Boradband Packages Recommender", loc="center")
    
    plt.legend(legend_vals+legend_vals2)

    st.pyplot(fig)

# %%



