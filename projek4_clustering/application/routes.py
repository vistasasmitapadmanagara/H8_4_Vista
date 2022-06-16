from application import app
from flask import Flask, render_template, request

# plotting
import numpy as np
import pandas as pd 
import json 
import plotly 
import plotly.express as px

# labelEncoder
from sklearn.preprocessing import LabelEncoder

# scaling data
from sklearn.preprocessing import RobustScaler

# PCA
from sklearn.decomposition import PCA

# elbow method
from yellowbrick.cluster import KElbowVisualizer

# modelling
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# fitur penting
from sklearn.ensemble import ExtraTreesClassifier

from math import sqrt

df = pd.read_csv('./credit-card-general-is-clean.csv')
df_cluster = pd.read_csv('./df_cluster.csv')

# Home
@app.route("/data")
@app.route("/")
def data():
    fig = px.histogram(df_cluster, x='cluster', color='cluster', title='Distribusi Segmentasi Pasar',
                color_discrete_sequence=px.colors.sequential.Plasma, template="plotly_white",
                labels={'cluster':'Segmentasi Pasar'}, histfunc='avg')

    fig.update_layout(xaxis_title="Segmentasi Pasar")

    graph1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('data.html', graph1=graph1) 

# Normalized
scaler = RobustScaler()
X = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2) 
reduced_X = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2'])

# elbow method
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)

# KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
model_fit = kmeans.fit(X)
sigmoid = pca.transform(model_fit.cluster_centers_)
cluster = kmeans.fit_predict(X)
reduced_X['cluster'] = cluster

# Conclusion
@app.route("/eda")
def eda():
    # Fitur Importances
    X = df_cluster.drop(['cluster'], axis=1)
    y = df_cluster['cluster']
    model = ExtraTreesClassifier()
    model.fit(X,y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=False).head(6)
    fig =  px.bar(feat_importances, orientation='h',
                template="plotly_white", title='Features Importance',
                color_discrete_sequence=px.colors.sequential.Plasma)
    graph_con_fit = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # distribusi chart
    feature = 'balance'
    balance = create_plot(feature)

    return render_template('eda.html', graph_con_fit=graph_con_fit, plot=balance)

def create_plot(feature):
    if feature == 'balance':
        bal = df_cluster['BALANCE']
        group_labels = ['BALANCE']
        target=df_cluster['cluster']
        data = px.histogram(x=target, y=bal, color=target, color_discrete_sequence=px.colors.sequential.Plasma,
                             template="plotly_white", histfunc='avg', labels={'color':'Segmentasi Pasar', 'x':'Balance'})
        data.update_layout(bargap=0.1, title_text='Balance Distribution')

    elif feature == 'cash_advance':
        ca = df_cluster['CASH_ADVANCE']
        group_labels = ['CASH_ADVANCE']
        target=df_cluster['cluster']
        data = px.histogram(x=target, y=ca, color=target, color_discrete_sequence=px.colors.sequential.Plasma,
                             template="plotly_white", histfunc='avg', labels={'color':'Segmentasi Pasar', 'x':'Cash Advance'})
        data.update_layout(bargap=0.1, title_text='Cash Advance Distribution')

    elif feature == 'cash_advance_freq':
        caf = df_cluster['CASH_ADVANCE_FREQUENCY']
        group_labels = ['CASH_ADVANCE_FREQUENCY']
        target=df_cluster['cluster']
        data = px.histogram(x=target, y=caf, color=target, color_discrete_sequence=px.colors.sequential.Plasma,
                             template="plotly_white", histfunc='avg', labels={'color':'Segmentasi Pasar', 'x':'Cash Advance Frequency'})
        data.update_layout(bargap=0.1, title_text='Cash Advance Frequency Distribution')

    elif feature == 'cash_advance_trx':
        catx = df_cluster['CASH_ADVANCE_TRX']
        group_labels = ['CASH_ADVANCE_TRX']
        target=df_cluster['cluster']
        data = px.histogram(x=target, y=catx, color=target, color_discrete_sequence=px.colors.sequential.Plasma,
                             template="plotly_white", histfunc='avg', labels={'color':'Segmentasi Pasar', 'x':'Cash Advance TRX'})
        data.update_layout(bargap=0.1, title_text='Cash Advance TRX Distribution')
    
    elif feature == 'min_pay':
        mp = df_cluster['MINIMUM_PAYMENTS']
        group_labels = ['MINIMUM_PAYMENTS']
        target=df_cluster['cluster']
        data = px.histogram(x=target, y=mp, color=target, color_discrete_sequence=px.colors.sequential.Plasma,
                             template="plotly_white", histfunc='avg', labels={'color':'Segmentasi Pasar', 'x':'Minimum Payments'})
        data.update_layout(bargap=0.1, title_text='Minimum Payments Distribution')

    else:
        purc = df_cluster['PURCHASES']
        group_labels = ['PURCHASES']
        target=df_cluster['cluster']
        data = px.histogram(x=target, y=purc, color=target, color_discrete_sequence=px.colors.sequential.Plasma,
                             template="plotly_white", histfunc='avg', labels={'color':'Segmentasi Pasar', 'x':'Purchases'})
        data.update_layout(bargap=0.1, title_text='Purchases Distribution')

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/plotting', methods=['GET', 'POST'])
def change_features():
    feature = request.args['selected']
    graphJSON= create_plot(feature)

    return graphJSON

# Conclusion
@app.route("/conclu")
def conclu():
    return render_template('conclu.html')

# Prediction
df['cluster'] = cluster
b = df['BALANCE']
ca = df['CASH_ADVANCE']
caf = df['CASH_ADVANCE_FREQUENCY']
catx = df['CASH_ADVANCE_TRX']
mpy = df['MINIMUM_PAYMENTS']
pur = df['PURCHASES']
features = list(zip(b, ca, caf, catx, mpy, pur))
label = df['cluster']

# Random Forest
rf = RandomForestClassifier()
rf.fit(features, label)

pred = rf.predict(features)

@app.route("/prediction",  methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        saldo = request.form["saldo"]
        tarik_tunai = request.form["tarik_tunai"]
        freq_tunai = request.form["freq_tunai"]
        freq_tunai_trx = request.form["freq_tunai_trx"]
        min_paym = request.form["min_paym"]
        purc = request.form["purc"]

        pred_list = pd.DataFrame([[saldo, tarik_tunai, freq_tunai, freq_tunai_trx, min_paym, purc]])
        sample = np.array(pred_list)
        sample_re = sample.reshape(1,-1)
        prediction = rf.predict(sample_re)

        output = {
            0 : "Termasuk Kelompok Pengguna Lumayan Sedikit",
            1 : "Termasuk Kelompok Pengguna Terbanyak",
            2 : "Termasuk Kelompok Pengguna Paling Sedikit",
            3 : "Termasuk Kelompok Pengguna Lumayan Banyak"
        }

        return render_template('prediction.html', prediction=output[prediction[0]])

    return render_template('prediction.html')