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
from sklearn.preprocessing import StandardScaler

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
scaler = StandardScaler()
X = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2) 
reduced_X = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2'])

@app.route("/modelling")
def modelling():
    fig = px.scatter(reduced_X, x='PC1', y='PC2', title='Principal Component Analysis (PCA)',
                color_discrete_sequence=px.colors.sequential.Plasma, template="plotly_white")

    graph2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('modelling.html', graph2=graph2)

# elbow method
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)

@app.route("/elbow")
def elbow():
    fig = px.line(x=K, y=distortions, markers="o",
                     title='Elbow Method',template="plotly_white",
                     color_discrete_sequence=px.colors.sequential.Plasma)

    fig.update_layout(xaxis_title='K values', yaxis_title='WCSS')

    graph_elbow = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('elbow.html', graph_elbow=graph_elbow)

# KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)

model_fit = kmeans.fit(X)
sigmoid = pca.transform(model_fit.cluster_centers_)

cluster = kmeans.fit_predict(X)
reduced_X['cluster'] = cluster

@app.route("/kmeans")
def kmeans():
    fig = px.scatter(reduced_X, x='PC1', y='PC2', color='cluster',
                     title='KMeans',template="plotly_white",
                     color_discrete_sequence=px.colors.sequential.Plasma)

    graph_kmeans = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('kmeans.html', graph_kmeans=graph_kmeans)

# Features Importance
@app.route("/fitur_penting")
def fitur_penting():
    X = df_cluster.drop(['cluster'], axis=1)
    y = df_cluster['cluster']

    model = ExtraTreesClassifier()
    model.fit(X,y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.sort_values(inplace=True)

    fig =  px.bar(feat_importances, orientation='h',
                template="plotly_white", title='Features Importance',
                color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(height=550)
    
    graph_fs = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('fitur_penting.html', graph_fs=graph_fs)

# Conclusion
@app.route("/conclu")
def conclu():
    fig =  px.histogram(df_cluster, x='cluster', color='cluster',
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        template="plotly_white", title='Distribusi Segmentasi Pasar',
                        labels={'cluster':'Segmentasi Pasar'}, histfunc='sum')

    graph_cl1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('conclu.html', graph_cl1=graph_cl1)

# Fitur Paling Berpengaruh
@app.route("/con_fitur")
def con_fitur():
    # Fitur Importances
    X = df_cluster.drop(['cluster'], axis=1)
    y = df_cluster['cluster']

    model = ExtraTreesClassifier()
    model.fit(X,y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=False).head(5)

    fig =  px.bar(feat_importances, orientation='h',
                template="plotly_white", title='Features Importance',
                color_discrete_sequence=px.colors.sequential.Plasma)
    
    graph_con_fit = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('con_fitur.html', graph_con_fit=graph_con_fit)

# Frekuensi Pembelian
@app.route("/purc_freq")
def purc_freq():
    fig =  px.histogram(df_cluster, x='cluster', y=['PURCHASES_FREQUENCY'], color='cluster',
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        template="plotly_white", title='Rata-Rata Frekuensi Pembelian',
                        labels={'cluster':'Segmentasi Pasar',
                                'PURCHASES_FREQUENCY':'Frekuensi Pembelian'}, histfunc='avg')

    graph_purc = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('purc_freq.html', graph_purc=graph_purc)

# Frekuensi Cicilan Pembelian
@app.route("/inst_purc_freq")
def inst_purc_freq():
    fig =  px.histogram(df_cluster, x='cluster', y='PURCHASES_INSTALLMENTS_FREQUENCY', color='cluster',
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        template="plotly_white", title='Rata-Rata Frekuensi Cicilan Pembelian',
                        labels={'cluster':'Segmentasi Pasar',
                                'PURCHASES_INSTALLMENTS_FREQUENCY':'Frekuensi Cicilan Pembelian'}, histfunc='avg')

    graph_inst_purc = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('inst_purc_freq.html', graph_inst_purc=graph_inst_purc)

# Frekuensi Penarikan Tunai
@app.route("/cash_ad_freq")
def cash_ad_freq():
    fig =  px.histogram(df_cluster, x='cluster', y='CASH_ADVANCE_FREQUENCY', color='cluster',
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        template="plotly_white", title='Rata-Rata Frekuensi Penarikan Tunai',
                        labels={'cluster':'Segmentasi Pasar',
                                'CASH_ADVANCE_FREQUENCY':'Frekuensi Penarikan Tunai'}, histfunc='avg')

    graph_cash_freq = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('cash_ad_freq.html', graph_cash_freq=graph_cash_freq)

# Sisa Saldo
@app.route("/balance")
def balance():
    fig =  px.histogram(df_cluster, x='cluster', y='BALANCE', color='cluster',
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        template="plotly_white", title='Rata-Rata Sisa Saldo',
                        labels={'cluster':'Segmentasi Pasar',
                                'BALANCE':'Sisa Saldo'}, histfunc='avg')

    graph_balance = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('balance.html', graph_balance=graph_balance)

# Penarikan Tunai
@app.route("/cash_ad")
def cash_ad():
    fig =  px.histogram(df_cluster, x='cluster', y='CASH_ADVANCE', color='cluster',
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        template="plotly_white", title='Rata-Rata Penarikan Tunai',
                        labels={'cluster':'Segmentasi Pasar',
                                'CASH_ADVANCE':'Penarikan Tunai'}, histfunc='avg')

    graph_cash_ad = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('cash_ad.html', graph_cash_ad=graph_cash_ad)

# Prediction
df['cluster'] = cluster

pif = df['PURCHASES_INSTALLMENTS_FREQUENCY']
pf = df['PURCHASES_FREQUENCY']
caf = df['CASH_ADVANCE_FREQUENCY']
b = df['BALANCE']
ca = df['CASH_ADVANCE']
features = list(zip(pif, pf, caf, b, ca))
label = df['cluster']

# Random Forest
rf = RandomForestClassifier()
rf.fit(features, label)

pred = rf.predict(features)

@app.route("/prediction",  methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        freq_beli = request.form["freq_beli"]
        freq_beli_cicilan = request.form["freq_beli_cicilan"]
        freq_tunai = request.form["freq_tunai"]
        saldo = request.form["saldo"]
        tarik_tunai = request.form["tarik_tunai"]

        pred_list = pd.DataFrame([[freq_beli, freq_beli_cicilan, freq_tunai, saldo, tarik_tunai]])
        sample = np.array(pred_list)
        sample_re = sample.reshape(1,-1)
        prediction = rf.predict(sample_re)

        output = {
            0 : "Segmentasi Pasar: Besar",
            1 : "Segmentasi Pasar: Cukup Besar",
            2 : "Segmentasi Pasar: Cukup Kecil",
            3 : "Segmentasi Pasar: Kecil"
        }

        return render_template('prediction.html', prediction=output[prediction[0]])

    return render_template('prediction.html')