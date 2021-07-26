import json
from flask import Flask, url_for, render_template, request, redirect, Response, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
from sklearn.manifold import MDS


full_data = pd.read_csv('C:/Users/Dell/Desktop/dataset for d3/merge_ds_2_numerical_new.csv')

def readData(file):
	data = pd.read_csv(file)
	return data

def find_pca(heading, data, columns, full_data):
	
	scaled_data = preprocessing.scale(data)
	components = 16
	pca = PCA(n_components=components)
	final_data = pca.fit_transform(scaled_data)
	final_data = MinMaxScaler().fit_transform(final_data)
	final_data = (final_data * 2) - 1
	dataElbow = final_data.copy()
	model = KMeans(n_clusters=3).fit(dataElbow)
	predictions = model.predict(dataElbow)
	full_data["cluster"] = predictions
    
	per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
	labels = [x for x in range(1, len(per_var)+1)]
	

	attributes = ['PC' + str(i+1) for i in range(components)]

	scatter = pd.DataFrame(final_data, columns=attributes)
	scatter.to_csv('scatter.csv', index=False)

	pca_components = pca.components_.T
	pca_components = MinMaxScaler().fit_transform(pca_components)
	pca_components = (pca_components * 2) - 1

	loadings = pd.DataFrame(pca_components, columns=attributes, index=columns)
	loadings["attributes"] = columns
	loadings.to_csv('loadings_csv.csv', index= False)
	full_data.to_csv('data_with_clusters.csv', index=False)
	pca_csv = pd.DataFrame(list(zip(labels, per_var)), columns=['PC_Number','Variance_Explained'])
	pca_csv.to_csv('pca.csv', index=False)
def find_mds(data,columns):

    scaled_data = preprocessing.scale(data)
    mds = MDS(n_components=2, dissimilarity='precomputed')
    final_data = MinMaxScaler().fit_transform(scaled_data)
    final_data = (final_data * 2) - 1
    final_data = pd.DataFrame(final_data)
    dmatrix_euc = pairwise_distances(final_data, metric='euclidean')
    mds_euc = mds.fit_transform(dmatrix_euc)
    mds_euc = pd.DataFrame(mds_euc, columns=['MDS1_euc', 'MDS2_euc'])
    dmatrix_cor = 1- final_data.corr()
    mds_cor = mds.fit_transform(dmatrix_cor)
    mds_cor = pd.DataFrame(mds_cor, columns=['MDS1_cor', 'MDS2_cor'])
    # mds_orig = pd.concat([mds_euc, mds_cor], axis=1)
    dataElbow = mds_euc.copy()
    model = KMeans(n_clusters=3).fit(dataElbow)
    predictions = model.predict(dataElbow)
    mds_euc["cluster"] = predictions
    mds_cor["attributes"] = columns
    
    mds_cor.to_csv('MDS_cor.csv', index=False)
    mds_euc.to_csv('MDS_euc.csv', index=False)
app = Flask(__name__)
# @app.route("/", methods = ['POST', 'GET'])
@app.route("/index.html")
def index():
    full_data = pd.read_csv('C:/Users/Dell/Desktop/dataset for d3/merge_ds_2_numerical_new.csv')
    # full_data = full_data[:550]
    full_data = full_data[:550]
    numeric = full_data._get_numeric_data()
    find_pca("original", numeric, numeric.columns, full_data)
    original = pd.read_csv('pca.csv')
    loadings = pd.read_csv('loadings_csv.csv')
    scatter = pd.read_csv('scatter.csv')
    data_with_clusters = pd.read_csv('data_with_clusters.csv')
    numeric_with_clusters = data_with_clusters._get_numeric_data()
    data = {}
    data['full_data'] = full_data.to_dict(orient='records')
    data['original'] = original.to_dict(orient='records')
    data['loadings'] = loadings.to_dict(orient='records')
    data['clustered_data'] = data_with_clusters.to_dict(orient='records')
    data['numeric_clustered_data'] = numeric_with_clusters.to_dict(orient='records')
    data['loadings_of_dataset'] = scatter.to_dict(orient='records')
    data['numeric'] = numeric.to_dict(orient='records')
    json_data = json.dumps(data)
    final_data = {'chart_data': json_data}
    return render_template("index.html", data=final_data)

@app.route("/MDS.html")
def mds_render():
    full_data = pd.read_csv('C:/Users/Dell/Desktop/dataset for d3/merge_ds_2_numerical_new.csv')
    # full_data = full_data[:550]
    full_data = full_data[:550]
    numeric = full_data._get_numeric_data()
    find_pca("original", numeric, numeric.columns, full_data)
    find_mds(numeric, numeric.columns)
    original = pd.read_csv('pca.csv')
    loadings = pd.read_csv('loadings_csv.csv')
    scatter = pd.read_csv('scatter.csv')
    data_with_clusters = pd.read_csv('data_with_clusters.csv')
    mdsdata_cor = pd.read_csv('MDS_cor.csv')
    mdsdata_euc = pd.read_csv('MDS_euc.csv')
    data = {}
    data['full_data'] = full_data.to_dict(orient='records')
    data['numeric'] = numeric.to_dict(orient='records')
    data['original'] = original.to_dict(orient='records')
    data['loadings'] = loadings.to_dict(orient='records')
    data['clustered_data'] = data_with_clusters.to_dict(orient='records')
    data['loadings_of_dataset'] = scatter.to_dict(orient='records')
    data['MDS_cor'] = mdsdata_cor.to_dict(orient='records')
    data['MDS_euc'] = mdsdata_euc.to_dict(orient='records')    
    json_data = json.dumps(data)
    final_data = {'chart_data': json_data}
    return render_template("MDS.html", data=final_data)

@app.route("/PCP.html")
def PCP_render():
    full_data = pd.read_csv('C:/Users/Dell/Desktop/dataset for d3/merge_ds_2_numerical.csv')
    # full_data = full_data[:550]
    full_data = full_data[:550]
    numeric = full_data._get_numeric_data()
    # find_pca("original", numeric, numeric.columns, full_data)
    # find_mds(numeric)
    # original = pd.read_csv('pca.csv')
    # loadings = pd.read_csv('loadings_csv.csv')
    scatter = pd.read_csv('scatter.csv')
    data_with_clusters = pd.read_csv('data_with_clusters.csv')
    numeric_with_clusters = data_with_clusters._get_numeric_data()
    
    # mdsdata = pd.read_csv('MDS_data.csv')
    data = {}
    data['full_data'] = full_data.to_dict(orient='records')
    data['numeric'] = numeric.to_dict(orient='records')
    # data['original'] = original.to_dict(orient='records')
    # data['loadings'] = loadings.to_dict(orient='records')
    data['clustered_data'] = data_with_clusters.to_dict(orient='records')
    data['loadings_of_dataset'] = scatter.to_dict(orient='records')
    data['numeric_clustered_data'] = numeric_with_clusters.to_dict(orient='records')
    # data['MDS_data'] = mdsdata.to_dict(orient='records')
    json_data = json.dumps(data)
    final_data = {'chart_data': json_data}
    return render_template("PCP.html", data=final_data)

if __name__ == "__main__":
	
	app.run(debug=True)
