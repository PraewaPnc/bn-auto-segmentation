from .data_encoding import *

# import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

from kneed import KneeLocator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector as selector
import plotly.express as px
import plotly.graph_objects as go

#-----------------------------------#
# Find Cluster
class AutoClustering():
    """
    A class to represent a segmentation

    ...

    Attributes
    ----------
    method : str
        method to define number of cluster    

    Methods
    -------
    find_cluster(X):
        Prints the number of clusters for k-means clustering.
    fit(X):
        
    predict(X):

    get_params():

    plot_elbow():

    
    """

  def __init__(self, method='auto',  **kwargs):
      self.method = method
      self.kwargs = kwargs

  def find_cluster(self, X):
    X = X.copy()

    clusters = range(2, 20)
    wcss = {}
    cluster = []
    score = []

    if self.method=='auto':
      for k in clusters:
          kmeans = KMeans(n_clusters=k, init="k-means++",  random_state=42)
          kmeans.fit(X)
          wcss[k] = kmeans.inertia_

      kn = KneeLocator(x=list(wcss.keys()),
                  y=list(wcss.values()),
                  curve='convex',
                  direction='decreasing')
      k = kn.knee

      self.wcss = wcss

    elif type(self.method) == int:
      k = self.method

    return k

  def fit(self, X):
    X = X.copy()
    self.k = self.find_cluster(X)

    self.model = KMeans(n_clusters=self.k, **self.kwargs, random_state=42)
    self.model.fit(X)

    return self

  def predict(self, X):
    cluster = self.model.predict(X)

    return cluster

  def get_params(self):
      return self.model.get_params()

  def plot_elbow(self):
    fig = px.line(x=list(self.wcss.keys()),
              y=list(self.wcss.values()),
              markers=True)

    fig.update_layout(width = 620,
                        height = 400,
                        title = 'The Elbow Method')

    fig.update_xaxes(title_text='the number of clusters(k)')
    fig.update_yaxes(title_text='Intra sum of distances(WCSS)')
    return fig

#   def plot_silho(self):
#     data = pd.DataFrame(list(zip(self.cluster, self.score)),
#                         columns =['n_clusters', 'silhouette_score'])

#     fig = px.line(data, x='n_clusters', y='silhouette_score',
#               markers=True)

#     fig.update_layout(width = 620,
#                         height = 400,
#                         title = 'The Silhouette Method')

#     fig.update_xaxes(title_text='the number of clusters(k)')
#     fig.update_yaxes(title_text='Silhouette Score')
#     return fig


#-----------------------------------#
# Visualization
class AutoResult():
    def __init__(self, data):
        self.data = data
        self.X = self.num_feature(self.data)
        self.C = self.cat_feature(self.data)
        self.summary = self.data.groupby('Cluster')[self.X].mean()
        self.summary_cat = self.data.groupby('Cluster')[self.C].agg(lambda x:x.value_counts().index[0])
        self.norm_sum = self._norm_sum(self.summary)


    def report_cluster(self, cluster):
        df = self.data[self.data['Cluster'] == cluster]
        return df

    def _norm_sum(self, summary):
        norm_sum = (summary - summary.min()) / (summary.max() - summary.min())
        return norm_sum

    def num_feature(self, X):
        num_features = X.select_dtypes(include='number').columns
        return num_features

    def cat_feature(self, X):
        cat_features = X.select_dtypes(exclude='number').columns
        return cat_features

    def num_heatmap(self):
        heatmap = self.summary.style.background_gradient()
        return heatmap

    def plot_radar_chart(self, n):
        radar_cluster = self.norm_sum.loc[n].to_frame().reset_index()
        radar_cluster.columns = ['theta','r']

        # radar plot
        fig = px.line_polar(radar_cluster, r='r', theta='theta', line_close=True)
        fig.update_layout(
            polar = dict(
                radialaxis = dict(range=[0, 1], visible=True)
            )
        )
        return fig

    def compare_radar_chart(self, list_n):
        fig = go.Figure()
        for n in list_n:
            radar_cluster = self.norm_sum.loc[n].to_frame().reset_index()
            radar_cluster.columns = ['theta','r']
            fig.add_trace(go.Scatterpolar(
                r=radar_cluster['r'],
                theta=radar_cluster['theta'],
                fill='toself',
                name=f'Cluster: {n}'
            ))
        fig.update_layout(
            polar = dict(radialaxis_tickfont_size = 15,
                angularaxis = dict(
                    tickfont_size = 15,
                    rotation = 90,
                    direction = "clockwise" )),
            width = 600,
            height = 600,
        )
        return fig

    def plot_cluster(self):
        count_clus = self.data['Cluster'].value_counts(normalize=True)
        fig = px.bar(count_clus)
        fig.update_layout(
            title="The total number of clusters",
            xaxis_title="Cluster",
            yaxis_title="Ratio",
            legend_title=None)

        return fig




