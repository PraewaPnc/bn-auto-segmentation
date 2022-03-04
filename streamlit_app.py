
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from botnoi_autosegment.model import AutoClustering, AutoResult
from botnoi_autosegment.data_encoding import ColumnPreprocessor
from sklearn.compose import make_column_selector as selector


#-----------------------------------#
# Page layout
st.set_page_config(page_title='The Segmentation App',
    layout='centered')

#------------------------------------#

st.write("""
# The Auto Segmentation App
In this implementation, the *KMeans()* function is used this app for build a clustering model using the **KMeans** algorithm.
Try uploading the file and adjusting the hyperparameter!
""")

#-----------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/PraewaPnc/auto-segmentation/ba5138f38ec21a2c1f0225d039584d673efb49ca/small_titanic.csv)
""")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        drop_col = st.sidebar.multiselect('Drop columns', df.columns.tolist())
    else:
        st.info('Awaiting for CSV file to be uploaded. **!!!**')

# Transform options
with st.sidebar.header('2. Transform Data'):
    st.sidebar.subheader('Fill missing values')
    fill_numerical = st.sidebar.selectbox('Fill numerical features', ['mean','most_frequent','median','constant'], index=0)
    num_value = None
    if fill_numerical == 'constant':
        num_value = st.sidebar.number_input('Constant value for filling in numerical features', value=0)
    fill_categorical = st.sidebar.selectbox('Fill categorical features', ['most_frequent','constant'], index=0)
    cat_value = None
    if fill_categorical == 'constant':
        cat_value = st.sidebar.text_input('Constant value for filling in categorical features', value='NoData')

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy=fill_numerical, fill_value=num_value)),
                                ('scaler', MinMaxScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy=fill_categorical, fill_value=cat_value)),
                                ('encode', OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnPreprocessor([('num', num_transformer, selector(dtype_include='number')),
                                            ('cat', cat_transformer, selector(dtype_exclude='number'))])

# Select method for choose n_clusters
with st.sidebar.subheader('3. Select number of cluster'):
    # max_cluster = st.sidebar.slider('Maximum number of clusters', 1, 100, 10, 1)
    method = st.sidebar.selectbox('Method', ['auto', 'manual'])
    if method == 'manual':
        method = st.sidebar.number_input('Number of clusters (n_clusters)', value=2)
    else:
        method = 20

# Sidebar - Specify parameter settings
# with st.sidebar.subheader('4. Set Parameters in **KMeans**'):
#     init = st.sidebar.selectbox('Method for initialization (init)', ['k-means++', 'random'])
#     n_init = st.sidebar.slider('Initial cluster centroids (n_init)', 1, 10, 10, 1)
#     max_iter = st.sidebar.slider('Maximum number of iterations (max_iter)', 1, 2000, 300, 1)
#     verbose = st.sidebar.slider('Verbosity mode (verbose)', 1, 10, 0, 1)
#     random_state = st.sidebar.slider('Random State (random_state)', 1, 100, 42, 1)
#     algorithm = st.sidebar.selectbox('K-means algorithm to use (algorithm)', ['auto', 'full', 'elkan'])

with st.sidebar.subheader('4. Dimension Reduction [Optional]'):
    pca = st.sidebar.checkbox('PCA')
    if pca:
        n_components = st.sidebar.number_input('Number of Dimension (n_components)', value=3)


#-----------------------------------#
# Main panel

if uploaded_file:
# Displays the dataset
    st.subheader('Dataset')

    autoClustering = AutoClustering(method=method)

    X = df.drop(columns=drop_col, axis=1)
    st.markdown('**Glimpse of dataset**')
    st.write(X.head(5))
    st.write('Shape of Dataset')
    st.info(X.shape)
    st.write('Features for clustering')
    st.info(X.columns.to_list())

    transformed_X = preprocessor.fit_transform(X)
    if pca:
        pca = PCA(n_components=n_components, random_state=42)
        pca_X = pca.fit_transform(transformed_X)
        autoClustering.fit(pca_X)
        clusters = autoClustering.predict(pca_X)
        number_clusters = autoClustering.find_cluster(pca_X)
    else:
        autoClustering.fit(transformed_X)
        clusters = autoClustering.predict(transformed_X)
        number_clusters = autoClustering.find_cluster(transformed_X)

    df['Cluster'] = clusters + 1

    st.subheader('Clustering Result')

    if method == 'auto':
        fig = autoClustering.plot_elbow()
        st.plotly_chart(fig)
    # elif method == 'silhouette':
    #     fig = autoClustering.plot_silho()
    #     st.plotly_chart(fig)
    else:
        pass

    st.write('Optimal Number of Clusters')
    st.info(number_clusters)
    # st.write('Predict cluster index for each sample.')
    # st.info(clusters + 1)

    st.markdown('Model Parameters')
    st.write(autoClustering.get_params())

    st.subheader('Dataset with Cluster')
    st.write(df)

    def convert_df(df):
       return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
       "Press to Download",
       csv,
       "file.csv",
       "text/csv",
       key='download-csv'
    )

    #--------------------------#
    autoResult = AutoResult(data=df)

    # Visualization
    st.subheader('Visualization')
    # plot the total number of cluster
    pc = autoResult.plot_cluster()
    st.plotly_chart(pc)

    list_clust = st.multiselect('Select cluster', df.Cluster.unique().tolist(), default=[1])

    # Compare 2 clusters
    plot = autoResult.compare_radar_chart(list_n=list_clust)
    st.plotly_chart(plot)

    #---------------------------#

    st.subheader('Cluster Summary')

    cluster = st.selectbox('Select Cluster', df.Cluster.unique().tolist())
    st.write(f'Dataset of Cluster {cluster}')
    report = autoResult.report_cluster(cluster=cluster)
    st.write(report)

    csv = convert_df(report)

    st.download_button(
       "Press to Download",
       csv,
       f"Cluster{cluster}.csv",
       "text/csv",
       key='download-csv'
    )

    #----------------------------#

    ht = autoResult.num_heatmap()
    st.write('Mean of Numerical feature by Cluster')
    st.write(ht)
    st.write('Mode of Categorical feature by Cluster')
    st.write(autoResult.summary_cat)