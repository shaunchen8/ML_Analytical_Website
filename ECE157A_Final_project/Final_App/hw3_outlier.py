import mpld3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import matplotlib.colors as colors
from matplotlib.patches import Patch

selected_features = ['PTS','TRB','AST']

def load_clean_normed_data(filename):
    data = pd.read_csv(filename)[['Player']+selected_features]
    for stat in selected_features:
        data[stat]=data[stat]/data[stat].max()
    return data

def train_one_class_svm(data):
    from sklearn.svm import OneClassSVM
    return OneClassSVM(kernel='rbf', nu= 0.05).fit(data[selected_features])

def train_elliptic_envelope(data):
    from sklearn.covariance import EllipticEnvelope
    return EllipticEnvelope(contamination=0.0367,random_state=42).fit(data[selected_features])

def train_isolation_forest(data):
    from sklearn.ensemble import IsolationForest
    return IsolationForest(contamination=0.0367,random_state=42).fit(data[selected_features])

def run_algo3(file_path):

    data = load_clean_normed_data(file_path)

    #clf = train_one_class_svm(data)
    clf = train_elliptic_envelope(data)
    #clf = train_isolation_forest(data)

    scores = clf.decision_function(data[selected_features])
    #neg is outlier, pos is in range

    topthreeIndices = np.argsort(scores)[:3]
    topthree = data.iloc[topthreeIndices]

    #print(topthree)

    outliers = scores < 0
    outliersSansTopThree = outliers.copy()
    outliersSansTopThree[topthreeIndices] = False

    #fig,ax = plt.subplots(3,3,sharex=True,squeeze=False)

    #for rownum, (_,obj) in enumerate(topthree.iterrows()):
    #    for colnum, stat in enumerate(selected_features):
    #        hist, bin_edges = np.histogram(data[stat],bins=100)
    #        cdf = np.cumsum(hist)/data.shape[0]
    #        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    #        ax[rownum,colnum].plot(bin_centers,cdf)
    #        ax[rownum,colnum].set_title(obj['Player']+' on '+stat)
    #        ax[rownum,colnum].plot([obj[stat],obj[stat]],[0.0,1.0])

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    inliers = ax.scatter(data.iloc[~outliers][selected_features[0]],
            data.iloc[~outliers][selected_features[1]],
            data.iloc[~outliers][selected_features[2]])
    outliers = ax.scatter(data.iloc[outliersSansTopThree][selected_features[0]],
            data.iloc[outliersSansTopThree][selected_features[1]],
            data.iloc[outliersSansTopThree][selected_features[2]])
    topthree = ax.scatter(topthree[selected_features[0]],
            topthree[selected_features[1]],
            topthree[selected_features[2]])
    ax.set_xlabel(data.columns[1])
    ax.set_ylabel(data.columns[2])
    ax.set_zlabel(data.columns[3])
    plt.title(label="Outlier Detection of 2019-2020 NBA Players using Elliptic Envelope")
    ax.legend(handles=[inliers, outliers, topthree], labels=['Inliers','Outliers','Top Three Outliers'], loc="upper left", title="Legend")
    #plt.show()

    with io.StringIO() as stringbuffer:
        fig.savefig(stringbuffer,format='svg')
        svgstring = stringbuffer.getvalue()
    return svgstring


#run_algo3('D:/Documents - Data Drive/VSCode/ECE157A_Final_Project/Final_Project/media/upload/nba_players_stats_19_20_per_game.csv')
#plt.show()   