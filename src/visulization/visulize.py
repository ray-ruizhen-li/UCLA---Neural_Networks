import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score


# Plot cluster method scatter chart
def plot_scatter(df):
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.show()

# Plot elbow chart
def elbow(df, start_range, end_range, *args):
    # try using a for loop
    k = range(start_range,end_range)
    K = []
    WCSS = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[column])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
    # Store the number of clusters and their respective WSS scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()

def silhouette(df, start_range, end_range, *args):
    # same as above, calculate sihouetter score for each cluster using a for loop
    k = range(start_range,end_range)
    K = []
    ss = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[column])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[column], ypred)
        K.append(i)
        ss.append(sil_score)
    # Store the number of clusters and their respective silhouette scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':ss})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()
    
def loss_curve(df):
    loss_values = df.loss_curve_

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()