
import numpy as np
import networkx as nx
import scipy.sparse as sp

def get_folds_disconnected_G(edge_df):
    # Create NetworkX graph object from edge_df
    G = nx.from_pandas_edgelist(edge_df, "src", "tgt")

    # Getting all connected components
    connected_subgraphs = list(nx.connected_components(G))
    folds = {i: [] for i in range(5)}
    for graph in connected_subgraphs:
        G_sub = G.subgraph(graph)
        mst = nx.minimum_spanning_tree(G_sub)
        # Random seed for reproducibility (optional)
        srn = np.random.choice(list(mst.nodes))
        path = dict(nx.shortest_path_length(mst, source=srn))
        for node, length in path.items():
            folds[length % 5].append(node)

    return srn, folds, G, mst

def _csr_adjacency(G, nodelist=None):
    if nodelist is None:
        nodelist = sorted(G.nodes())
    return nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr")

def interpolate_X(X, G, folds, foldnum):
    """Vectorized neighbor interpolation for held-out nodes in `folds[foldnum]`.

    For each node i in the fold, replace row X[i,:] with the average of its
    neighbors outside the fold.
    """
    # G is a nx.Graph() object
    # X is the input to graphSVD() from graphsvd.py
    nodelist = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    A = _csr_adjacency(G, nodelist=nodelist)  # (n x n) CSR

    fold = np.array([node_to_idx[i] for i in folds[foldnum]], dtype=int)
    if fold.size == 0:
        return X

    mask = np.ones(A.shape[1], dtype=bool)
    mask[fold] = False
    A_nf = A[:, mask]
    # print(f"Type of A_nf: {type(A_nf)}")

    # Degrees of nodes in the fold restricted to non-fold neighbors
    deg = np.asarray(A_nf[fold, :].sum(axis=1)).ravel()
    deg[deg == 0] = 1.0  # guard
    # print(f"Type of deg: {type(deg)}")

    X_tilde = X.copy()
    # X_tilde = X.copy().tocsr()
    # print(f"Type of X: {type(X)}")
    X_tilde[fold, :] = (A_nf[fold, :] @ X_tilde[mask, :]) / deg[:, None]
    # X_tilde[fold, :] = (A_nf[fold, :] @ X[mask, :]) / deg[:, None]
    return X_tilde
