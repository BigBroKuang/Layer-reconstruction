import argparse
import postpred as MP
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

plt.style.use('seaborn-whitegrid')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LayerReconstruction.")

    parser.add_argument('--true-network', type=str, default='MasterHVR.txt')
    parser.add_argument('--alpha', type=str, default='sim/alpha.txt')#input alpha
    parser.add_argument('--beta', type=str, default='sim/beta.txt')#input alpha
    
    parser.add_argument('--dimensions', type=int, default=50)#node vector dimension
    parser.add_argument('--directed', type=str, default=False)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--sim-layars', type=int, default=5)

    parser.add_argument('--rseed', type=int, default=1)
    parser.add_argument('--predict-layer', type=int, default=1)
    parser.add_argument('--decision', type=int, default=1)
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--N-real', type=int, default=10)
    parser.add_argument('--Inf', type=int, default=1e5)
    parser.add_argument('--maxit', type=int, default=1000)
    parser.add_argument('--err-max', type=float, default=0.00001)

    return parser.parse_args()

def post_process(true_net,source,target,node_list,lay):
    EdgePosterior =np.einsum('iz,jz->ij',source,target)
    ulist=node_list
    
    edgeList=[]
    gr = open(true_net,'r')
    EdgeTrue= list(csv.reader(gr,delimiter=' '))
    gr.close()
    for i in range(len(EdgeTrue)):
        if EdgeTrue[i][lay+2] in ['1']:
            edgeId1=ulist.index(EdgeTrue[i][1])
            edgeId2=ulist.index(EdgeTrue[i][2])            
            if edgeId1<edgeId2:
                edgeList.append((edgeId1,edgeId2))
            else:
                edgeList.append((edgeId2,edgeId1))
    
    predval_link=[]
    predval_nonlink=[]
    roc2a=[]
    roc1t=[]
    
    for j in range(EdgePosterior.shape[0]-1):
        for k in range(j+1,EdgePosterior.shape[1]):   
            roc2a.append(EdgePosterior[j,k])
            if (j,k) in edgeList or (k,j) in edgeList:
                predval_link.append(EdgePosterior[j,k])
                roc1t.append(1)
            else:
                predval_nonlink.append(EdgePosterior[j,k])
                roc1t.append(0)
    fpr, tpr, thresholds = roc_curve(roc1t, roc2a)
    print(roc_auc_score(roc1t, roc2a))

    
    
def main(args):
    mp= MP.postpred(K=args.dimensions,)
    mp.readgraph(adj=args.beta,alphafile=args.alpha,lay=args.predict_layer,nprior=args.sim_layers)
    mp.realization()
    post_process(true_net=args.true_network,source=mp.s,target=mp.t,node_list=mp.nl,lay=args.predict_layer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    

 
    
 
    
 
    

 
    
 
    
 
    
