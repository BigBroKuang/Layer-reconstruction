import time
import numpy as np
import csv
import copy


class postpred():
    def __init__(self,directed=False,K=1,N=1,nl=[],rseed=1,prior=True,
                 decision=1,tolerance=10,N_real=1,inf=1e5,maxit=1000,err_max=0.00001,savefile='123.txt'):
        self.N=N           ## nodes 
        self.K=K
        self.nl=nl  #node list
        self.rseed=rseed
        self.decision =decision
        self.directed =directed
        self.inf=inf
        self.maxit=maxit
        self.N_real=N_real
        self.err_max=err_max
        self.tolerance=tolerance
        self.savefile=savefile
        self.prior=prior
        
        self.s = []
        self.t = []
        self.A = np.zeros((self.N,self.N),dtype=float)
        self.beta=[]
        self.w=[]
    def readgraph(self,adj='adjacency.txt',alphafile='alpha.txt',lay=1,nprior=5):
        nw = open(adj,'r')
        nwl = list(csv.reader(nw,delimiter=' '))
        nw.close()
        for ele in nwl:
            if ele[1] not in self.nl:
                self.nl.append(ele[1])
            if ele[2] not in self.nl:
                self.nl.append(ele[2])
        self.N = len(self.nl)
        numedge=0
        self.A = np.zeros((self.N,self.N),dtype=float)
        self.s = np.zeros((self.N,self.K),dtype=float)  #degree of node in each layer
        self.t = np.zeros((self.N,self.K),dtype=float)
        
        for i in range(len(nwl)):
            idx,idy=self.nl.index(nwl[i][1]),self.nl.index(nwl[i][2])
            dl=float(nwl[i][lay+2])
            numedge+=dl
            self.A[idx,idy]=dl
            self.A[idy,idx]=dl
        
        self.beta = np.zeros((self.N,self.N),dtype=float)
        for i in range(self.beta.shape[0]):
            for j in range(self.beta.shape[1]):
                self.beta[i,j]=nprior*2
                    
        if not self.prior:
            for i in range(self.beta.shape[0]):
                for j in range(self.beta.shape[1]):
                    self.beta[i,j]=1
        else:
            nw = open(alphafile,'r')
            bf = list(csv.reader(nw,delimiter=','))
            nw.close()
            
            for i in range(len(bf)):
                idx,idy=self.nl.index(bf[i][1]),self.nl.index(bf[i][2])
                self.beta[idx,idy]=float(bf[i][4])+1
                self.beta[idy,idx]=float(bf[i][4])+1
                if numedge>0:
                    self.A[idx,idy] =float(bf[i][3])-1+self.A[idx,idy] 
                    self.A[idy,idx] =float(bf[i][3])-1+self.A[idy,idx]    
                else:
                    self.A[idx,idy] =float(bf[i][3])-1+float(bf[i][3])/float(bf[i][4])
                    self.A[idy,idx] =float(bf[i][3])-1+float(bf[i][3])/float(bf[i][4])

                        

    def randomize_s_t(self,rng):   
        for j in range(self.N):
            for k in range(self.K):
                self.s[j,k]=rng.random_sample(1)   #LNK
        self.t=self.s   #LNK  

                      
    def update_s(self):
        
        q_ijz=np.einsum('iz,jz->ijz',self.s,self.t)
        q_ij=np.einsum('ijz->ij',q_ijz)

        num=np.einsum('ij,ijz->iz',self.A,q_ijz)
        den=np.einsum('ij,jz->ijz',self.beta,self.t)
        den=np.einsum('ijz,ij->iz',den,q_ij)
        non_zeros=den>0.
        num[non_zeros]/=den[non_zeros]
        low_values_indices = num < self.err_max  # Where values are low
        num[low_values_indices] = 0.  # All low values set to 0
        dist_s=np.amax(abs(num-self.s))	
        self.s=num 
        return dist_s
    
    def update_t(self):
        self.t=self.s

    def likelihood(self):
        q_ijz =np.einsum('iz,jz->ij',self.s,self.t) 
        bq=np.einsum('ij,ij->ij',self.beta,q_ijz)
        non_zeros =q_ijz>0
        logq =np.log(q_ijz[non_zeros])
        alog = self.A[non_zeros]*logq
        lf =alog.sum() - bq.sum()
        #print(lf)
        return lf

    def check_convergence(self,it,l2,coincide,convergence):
        if (it%10==0):
            old_L=l2
            l2=self.likelihood()
            if (abs(l2-old_L)<self.tolerance):
                coincide+=1
            else:
                coincide=0
        if (coincide>self.decision):
            convergence=True
        it+=1
        return it,l2,coincide,convergence

    def output_save(self,maxl):
        s_f=[]
        s_f.append(list(self.w))

        for j in range(self.s.shape[0]):
            s_f.append([self.nl[j]]+list(self.s[j,:]))

        s_f.insert(0,['maxmum likelihood=',self.likelihood()])
        gi = open(self.savefile,'w',newline='')
        cw = csv.writer(gi,delimiter=',')
        cw.writerows(s_f)
        gi.close()   

    def update_EM(self):     
        dist_s =self.update_s()
        self.update_t()
        return dist_s

    def alpha_save(self):
        alpha =np.einsum('jk,k->jk',self.s,self.w)
        alpha =np.einsum('ik,jk->ij',self.s,alpha)  
        al_save =[]
        for j in range(alpha.shape[0]):
            al_save.append(list(alpha[j,:]))
        gi = open(self.savefile,'w',newline='')
        cw = csv.writer(gi,delimiter=',')
        cw.writerows(al_save)
        gi.close()           
                
    def realization(self):
        maxl=-100000000
        for r in range(self.N_real):
            rng =np.random.RandomState(self.rseed)
            self.rseed+=1
            self.randomize_s_t(rng)
            coincide =0
            convergence =False
            it =0
            l2 =self.inf
            delta_s = delta_t =self.inf
            # tic =time.clock()
            while (convergence==False and it<self.maxit):
                if self.directed:
                    delta_s,delta_t=self.update_EM()
                else:
                    delta_s=self.update_EM()
                    #print(delta_s)
                it,l2,coincide,convergence =self.check_convergence(it,l2,coincide,convergence)
                #print(self.likelihood())
            #print(self.likelihood(),it)

        print('converged=',convergence,"Final Likelihood=",self.likelihood())
        #self.output_save(self.likelihood())
        #self.alpha_save()

	












