import torch
import torch_scatter
import numpy as np
import scanpy as sc
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clt

class SHARE_topic:

    def __init__(self, sc_data_atac, sc_data_rna, n_topics, alpha, beta, gamma, tau):

        self.sc_data_atac = sc_data_atac
        self.sc_data_rna = sc_data_rna
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def sc_to_tensor(self):
    
        n = self.sc_data_atac.X.nonzero()[0].shape[0]
        atac = torch.zeros([2,n])
        atac[0,:] = torch.from_numpy(self.sc_data_atac.X.nonzero()[0])
        atac[1,:] = torch.from_numpy(self.sc_data_atac.X.nonzero()[1])
        atac = atac.type(torch.LongTensor)
       
        rna = self.sc_data_rna.X.toarray()
        rna = torch.tensor(rna)
        rna = rna.type(torch.DoubleTensor)

        self.atac = atac
        self.rna = rna

        #return rna, atac
    
    def create_cell_batches (self,n_batches,batch_size,n_cells):
    ## this function is used to create batch of cells for the atac data 
    # that should be transfered one at a time to the GPU to avoid GPU running 
     # out of memory. Because of the atac data format used to feed share-topic this function
       # is necessary to find the coordinates of the non-zero reads per batch.
        c = torch.arange(0,n_batches,batch_size)
        c = torch.hstack((c,torch.tensor(n_cells)))
        
            
        a,b,rep_c = torch.unique(self.atac[0,:],return_inverse =True,return_counts=True)
        rep_c=rep_c.type(torch.LongTensor) 
        cell_array = torch.arange(n_cells)
        rep_c_ = torch.repeat_interleave(cell_array, rep_c, dim = 0)
        

        q = 1
        t = 0
        t_ = 0
            
        atac_cell_batches = torch.zeros(c.shape[0],dtype = torch.int64)   
        for i in  torch.arange(batch_size,n_batches,batch_size):
            atac_cell_batches[q] = t+torch.sum(rep_c[t_:i])
            t = int(atac_cell_batches[q].item())
            t_ = i
            q+=1
            
        atac_cell_batches[q] = t+torch.sum(rep_c[t_:])
        
        return atac_cell_batches, c, rep_c,rep_c_, cell_array
    
    def create_region_batches(self, atac_cell_batches, cell_array):
        ###This function take the regions presented per batch of cells in the atac data with share-topic format. The presented regions
        # are ord
        indices = torch.zeros([self.atac.shape[1]])
        indices2 = torch.zeros([self.atac.shape[1]])
        regions = torch.tensor([])
        region_rep = torch.tensor([])
        region_rep_ = torch.tensor([])
        region_batching = torch.zeros([atac_cell_batches.shape[0]])
        for i in torch.arange(1,atac_cell_batches.shape[0]):
            
            sorted, indices[int(atac_cell_batches[i-1].item()):
                            int(atac_cell_batches[i].item())] = torch.sort(
                self.atac[1,int(atac_cell_batches[i-1].item()):
                            int(atac_cell_batches[i].item())])
            sorted2,indices2[int(atac_cell_batches[i-1].item()):
                                int(atac_cell_batches[i].item())] = torch.sort(
                indices[int(atac_cell_batches[i-1].item()):
                        int(atac_cell_batches[i].item())])
                
            a,b,rep_r = torch.unique(
                sorted,return_inverse =True,return_counts=True)
            regions = torch.cat((regions,a),0)
            region_batching[i] = a.shape[0]+region_batching[i-1] ##list of regiosn per batch
            region_rep = torch.cat((region_rep, rep_r),0) ##how many times presented region in a batch are repeated
            rep_r = rep_r.type(torch.LongTensor) 
            cell_array = torch.arange(a.shape[0]) 
            rep_r_ = torch.repeat_interleave(cell_array, rep_r,dim=0)
            region_rep_ = torch.cat((region_rep_, rep_r_),0) 

        region_rep = region_rep.type(torch.LongTensor) 
        regions = regions.type(torch.LongTensor) 
        indices = indices.int()
        indices2 = indices2.int()
        region_rep_ = region_rep_.type(torch.int64)
        return regions, region_batching, region_rep, region_rep_, indices, indices2
    
    def initialization(self, n_cells,G,device):
        m = torch.distributions.dirichlet.Dirichlet(self.alpha[0,:]) 
        theta_tmp = m.sample([n_cells])
        theta_tmp = theta_tmp.to(device)
        
        m = torch.distributions.gamma.Gamma(self.gamma,self.tau)
        lam_tmp = m.sample([self.n_topics,G])
        lam_tmp = lam_tmp.to(device)
        
        m =  torch.distributions.dirichlet.Dirichlet(self.beta[0,:])
        phi_tmp = m.sample([self.n_topics])
        phi_tmp = phi_tmp.to(device)
        
        return theta_tmp, lam_tmp, phi_tmp
    
    def move_to_GPU(self, c,region_rep, regions, indices, indices2, rep_c, rep_c_, region_rep_, R, device):
    
        c=c.to(device)
        region_rep = region_rep.to(device)
        regions = regions.to(device)
        indices = indices.to(device)
        indices2 = indices2.to(device)
        rep_c = rep_c.to(device)
        rep_c_ = rep_c_.to(device)
        region_rep_ = region_rep_.to(device)
        t = torch.arange(0,self.n_topics,dtype=torch.uint8,device=device).reshape(self.n_topics,1)
        self.alpha = torch.ones([1,self.n_topics,],device=device)*(50/self.n_topics)
        self.beta = torch.ones([1,R],device=device)*0.1
        
        return  c, region_rep, regions, indices, indices2, rep_c, rep_c_, region_rep_, t
    
    def Gibbs_Sampler(self, n_samples,n_burnin,
                       n_b_samples, batch_size, n_cells,R, G, c, region_rep,
                         regions,region_batching, indices, indices2, rep_c, rep_c_, region_rep_, 
                         t,atac_cell_batches , theta_tmp, phi_tmp, lam_tmp,device):
    
        
         
        theta = torch.zeros([n_b_samples,n_cells, self.n_topics])
        lam = torch.zeros([n_b_samples,self.n_topics, G])
        phi = torch.zeros([n_b_samples,self.n_topics, R])
        L_g = torch.zeros(n_b_samples, device=device)
        L_r = torch.zeros(n_b_samples, device=device)
        with tqdm(total=n_samples, desc="Processing") as pbar:
            for sample in range(0,n_samples):
                n_t_g_shape = torch.zeros([G,self.n_topics],device=device)
                n_t_g_scale = torch.zeros([G,self.n_topics],device=device)
                n_t_c = torch.zeros([n_cells,self.n_topics],device=device)
                n_t_r = torch.zeros([R,self.n_topics],device=device)
                L_g[int(sample/n_burnin)] = 0
                L_r[int(sample/n_burnin)] = 0
                
                for i in torch.arange(1,c.shape[0]):
                ########################################RNA-BATCH###################################
                
                    rna_batch = self.rna[c[i-1]:c[i],:].to(device)
                    m = torch.distributions.poisson.Poisson(lam_tmp.reshape([self.n_topics,1,G]))
                    z_ = torch.mul(torch.exp(m.log_prob(rna_batch)),(theta_tmp[c[i-1]:c[i],:].T)[:,:,None])
                    z_L_g = torch.sum(z_,axis=0)
                    z_L_g = torch.sum(torch.log(z_L_g))
                    L_g[int(sample/n_burnin)]+= z_L_g
                    z_ = torch.div(z_,torch.sum(z_,axis=0))
                    z_[z_ != z_] = 0
                    k = torch.nonzero(torch.sum(z_,axis=0)==0)
                        
                    del m
                    
                    for j in k:
                        
                        m = torch.distributions.poisson.Poisson(lam_tmp[:,j[1]])
                        s = m.log_prob(rna_batch[j[0],j[1]])+theta_tmp[j[0],:].view(1, -1)
                        del m


                        z_[:,j[0],j[1]] = torch.exp(s-max(s))
                        
                        z_[:,j[0],j[1]] = z_[:,j[0],j[1]]/torch.sum(z_[:,j[0],j[1]])
                        
                    
                    z_ = torch.cumsum(z_,axis=0)
                    u = torch.rand([1,(c[i]-c[i-1]).item(),G],dtype=torch.half,device=device)
                    
                        
                    z_c = (torch.swapaxes(z_, 0, 2)).contiguous()
                    u_c = (torch.swapaxes(u, 0, 2)).contiguous()
                    z_ = torch.searchsorted(z_c,u_c)
                    
                    t_test = torch.cat(int((c[i]-c[i-1]).item())*[t],1).T
                    z_ = torch.gt((z_==t_test[None,:,:]), 0).int()
                    n_t_c[c[i-1]:c[i],:] = torch.sum(z_,axis=0)
                    n_t_g_shape+= torch.sum(z_*rna_batch.T[:,:,None],axis=1)
                    n_t_g_scale+= torch.sum(z_,axis=1)
                    del rna_batch
                #########################################ATAC_BATCH##########################################

                    theta_ = theta_tmp[c[i-1]:c[i],:]
                    phi_ = phi_tmp[:,regions[int(region_batching[i-1].item()):int(region_batching[i].item())]]
                    n_regions = phi_.shape[1]
                    z_atac_ = torch.repeat_interleave(
                        theta_, rep_c[c[i-1]:c[i]],dim=0)##cells in this batch
                    phi_ = torch.repeat_interleave(
                        phi_,region_rep[int(region_batching[i-1].item()):int(region_batching[i].item())],dim=1)##regions in this batch
                    phi_ = torch.index_select(phi_,1, indices2[int(atac_cell_batches[i-1].item()):
                                                                int(atac_cell_batches[i].item())])##reorder the regions acording to data
                    
                    z_atac_ = torch.multiply(z_atac_.T,phi_)
                    z_L_r = torch.sum(z_atac_,axis=0)
                    z_L_r = torch.sum(torch.log(z_L_r))
                    L_r[int(sample/n_burnin)]+= z_L_r
                    z_atac_ = z_atac_/torch.sum(z_atac_,axis=0)
                    
                    z_atac_ = torch.cumsum(z_atac_,axis=0)
                    u_atac = torch.rand([1,z_atac_.shape[1]],device=device)
                    
                    z_atac_ = torch.searchsorted( (z_atac_.T).contiguous(),(u_atac.T).contiguous()).contiguous()
                    
                    z_atac_ = torch.gt((z_atac_==t.T), 0).int()
                    
                    h = rep_c_[int(atac_cell_batches[i-1].item()):
                                int(atac_cell_batches[i].item())]%batch_size###update for the cells in this batch
                    
                    n_t_c[c[i-1]:c[i],:]+= (
                        torch_scatter.scatter(z_atac_,h,dim=0,reduce="sum").reshape([int((c[i]-c[i-1]).item()),self.n_topics]))
                    
                    z_atac_ = torch.index_select(z_atac_,0,indices[int(atac_cell_batches[i-1].item()):
                                                                    int(atac_cell_batches[i].item())])
                    
                    h = region_rep_[int(atac_cell_batches[i-1].item()):
                                    int(atac_cell_batches[i].item())]
                    
                    n_t_r[regions[int(region_batching[i-1].item()):int(region_batching[i].item())],:]+= (
                        torch_scatter.scatter(z_atac_,h,dim=0,reduce="sum").reshape([n_regions,self.n_topics]))
                    
                #######################################Sampling theta, lambda, and phi##########################

                m = torch.distributions.gamma.Gamma(self.gamma+n_t_g_shape,(n_t_g_scale*self.tau+1)/self.tau)
                lam_tmp=m.sample()
                lam_tmp=lam_tmp.T
                lam[int(sample/n_burnin),:,:]=lam_tmp 
                
                m = torch.distributions.dirichlet.Dirichlet(self.beta+n_t_r.T)
                phi_tmp = m.sample()
                phi[int(sample/n_burnin),:,:] = phi_tmp 
                m = torch.distributions.dirichlet.Dirichlet(self.alpha+n_t_c)
                
                theta_tmp = m.sample()
                theta[int(sample/n_burnin),:,:] = theta_tmp
                time.sleep(0.1)
                pbar.update(1)
        return theta, lam, phi 
    
    def fit(self,batch_size,n_samples,n_burnin,dev= "cpu",
                save_data=True, save_samples = False, path=""):
    
        self.sc_to_tensor()
       
        
        if save_data:
            torch.save(self.atac, path+str("atac_share_topic.txt"))
            torch.save(self.rna, path+str("rna_share_topic.txt"))
        ### add path to save the files
        
        device = torch.device(dev)
        
        G = self.rna.shape[1]
        n_cells = self.rna.shape[0]
        l = torch.unique(self.atac[1,:])
        R = l.shape[0]
        self.alpha = torch.ones([1, self.n_topics])*self.alpha
        self.beta = torch.ones([1,R])*self.beta
        n_b_samples = int(n_samples/n_burnin)
        n_batches = n_cells+batch_size-(n_cells%batch_size)
        
        atac_cell_batches, c, rep_c,rep_c_, cell_array = self.create_cell_batches(n_batches,batch_size,n_cells)
        
        regions, region_batching, region_rep, region_rep_, indices, indices2 = self.create_region_batches(atac_cell_batches, cell_array)
        
        theta_tmp, lam_tmp, phi_tmp = self.initialization(n_cells,G,device)
        
        if dev!="cpu":
            c, region_rep, regions, indices, indices2, rep_c, rep_c_, region_rep_, t = self.move_to_GPU(c,region_rep, regions, indices, indices2, rep_c, rep_c_, region_rep_, R, device)
        
        
        else:
            t = torch.arange(0,self.n_topics,dtype=torch.uint8,device=device).reshape(self.n_topics,1)
            self.alpha = torch.ones([1,self.n_topics,],device=device)*(50/self.n_topics)
            self.beta = torch.ones([1,R],device=device)*0.1

        theta, lam, phi = self.Gibbs_Sampler(n_samples,n_burnin,
                                        n_b_samples, batch_size, n_cells,R, G, c, region_rep,
                                            regions,region_batching, indices, indices2, rep_c, rep_c_, region_rep_, 
                                            t,atac_cell_batches , theta_tmp, phi_tmp, lam_tmp,device)
        
        torch.cuda.empty_cache()

        if save_samples:
            torch.save(theta, str(path)+"theta_"+str(t)+".txt")
            torch.save(lam, str(path)+"lam_"+str(t)+".txt")
            torch.save(phi, str(path)+"phi_"+str(t)+".txt")

        return theta, lam, phi
    
    def compute_WAIC_ATAC(self, c,P, device, n_samples, theta, phi, regions, region_batching, rep_c,
                      region_rep, atac_cell_batches, indices2):
        c=c.to(device)
        region_rep = region_rep.to(device)
        regions = regions.to(device)
        indices2 = indices2.to(device)
        rep_c = rep_c.to(device)
        
        
        L=torch.zeros(P,dtype=torch.double,device=device)
        V=torch.zeros(P,device=device)
        V_2=torch.zeros(P,device=device)
        
        for sample in range(0,n_samples):
            theta_tmp=theta[sample,:,:].to(device)
            phi_tmp=phi[sample,:,:].to(device)
            for i in torch.arange(1,c.shape[0]):
            
                theta_=theta_tmp[c[i-1]:c[i],:]
                phi_=phi_tmp[:,regions[int(region_batching[i-1].item()):int(region_batching[i].item())]]
                n_regions=phi_.shape[1]
                z_atac_=torch.repeat_interleave(
                theta_, rep_c[c[i-1]:c[i]],dim=0)##cells in this batch
                phi_=torch.repeat_interleave(
                phi_,region_rep[int(region_batching[i-1].item()):int(region_batching[i].item())],dim=1)##regions in this batch
                phi_=torch.index_select(phi_,1, indices2[int(atac_cell_batches[i-1].item())
                                                    :int(atac_cell_batches[i].item())])##reorder the regions acording to data
                z_atac_=torch.multiply(z_atac_.T,phi_)
            
                z_atac_=torch.sum(z_atac_,axis=0)
                L[int(atac_cell_batches[i-1].item())
                :int(atac_cell_batches[i].item())]+=z_atac_
                V[int(atac_cell_batches[i-1].item())
                :int(atac_cell_batches[i].item())]+=torch.pow(torch.log(z_atac_),2)
                V_2[int(atac_cell_batches[i-1].item())
                :int(atac_cell_batches[i].item())]+=torch.log(z_atac_)
        WAIC_ATAC=-2*(torch.sum(torch.log(L/n_samples))-(torch.sum(V)-torch.sum(torch.pow(V_2,2)
                                                                                /n_samples))/(n_samples-1))
        return WAIC_ATAC
    
    def WAIC_ATAC(self, theta, phi, device, batch_size=100, burn_in=50):
    
        
        theta = theta[burn_in:,:,:]
        phi = phi[burn_in:,:,:]

          
        n_samples=phi.shape[0]
        n = theta.shape[2]
        P= self.atac.shape[1]
        n_cells = theta.shape[1]
        n_batches = n_cells+batch_size-(n_cells%batch_size)    
        atac_cell_batches,c, rep_c,rep_c_,ren = self.create_cell_batches(n_batches, batch_size, n_cells)
            
        regions, region_batching, region_rep, region_rep_, indices, indices2 = self.create_region_batches(atac_cell_batches,ren)
            
            

        WAIC_ATAC = self.compute_WAIC_ATAC(c, P, device, n_samples, theta, phi, regions, region_batching, rep_c,
                            region_rep, atac_cell_batches,indices2)

        return WAIC_ATAC
            
    
    
    def WAIC_RNA (self, batch_size, theta, lam, device):

        n_cells = theta.shape[1]
        n_batches = n_cells+batch_size-(n_cells%batch_size)
        c = torch.arange(0,n_batches,batch_size)
        c = torch.hstack((c,torch.tensor(n_cells)))
        c = c.to(device)

        
        N=lam.shape[0]
        T=lam.shape[1]
        G=lam.shape[2]
        C=theta.shape[1]
        
        L=torch.zeros([C,G],dtype=torch.double,device=device)
        V=torch.zeros([C,G],device=device)
        V_2=torch.zeros([C,G],device=device)
        
        for n in torch.arange(0,N):
            theta_tmp=theta[n,:,:].to(device)
            lam_tmp=lam[n,:,:].to(device)
            if device!="cpu":

                theta_tmp=theta_tmp.type(torch.cuda.DoubleTensor)
                lam_tmp=lam_tmp.type(torch.cuda.DoubleTensor)
                
            for i in torch.arange(1,c.shape[0]): 
                data_RNA_1=self.rna[c[i-1]:c[i],:].to(device)
                m=torch.distributions.poisson.Poisson(lam_tmp.reshape([T,1,G]))
                z_=torch.mul(torch.exp(m.log_prob(data_RNA_1)),(theta_tmp[c[i-1]:c[i],:].T)[:,:,None])

                
                z_=torch.sum(z_,axis=0)
                L[c[i-1]:c[i],:]+=z_
                
                V[c[i-1]:c[i],:]+=torch.pow(torch.log(z_),2)
                
                V_2[c[i-1]:c[i],:]+=torch.log(z_)
        
        WAIC_rna=-2*(torch.sum(torch.log(L/N))-(torch.sum(V)-torch.sum(torch.pow(V_2,2)/N))/(N-1))
        return WAIC_rna

    def WAIC (self, batch_size, theta, lam, phi, device):

        WAIC_atac= self.WAIC_ATAC (theta, phi, device)
        torch.cuda.empty_cache()
        WAIC_rna = self.WAIC_RNA (batch_size, theta, lam, device)
        torch.cuda.empty_cache()

        return WAIC_atac+WAIC_rna
    
    def read_samples (self,t, path, burnin_samples):
    
        theta = torch.load(str(path)+"theta_"+str(t)+".txt",map_location=torch.device('cpu'))
        m_theta = theta[burnin_samples:,:,:].mean(axis=0)
        m_theta = m_theta/m_theta.sum(axis=1)[:,np.newaxis] 
        
        lam = torch.load(str(path)+"lam_"+str(t)+".txt",map_location=torch.device('cpu'))
        m_lam = lam[burnin_samples:,:,:].mean(axis=0)
        
        phi = torch.load(str(path)+"phi_"+str(t)+".txt",map_location=torch.device('cpu'))
        m_phi = phi[burnin_samples:,:,:].mean(axis=0)

        
        
        return m_theta, m_lam, m_phi

    def read_samples_STS (self, t, path, burnin_samples):
    
        m_theta, m_lam, m_phi = self.read_samples (t, path, burnin_samples)
        
        m_lam_ = m_lam/m_lam.sum(axis=0)
        m_phi_ = m_phi/m_phi.sum(axis=0)
        
        return m_theta, m_lam_, m_phi_
    
    def share_topic_output(self, sc_data, t, path, burnin_samples=50):
    
        m_theta, m_lam, m_phi = self.read_samples (t, path, burnin_samples)


        sc_data.obsm["SHARE_Topic"]=m_theta.numpy()
        sc.pp.neighbors(sc_data, use_rep="SHARE_Topic")
        sc.tl.umap(sc_data)
            
        return sc_data, m_theta, m_lam, m_phi
    
    def cell_types_visualization (self, sc_data, key=None):
    
        sc.pl.umap(sc_data,color=key, title=" ", return_fig=True, legend_fontweight='bold')

    def topic_cell_visualization(self,sc_data, topics, theta, nrows, ncols, figsize):

    
    
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=figsize)
        cmap = clt.LinearSegmentedColormap.from_list("wr", ("white", "red"))
        vmin, vmax = theta.min(), theta.max()
        i=0
        for c_, ax in enumerate(axes.flat):
            ax.set_title("topic"+str(topics[i]))
            im=ax.scatter(
            x=sc_data.obsm["X_umap"][:,0],
            y=sc_data.obsm["X_umap"][:,1],
            c=theta[:,topics[i]-1],
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            s=0.1)
            i+=1 
            if i==len(topics):
                    
                    break
        fig.colorbar(im, ax=axes.ravel().tolist())

    def plot_STS_gene(self,gene_name,STS_score, distance, df_gene_regions_all_topic,save_fig):
       
        plt.scatter(x=distance-df_gene_regions_all_topic["start_gene"].iloc[1],y=STS_score)
        plt.axvline(x=0, ymin=0, ymax=10,color='green', linestyle='dotted')
        plt.axvline(x=df_gene_regions_all_topic["end_gene"].iloc[1]-df_gene_regions_all_topic["start_gene"].iloc[1]
             , ymin=0, ymax=10,color='red', linestyle='dotted')
        plt.ylabel("$P_g^r$")
        plt.xlabel("d")
        plt.title(gene_name)
        if save_fig:
            plt.savefig('STS_scores_'+str(gene_name)+'.png',bbox_inches='tight',dpi=1000)
        plt.show()

    def compute_gene_STS (self,gene_name,t,path, bed_file, burnin_samples = 50,
                      norm_to_max = True, plot = True, save_fig = False): ########Specify the burnin samples(default is 50)
    
    
        df_genes_regions = pd.read_csv(bed_file,delimiter="\t")
        m_theta, m_lam, m_phi = self.read_samples_STS (t, path, burnin_samples)
    

        gene_num=df_genes_regions["gene_number"][df_genes_regions["gene_name"]==gene_name].iloc[0]
        p_phi = 0
        df_gene_regions_all_topic = df_genes_regions[df_genes_regions["gene_name"]==gene_name]
        i = 0
        num_regions = df_genes_regions[df_genes_regions["gene_name"]==gene_name]["region_number"].shape[0]
        corr = np.zeros(num_regions)
        distance = np.zeros(num_regions)
        distance_end = np.zeros(num_regions)
        
        for n in np.arange(0,num_regions):
            region_num = df_gene_regions_all_topic["region_number"].iloc[n]
    
            p_phi = torch.multiply(m_phi[:,region_num],m_theta)
        
            corr[i] = torch.multiply(p_phi,m_lam[:,gene_num]).sum(axis=0).sum(axis=0).numpy()
            distance[i] = df_gene_regions_all_topic["start_region"].iloc[n]
            i+=1
            
        STS_score = corr/m_theta.shape[0]
        
        if plot:
            
            self.plot_STS_gene(gene_name, STS_score, distance, df_gene_regions_all_topic,save_fig)
        
        
        if norm_to_max:
            
            return STS_score/STS_score.max(),distance,df_gene_regions_all_topic
        else:

            return STS_score,distance,df_gene_regions_all_topic
        
    def plot_STS (self, STS_list, distance_list, gene_pos,save_fig):
        max_STS = max(max(inner_list) for inner_list in STS_list)
        plt.figure(figsize = (5,3))
        for i,j in enumerate(gene_pos[:len(distance_list)]):
            plt.scatter(x=distance_list[i]*(-1), y=STS_list[i]/max_STS, alpha=0.3, c='#1f77b4')
        plt.ylabel("$P_g^r$")
        plt.xlabel("d")
        if save_fig:
            plt.savefig('STS_scores.png',bbox_inches='tight',dpi=1000)
        plt.show()

    def compute_STS(self,t, path, bed_file,  burnin_samples = 50, plot = True, save_fig = False):
    
        df_genes_regions = pd.read_csv(bed_file,delimiter="\t")
        
        m_theta, m_lam, m_phi = self.read_samples_STS (t, path, burnin_samples)

        gene_list = list(df_genes_regions["gene_name"].drop_duplicates())
        distance_list = []
        STS_list = []
        gene_pos = np.zeros(len(gene_list))
        
        for l,gene in enumerate(gene_list):
            if df_genes_regions[df_genes_regions["gene_name"]==gene].shape[0] != 0:
                gene_num = df_genes_regions["gene_number"][df_genes_regions["gene_name"]==gene].iloc[0]
                x = df_genes_regions["gene_name"]==gene
                num_regions = df_genes_regions[x]["region_number"].shape[0]
                regions_num = df_genes_regions[x]["region_number"].values
                regions_begin = df_genes_regions[x]["start_region"].values
                distance_begin = df_genes_regions[x]["start_gene"]
                p_phi = m_phi[:,regions_num]
                p_phi = torch.multiply(m_theta[:,:,None],p_phi[None,:,:])
                corr = torch.multiply(p_phi,m_lam[:,gene_num][None,:,None]).sum(axis=0).sum(axis=0)
                STS_list.append(corr)
                gene_pos[l] = distance_begin.iloc[0]
                distance_list.append(distance_begin-regions_begin)
        if plot:
            print("ploting..")
            self.plot_STS(STS_list, distance_list, gene_pos,save_fig)
        
            
        return gene_list, STS_list, distance_list