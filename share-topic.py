import torch
import torch_scatter

def share-topic(path,atac_txt_file,rna_txt_file,gamma_=1,tau_=1,n_topics_,alpha_=None,beta_=0.1,n_samples_=3000,n_burnin_=10,batch_size_=500,
                return_samples=False,return_Likelihood=True,store_path=None):

 read_RNAsc=torch.load(path+rna_txt_file)
 data_atac=torch.load(path+atac_txt_file)  
 read_RNAsc=read_RNAsc.type(torch.DoubleTensor)

 dev = "cuda:0"
 device = torch.device(dev)

 gamma=gamma_
 tau=tau_
 n_topics=n_topics_
 G=read_RNAsc.shape[1]
 n_cells=read_RNAsc.shape[0]
 l=torch.unique(data_atac[1,:])
 R=l.shape[0]
 
 if alpha_ is None:
  alpha=torch.ones([1,n_topics])*(50/n_topics)
 else:
  alpha=torch.ones([1,n_topics])*(alpha_)  
 beta=torch.ones([1,R])*beta_

 n_samples=n_samples_
 n_burnin=n_burnin_
 n_b_samples=int(n_samples/n_burnin)
 batch_size=batch_size_
 n_batches=n_cells+batch_size-(n_cells%batch_size)


 warnings.filterwarnings("ignore")
 
 lam=torch.zeros([n_b_samples,n_topics,G])
 theta=torch.zeros([n_b_samples,n_cells,n_topics])
 phi=torch.zeros([n_b_samples,n_topics,R])
 L_g=torch.zeros(n_b_samples,device=device)
 L_r=torch.zeros(n_b_samples,device=device)
 
 c=torch.arange(0,n_batches,batch_size)
 c=torch.hstack((c,torch.tensor(n_cells)))
 c=c.to(device)



 a,b,rep_c=torch.unique(data_atac[0,:],return_inverse =True,return_counts=True)
 rep_c=rep_c.type(torch.LongTensor) 
 ren=torch.arange(n_cells)
 rep_c_=torch.repeat_interleave(ren, rep_c,dim=0)


 k_atac=torch.zeros(c.shape[0],dtype=torch.int64)

 q=1
 t=0
 t_=0
 for i in  torch.arange(batch_size,n_batches,batch_size):
    
    k_atac[q]=t+torch.sum(rep_c[t_:i])
    t=int(k_atac[q].item())
    t_=i
    q+=1
 k_atac[q]=t+torch.sum(rep_c[t_:])
 
 m = torch.distributions.dirichlet.Dirichlet(alpha[0,:]) 
 theta_tmp=m.sample([n_cells])
 theta_tmp=theta_tmp.to(device)
 m = torch.distributions.gamma.Gamma(gamma,tau)
 lam_tmp=m.sample([n_topics,G])
 lam_tmp=lam_tmp.to(device)
 m =  torch.distributions.dirichlet.Dirichlet(beta[0,:])
 phi_tmp=m.sample([n_topics])
 phi_tmp=phi_tmp.to(device)
 
 indices=torch.zeros([data_atac.shape[1]])
 indices2=torch.zeros([data_atac.shape[1]])
 regions=torch.tensor([])
 region_rep=torch.tensor([])
 region_rep_=torch.tensor([])
 region_batching=torch.zeros([k_atac.shape[0]])
 for i in torch.arange(1,k_atac.shape[0]):
    
   sorted, indices[int(k_atac[i-1].item()):int(k_atac[i].item())]= torch.sort(
       data_atac[1,int(k_atac[i-1].item()):int(k_atac[i].item())])
   sorted2,indices2[int(k_atac[i-1].item()):int(k_atac[i].item())]=torch.sort(
       indices[int(k_atac[i-1].item()):int(k_atac[i].item())])
   a,b,rep_r=torch.unique(
       sorted,return_inverse =True,return_counts=True)
   regions=torch.cat((regions,a),0)
   region_batching[i]=a.shape[0]+region_batching[i-1] ##list of regiosn per batch
   region_rep=torch.cat((region_rep,rep_r),0) ##how many times presented region in a batch are repeated
   rep_r=rep_r.type(torch.LongTensor) 
   ren=torch.arange(a.shape[0]) 
   rep_r_=torch.repeat_interleave(ren, rep_r,dim=0)
   region_rep_=torch.cat((region_rep_,rep_r_),0) 

 region_rep=region_rep.type(torch.LongTensor) 
 regions=regions.type(torch.LongTensor) 
 indices=indices.int()
 indices2=indices2.int()
 region_rep_=region_rep_.type(torch.int64)
 
 region_rep=region_rep.to(device)
 regions=regions.to(device)
 indices=indices.to(device)
 indices2=indices2.to(device)
 rep_c=rep_c.to(device)
 rep_c_=rep_c_.to(device)
 region_rep_=region_rep_.to(device)
 t=torch.arange(0,n_topics,dtype=torch.uint8,device=device).reshape(n_topics,1)
 
 
 for sample in range(0,n_samples):
 print(sample)

 n_t_g_shape=torch.zeros([G,n_topics],device=device)
 n_t_g_scale=torch.zeros([G,n_topics],device=device)
 n_t_c=torch.zeros([n_cells,n_topics],device=device)
 n_t_r=torch.zeros([R,n_topics],device=device)
 L_g[int(sample/n_burnin)]=0
 L_r[int(sample/n_burnin)]=0

 for i in torch.arange(1,c.shape[0]):
    
    read_RNAsc1=read_RNAsc[c[i-1]:c[i],:].to(device)
    m=torch.distributions.poisson.Poisson(lam_tmp.reshape([n_topics,1,G]))
    z_=torch.mul(torch.exp(m.log_prob(read_RNAsc1)),(theta_tmp[c[i-1]:c[i],:].T)[:,:,None])
    z_L_g=torch.sum(z_,axis=0)
    z_L_g=torch.sum(torch.log(z_L_g))
    L_g[int(sample/n_burnin)]+=z_L_g
    z_=torch.div(z_,torch.sum(z_,axis=0))
    z_[z_ != z_] = 0
    k=torch.nonzero(torch.sum(z_,axis=0)==0)
        
    del m
   
    for j in k:
         m=torch.distributions.poisson.Poisson(lam_tmp[:,j[1]])
         s=m.log_prob(read_RNAsc1[j[0],j[1]])+theta_tmp[j[0],:].T
         del m


         z_[:,j[0],j[1]]=torch.exp(s-max(s))
        
         z_[:,j[0],j[1]]=z_[:,j[0],j[1]]/torch.sum(z_[:,j[0],j[1]])
         
     
    z_=torch.cumsum(z_,axis=0)
    u=torch.rand([1,(c[i]-c[i-1]).item(),G],dtype=torch.half,device=device)
    z_=torch.searchsorted(torch.swapaxes(z_, 0, 2),torch.swapaxes(u, 0, 2))
    t_test=torch.cat(int((c[i]-c[i-1]).item())*[t],1).T
    z_=torch.gt((z_==t_test[None,:,:]), 0).int()
    n_t_c[c[i-1]:c[i],:]=torch.sum(z_,axis=0)
    n_t_g_shape+=torch.sum(z_*read_RNAsc1.T[:,:,None],axis=1)
    n_t_g_scale+=torch.sum(z_,axis=1)
    del read_RNAsc1 
    
    theta_=theta_tmp[c[i-1]:c[i],:]
    phi_=phi_tmp[:,regions[int(region_batching[i-1].item()):int(region_batching[i].item())]]
    n_regions=phi_.shape[1]
    z_atac_=torch.repeat_interleave(
        theta_, rep_c[c[i-1]:c[i]],dim=0)##cells in this batch
    phi_=torch.repeat_interleave(
        phi_,region_rep[int(region_batching[i-1].item()):int(region_batching[i].item())],dim=1)##regions in this batch
    phi_=torch.index_select(phi_,1, indices2[int(k_atac[i-1].item()):int(k_atac[i].item())])##reorder the regions acording to data
    z_atac_=torch.multiply(z_atac_.T,phi_)
    z_L_r=torch.sum(z_atac_,axis=0)
    z_L_r=torch.sum(torch.log(z_L_r))
    L_r[int(sample/n_burnin)]+=z_L_r
    z_atac_=z_atac_/torch.sum(z_atac_,axis=0)
     
    z_atac_=torch.cumsum(z_atac_,axis=0)
    u_atac=torch.rand([1,z_atac_.shape[1]],device=device)
    z_atac_=torch.searchsorted( z_atac_.T,u_atac.T)
    z_atac_=torch.gt((z_atac_==t.T), 0).int()
    
    h=rep_c_[int(k_atac[i-1].item()):int(k_atac[i].item())]%batch_size ###update for the cells in this batch
     
    n_t_c[c[i-1]:c[i],:]+=(
        torch_scatter.scatter(z_atac_,h,dim=0,reduce="sum").reshape([int((c[i]-c[i-1]).item()),n_topics]))
    
    z_atac_=torch.index_select(z_atac_,0,indices[int(k_atac[i-1].item()):int(k_atac[i].item())])
    
    h=region_rep_[int(k_atac[i-1].item()):int(k_atac[i].item())]
     
    n_t_r[regions[int(region_batching[i-1].item()):int(region_batching[i].item())],:]+=(
        torch_scatter.scatter(z_atac_,h,dim=0,reduce="sum").reshape([n_regions,n_topics]))
    
    
    
 m = torch.distributions.gamma.Gamma(gamma+n_t_g_shape,(n_t_g_scale*tau+1)/tau)
 lam_tmp=m.sample()
 lam_tmp=lam_tmp.T
 lam[int(sample/n_burnin),:,:]=lam_tmp 
 m = torch.distributions.dirichlet.Dirichlet(beta+n_t_r.T)
 phi_tmp=m.sample()
 phi[int(sample/n_burnin),:,:]=phi_tmp 
 m = torch.distributions.dirichlet.Dirichlet(alpha+n_t_c)
 theta_tmp=m.sample()
 theta[int(sample/n_burnin),:,:]=theta_tmp




 if return_samples==True:
  if store_path is None:
     store_path=path
 
  torch.save(theta,store_path+str(n_topics)+"_topics/theta_"+str(n_topics)+".txt") 
  torch.save(phi,store_path+str(n_topics)+"_topics/phi_"+str(n_topics)+".txt")
  torch.save(lam,store_path+str(n_topics)+"_topics/lam_"+str(n_topics)+".txt") 
  
 if return_Likelihood==True:
  if store_path is None:
     store_path=path 
  torch.save(L_r,store_path+str(n_topics)+"_topics/L_r_"+str(n_topics)+".txt") ### Likelihood of the scATAC-seq part
  torch.save(L_g,store_path+str(n_topics)+"_topics/L_g_"+str(n_topics)+".txt") ### Likelihood of the scRNA-seq part
 