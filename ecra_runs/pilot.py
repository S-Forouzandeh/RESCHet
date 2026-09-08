import math, random, json
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset

SEED=2026
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
K_VALUES=[2,4,6,8,10,12,14,16]
ETA_VALUES=[0.00,0.15,0.30,0.45,0.60,0.75,0.90,1.00]
D=24; MAX_USERS=1200; EPOCHS=1

print('loading real dataset...',flush=True)
ds=load_dataset('oyku-tugana/amazon-musical-instruments-2018-2023-5core','interactions',split='train')
by=defaultdict(list); item_map={}
for r in ds:
    u=r['user_id']; i=r['parent_asin']; t=int(r['timestamp'])
    if i not in item_map: item_map[i]=len(item_map)
    by[u].append((t,item_map[i]))
seqs=[]
for u,x in by.items():
    x=sorted(x)
    ids=[]; seen=set()
    for _,i in x:
        if i not in seen: ids.append(i); seen.add(i)
    if len(ids)>=5: seqs.append(ids[-50:])
random.shuffle(seqs); seqs=seqs[:MAX_USERS]
n=len(item_map)
print('users',len(seqs),'items',n,flush=True)

class M(nn.Module):
    def __init__(self,n,k):
        super().__init__(); self.k=k
        self.item=nn.Embedding(n,D)
        self.proto=nn.Parameter(torch.randn(k,D)/math.sqrt(D))
        self.proj=nn.Linear(D,D,bias=False)
        self.gru=nn.GRUCell(D,D)
        self.A=nn.Parameter(torch.zeros(k,k))
        nn.init.xavier_uniform_(self.item.weight)
    def step(self,h,state,alpha,eta):
        e=self.item(h); logits=e@self.proto.t(); p=F.softmax(logits,-1)
        z=self.proj(e).unsqueeze(1).expand(-1,self.k,-1)
        ns=[]
        for j in range(self.k): ns.append(self.gru(p[:,j:j+1]*z[:,j],state[:,j]))
        state=torch.stack(ns,1)
        sim=torch.einsum('bkd,bjd->bkj',state,state)/math.sqrt(D)
        T=F.softmax(self.A.unsqueeze(0)+sim,-1)
        prior=torch.bmm(alpha.unsqueeze(1),T).squeeze(1)
        pref=torch.einsum('bk,bkd->bd',prior,state)
        alpha=F.softmax(eta*torch.log(prior.clamp_min(1e-8))+(1-eta)*logits,-1)
        return state,alpha,pref

def train_k(k):
    m=M(n,k); opt=torch.optim.Adam(m.parameters(),lr=2e-3)
    for ep in range(EPOCHS):
        for s in seqs:
            tr=s[:-2]
            if len(tr)<2: continue
            state=torch.zeros(1,k,D); alpha=torch.full((1,k),1/k)
            seen=set(tr)
            for a,b in zip(tr[:-1],tr[1:]):
                state,alpha,pref=m.step(torch.tensor([a]),state,alpha,0.5)
                neg=random.randrange(n)
                while neg in seen: neg=random.randrange(n)
                pos=(pref*m.item(torch.tensor([b]))).sum(-1)
                negs=(pref*m.item(torch.tensor([neg]))).sum(-1)
                loss=-F.logsigmoid(pos-negs).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                state=state.detach(); alpha=alpha.detach()
    return m

def eval_eta(m,k,eta):
    vals=[]
    with torch.no_grad():
        for s in seqs:
            hist=s[:-1]; target=s[-1]
            state=torch.zeros(1,k,D); alpha=torch.full((1,k),1/k); pref=None
            for a in hist:
                state,alpha,pref=m.step(torch.tensor([a]),state,alpha,eta)
            scores=(pref@m.item.weight.t()).squeeze(0)
            scores[torch.tensor(list(set(hist)))]=-1e9
            ts=scores[target].item(); rank=int((scores>ts).sum().item())+1
            vals.append(0.0 if rank>10 else 1.0/math.log2(rank+1))
    return float(np.mean(vals))

Z=np.zeros((len(ETA_VALUES),len(K_VALUES)),dtype=float)
for xi,k in enumerate(K_VALUES):
    print('TRAIN K',k,flush=True)
    m=train_k(k)
    for yi,eta in enumerate(ETA_VALUES):
        z=eval_eta(m,k,eta); Z[yi,xi]=z
        print('RESULT',k,eta,z,flush=True)

np.savez('measured_K_eta_8x8.npz',K=np.array(K_VALUES),ETA=np.array(ETA_VALUES),NDCG10=Z)
with open('measured_K_eta_8x8.json','w') as f: json.dump({'K':K_VALUES,'eta':ETA_VALUES,'ndcg10':Z.tolist(),'users':len(seqs),'dataset':'oyku-tugana/amazon-musical-instruments-2018-2023-5core'},f,indent=2)
X,Y=np.meshgrid(np.array(K_VALUES),np.array(ETA_VALUES))
fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,cmap='viridis',edgecolor='k',linewidth=.35,antialiased=True)
ax.set_xlabel('Intent channels K'); ax.set_ylabel('Prior-evidence coefficient eta'); ax.set_zlabel('NDCG@10')
ax.set_title('Measured 8x8 Sensitivity: Musical Instruments')
ax.view_init(elev=28,azim=-58)
plt.tight_layout(); plt.savefig('measured_K_eta_8x8.png',dpi=300,bbox_inches='tight')
print('DONE',Z.tolist(),flush=True)
