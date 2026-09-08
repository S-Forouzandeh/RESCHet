import gzip, json, math, os, random, urllib.request
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F

REV='https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz'
SEEDS=[2026]
VARIANTS=['full','no_dynamic_preference','no_intent_module','no_temporal_graph','simple_aggregation']
EPOCHS=3
K=6; D=32

def fetch(url,path):
    if not os.path.exists(path): urllib.request.urlretrieve(url,path)

def load():
    fetch(REV,'musical.json.gz')
    by=defaultdict(list); items={}
    with gzip.open('musical.json.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); u=r['reviewerID']; i=r['asin']; t=int(r['unixReviewTime'])
            if i not in items: items[i]=len(items)
            by[u].append((t,items[i]))
    seqs=[]
    for u,x in by.items():
        x=sorted(set(x)); ids=[i for _,i in x]
        if len(ids)>=5: seqs.append(ids)
    return seqs,len(items)

class M(nn.Module):
    def __init__(self,n,var):
        super().__init__(); self.var=var
        self.item=nn.Embedding(n,D); self.proto=nn.Parameter(torch.randn(K,D)/math.sqrt(D))
        self.proj=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(K)])
        self.gru=nn.GRUCell(D,D); self.A=nn.Parameter(torch.zeros(K,K)); self.W=nn.Linear(D,D,bias=False); self.out=nn.Linear(D,D,bias=False)
        nn.init.xavier_uniform_(self.item.weight)
    def encode(self,h,state,alpha):
        e=self.item(h); logits=e@self.proto.t(); p=F.softmax(logits,-1)
        if self.var=='no_intent_module': p=torch.full_like(p,1/K); logits=torch.zeros_like(logits)
        zs=torch.stack([q(e) for q in self.proj],1)
        if self.var=='no_dynamic_preference': state=state+p.unsqueeze(-1)*zs
        else:
            ns=[]
            for k in range(K): ns.append(self.gru(p[:,k:k+1]*zs[:,k],state[:,k]))
            state=torch.stack(ns,1)
        if self.var=='no_temporal_graph': T=F.softmax(self.A,-1).unsqueeze(0).expand(h.size(0),-1,-1)
        elif self.var=='no_intent_module': T=torch.eye(K,device=h.device).unsqueeze(0).expand(h.size(0),-1,-1)
        else:
            c=torch.einsum('bkd,bjd->bkj',self.W(state),state)/math.sqrt(D)
            T=F.softmax(self.A.unsqueeze(0)+c,-1)
        prior=torch.bmm(alpha.unsqueeze(1),T).squeeze(1)
        if self.var=='simple_aggregation' or self.var=='no_intent_module': pref=state.mean(1)
        else: pref=torch.einsum('bk,bkd->bd',prior,state)
        alpha=F.softmax(.5*torch.log(prior.clamp_min(1e-8))+.5*logits,-1)
        return state,alpha,self.out(pref)

def train_eval(seqs,n,var,seed):
    random.seed(seed); torch.manual_seed(seed); m=M(n,var); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    train=[s[:-2] for s in seqs]; val=[s[-2] for s in seqs]; test=[s[-1] for s in seqs]
    for ep in range(EPOCHS):
        order=list(range(len(train))); random.shuffle(order)
        for ix in order:
            s=train[ix]
            if len(s)<2: continue
            state=torch.zeros(1,K,D); alpha=torch.full((1,K),1/K)
            seen=set(s)
            for a,b in zip(s[:-1],s[1:]):
                state,alpha,pref=m.encode(torch.tensor([a]),state,alpha)
                neg=random.randrange(n)
                while neg in seen: neg=random.randrange(n)
                pos=(pref*m.item(torch.tensor([b]))).sum(-1); ns=(pref*m.item(torch.tensor([neg]))).sum(-1)
                loss=-F.logsigmoid(pos-ns).mean(); opt.zero_grad(); loss.backward(); opt.step(); state=state.detach(); alpha=alpha.detach()
    recalls=[]; ndcgs=[]
    with torch.no_grad():
        for s,t in zip([x[:-1] for x in seqs],test):
            state=torch.zeros(1,K,D); alpha=torch.full((1,K),1/K); pref=None
            for a in s:
                state,alpha,pref=m.encode(torch.tensor([a]),state,alpha)
            scores=(pref@m.item.weight.t()).squeeze(0); scores[torch.tensor(list(set(s)))]=-1e9
            rank=int((scores>scores[t]).sum())+1; recalls.append(float(rank<=10)); ndcgs.append(0 if rank>10 else 1/math.log2(rank+1))
    return sum(recalls)/len(recalls),sum(ndcgs)/len(ndcgs)

seqs,n=load(); print('users',len(seqs),'items',n)
out=[]
for v in VARIANTS:
    for s in SEEDS:
        r,n10=train_eval(seqs,n,v,s); row={'variant':v,'seed':s,'recall10':r,'ndcg10':n10}; out.append(row); print(row,flush=True)
json.dump(out,open('pilot_results.json','w'),indent=2)
