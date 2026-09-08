import json, urllib.request
repo='genghaobuaa/Amazon-Review-Data-2018'
url=f'https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&expand=false'
req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req,timeout=60) as r:
    data=json.load(r)
rows=[]
for x in data:
    p=x.get('path','')
    if any(k in p.lower() for k in ['musical','office','grocery','clothing','meta']):
        rows.append({'path':p,'size':x.get('size')})
        print(p,x.get('size'),flush=True)
json.dump(rows,open('hf_probe.json','w'),indent=2)
print('MATCHES',len(rows))
