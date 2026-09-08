from huggingface_hub import HfApi
import json
api=HfApi()
repo='genghaobuaa/Amazon-Review-Data-2018'
info=api.dataset_info(repo, files_metadata=True)
print('DATASET', repo)
print('siblings', len(info.siblings or []))
rows=[]
for s in info.siblings or []:
    name=s.rfilename
    if any(k.lower() in name.lower() for k in ['musical','office','grocery','clothing','meta']):
        size=getattr(s,'size',None)
        rows.append({'path':name,'size':size})
        print(name,size)
json.dump(rows,open('hf_probe.json','w'),indent=2)
