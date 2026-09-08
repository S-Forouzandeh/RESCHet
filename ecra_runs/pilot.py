import urllib.request, json
urls={
'reviews':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments_5.json.gz',
'meta':'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Musical_Instruments.json.gz'}
out={}
for k,u in urls.items():
    req=urllib.request.Request(u,headers={'Range':'bytes=0-1023','User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        b=r.read(1024); out[k]={'status':r.status,'bytes':len(b),'content_type':r.headers.get('Content-Type')}
        print(k,out[k],flush=True)
json.dump(out,open('hf_probe.json','w'),indent=2)
