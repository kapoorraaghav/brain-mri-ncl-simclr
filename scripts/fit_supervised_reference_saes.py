"""Fit archived supervised CLS vectors with the historical native SAE recipe."""
from pathlib import Path
import hashlib,io,json,os,random,time,zipfile
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/feature_population_audit_v1'


class SAE(nn.Module):
    def __init__(self):
        super().__init__();self.pre_bias=nn.Parameter(torch.zeros(192))
        self.encoder=nn.Linear(192,1536);self.decoder=nn.Linear(1536,192,bias=False)
    def forward(self,x):
        raw=F.relu(self.encoder(x-self.pre_bias));v,i=raw.topk(24,dim=1)
        a=torch.zeros_like(raw).scatter_(1,i,v)
        return self.decoder(a)+self.pre_bias,a


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this new SAE fitting stage')
    torch.use_deterministic_algorithms(True)
    with zipfile.ZipFile(ROOT/'data/rank_control_embeddings_v1.zip') as z:
        manifest=pd.read_csv(io.BytesIO(z.read('split_manifest.csv')))
    rows=[]
    for enc in [10,42,52]:
        cached=OUT/'supervised_embeddings'/f'encoder_{enc}'
        provenance=OUT/'supervised_embedding_provenance.json'
        if provenance.exists() and all((cached/f'{split}.npz').exists() for split in ['train','val']):
            prior=json.loads(provenance.read_text());arrays={}
            for split in ['train','val']:
                item=next(r for r in prior if r['encoder_seed']==enc and r['split']==split)
                path=cached/f'{split}.npz';assert sha(path)==item['embedding_sha256']
                with np.load(path,allow_pickle=False) as f:
                    assert np.array_equal(f['sample_ids'],manifest[manifest.split==split].sample_id.to_numpy())
                    arrays[split]=f['x'].copy()
                rows.append(item)
            source_sha=rows[-1]['source_archive_sha256']
        else:
            choices=list((Path.home()/'Downloads').glob(f'brats_no_checkpoints_supervised_seed{enc}_*_ALL_FILES.zip'))
            if len(choices)!=1:raise ValueError(f'Expected one source supervised archive for seed {enc}')
            source=choices[0];source_sha=sha(source)
            with zipfile.ZipFile(source) as z:
                arrays={}
                for split in ['train','val']:
                    name=next(n for n in z.namelist() if f'encoder_{enc}/embeddings/{split}_cls_' in n and n.endswith('.npz'))
                    raw=z.read(name)
                    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                        assert np.array_equal(f['sample_ids'],manifest[manifest.split==split].sample_id.to_numpy())
                        arrays[split]=f['x'].copy()
                    cached.mkdir(parents=True,exist_ok=True)
                    (cached/f'{split}.npz').write_bytes(raw)
                    rows.append({'method':'supervised','encoder_seed':enc,'split':split,'source_archive_sha256':source_sha,
                                 'embedding_member':name,'embedding_sha256':hashlib.sha256(raw).hexdigest()})
        mean=arrays['train'].mean(0,dtype=np.float64).astype(np.float32)
        train=arrays['train']-mean
        for seed in [10,42,52]:
            folder=OUT/'supervised_models'/f'encoder_{enc}'/f'sae_{seed}';folder.mkdir(parents=True,exist_ok=True)
            target=folder/'completed_model.pt'
            if target.exists():print('REUSE supervised SAE',enc,seed,flush=True);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=SAE().cuda();opt=torch.optim.Adam(model.parameters(),lr=5e-4)
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(train)),batch_size=512,shuffle=True)
            counts=torch.zeros(1536,dtype=torch.long,device='cuda');history=[]
            for epoch in range(30):
                total=0.;n=0;started=time.monotonic()
                for (batch,) in loader:
                    batch=batch.cuda();opt.zero_grad(set_to_none=True);rec,a=model(batch)
                    loss=F.mse_loss(rec,batch)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite supervised-reference SAE loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    counts+=(a.detach()>0).sum(0);total+=float(loss.detach())*len(batch);n+=len(batch)
                history.append({'epoch':epoch+1,'train_mse':total/n,'seconds':time.monotonic()-started})
                atomic_json(folder/'history.json',history)
                print('SUPERVISED SAE',enc,seed,epoch+1,round(total/n,10),'seconds',round(history[-1]['seconds'],2),flush=True)
            state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                   'training_activity':{'positive_counts':counts.cpu()},'history':history,
                   'transform':{'mean':torch.from_numpy(mean),'basis':torch.eye(192),'scale':torch.ones(192),'condition':'baseline','rank':192},
                   'protocol':{'method':'supervised','encoder_seed':enc,'sae_seed':seed,'dictionary_size':1536,'k':24,'epochs':30,
                               'lr':5e-4,'batch_size':512,'gradient_clip':1.,'init':'torch-default untied','decoder_normalization':False,
                               'source_archive_sha256':source_sha,'scope':'new SAE fits on archived supervised encoder embeddings; separate reference'}}
            temp=target.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,target)
            del model,opt;torch.cuda.empty_cache()
    atomic_json(OUT/'supervised_embedding_provenance.json',rows)
    print('SUPERVISED REFERENCE: 9 SAE fits complete; encoder weights not retrained.',flush=True)


if __name__=='__main__':main()
