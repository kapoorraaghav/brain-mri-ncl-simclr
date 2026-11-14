from pathlib import Path
import pickle
import struct
import numpy as np

MAGIC=b'BRATSPK1'

def read_header(path):
    with open(path,'rb') as stream:
        header=pickle.load(stream)
    if header.get('format')!='brats-preprocessed-indexed-pickle-v1':
        raise ValueError('Unsupported dataset package')
    return header

def read_index(path):
    with open(path,'rb') as stream:
        stream.seek(-16,2)
        position,magic=struct.unpack('<Q8s',stream.read(16))
        if magic!=MAGIC:raise ValueError('Incomplete or invalid dataset file')
        stream.seek(position)
        return pickle.load(stream)

def iter_samples(path,split=None):
    """Yield (metadata, image, seg_mask) without loading the full dataset."""
    with open(path,'rb') as stream:
        header=pickle.load(stream)
        if header.get('format')!='brats-preprocessed-indexed-pickle-v1':raise ValueError('Unsupported format')
        for metadata in header['metadata']:
            record=pickle.load(stream)
            if split is None or metadata['split']==split:
                yield metadata,record['image'],record['seg_mask']

class PreparedDataset:
    """Random access to one split; only metadata/index stay in memory.

    Returns dictionaries with image, seg_mask and metadata. Add tensor conversion
    and the original augmentations in the training dataset wrapper as needed.
    """
    def __init__(self,path,split=None):
        self.path=str(Path(path))
        self.header=read_header(path)
        self.offsets=read_index(path)['offsets']
        if len(self.offsets)!=len(self.header['metadata']):raise ValueError('Index length mismatch')
        self.indices=[i for i,m in enumerate(self.header['metadata']) if split is None or m['split']==split]
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        source_index=self.indices[index]
        with open(self.path,'rb') as stream:
            stream.seek(self.offsets[source_index])
            record=pickle.load(stream)
        return {**record,'metadata':self.header['metadata'][source_index]}
