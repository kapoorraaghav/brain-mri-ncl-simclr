"""Oracle synthetic controls for what the fixed conditional reference can detect."""
from pathlib import Path
import importlib.util,io,json,zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/feature_population_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    with zipfile.ZipFile(audit.EMBED) as z:
        frame=pd.read_csv(io.BytesIO(z.read('split_manifest.csv')))
    val=frame[frame.split=='val'].reset_index(drop=True);cov=pd.read_csv(audit.COV)
    assert np.array_equal(val.sample_id,cov.sample_id)
    labels=val.dominant_label.to_numpy();s,_=audit.strata(cov,4)
    # Fixed diagnostic rule, not selection of a favorable learned feature.
    cell=next(int(c) for c in sorted(np.unique(s))
              if ((s==c)&(labels==4)).sum()>=10 and len(np.unique(labels[s==c]))>=2)
    absent=np.flatnonzero(cov.tumour_area_fraction.to_numpy()==0)[:10]
    target=np.flatnonzero((s==cell)&(labels==4))[:10]
    top=np.stack([absent,target]);classes=np.unique(labels)
    raw=np.stack([(labels[top]==c).mean(1) for c in classes]).max(0)
    global_null=audit.null_mean(top,audit.permuted_labels(labels),classes)
    conditional=audit.null_mean(top,audit.permuted_labels(labels,s),classes)
    assert np.allclose(raw,1)
    assert np.isclose(conditional[0],1) and 1-conditional[1]>.05
    rows=[]
    for i,name in enumerate(['absence_only','within_cell_label4_oracle']):
        rows.append({'control':name,'known_top10_purity':float(raw[i]),'global_null':float(global_null[i]),
                     'stratified_null':float(conditional[i]),'stratified_excess':float(raw[i]-conditional[i])})
    result={'scope':'synthetic oracle sensitivity diagnostic, not learned features or evidence of generalization',
            'protocol':'absence-only indicator; label4 indicator in first sorted mixed cell with >=10 label4 examples; first-ten index ties',
            'primary_bins':4,'permutations':audit.PLAN['null_permutations'],'target_cell':cell,
            'target_cell_slices':int((s==cell).sum()),
            'target_cell_label_counts':{str(c):int(((s==cell)&(labels==c)).sum()) for c in classes},
            'controls':rows,'status':'passed'}
    audit.atomic_json(audit.OUT/'null_sensitivity_controls.json',result)
    pd.DataFrame(rows).to_csv(audit.OUT/'null_sensitivity_controls.csv',index=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
