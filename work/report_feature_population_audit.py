from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/feature_population_audit_v1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'ncl':'#16866b','simclr':'#2477b4','supervised':'#ad5c27'}
LABELS={'ncl':'NCL','simclr':'SimCLR','supervised':'Supervised'}


def main():
    per=pd.read_csv(OUT/'per_encoder.csv')
    fits=pd.read_csv(OUT/'fit_summary.csv')
    assert len(fits)==27
    budget=pd.read_csv(OUT/'budget_per_encoder.csv')
    bf=pd.read_csv(OUT/'budget_fits.csv')
    summary={}
    for method in COLORS:
        group=per[per.method==method]
        summary[method]={k:{'mean':float(group[k].mean()),'sd':float(group[k].std())}
            for k in ['eligible_features','mean_purity','mean_global_excess','mean_stratified_excess',
                      'rho_purity_utility','rho_stratified_excess_utility','mean_stratified3_excess','mean_stratified5_excess']}
    paired=[]
    for selector in budget.selector.unique():
        b=budget[budget.selector==selector]
        methods={m:b[b.method==m].set_index(['encoder_seed','J']) for m in ['ncl','simclr']}
        shared=methods['ncl'].index.intersection(methods['simclr'].index)
        for index in shared:
            n=methods['ncl'].loc[index];s=methods['simclr'].loc[index]
            paired.append({'selector':selector,'encoder_seed':int(index[0]),'J':int(index[1]),
                           'gap':n.purity_conditional-s.purity_conditional,
                           'lower':n.purity_lower-s.purity_upper,'upper':n.purity_upper-s.purity_lower})
    paired=pd.DataFrame(paired)
    # Every displayed mean/SD uses all three encoders with all three SAE fits each.
    valid=paired.groupby(['selector','J']).encoder_seed.nunique()
    paired=paired.set_index(['selector','J']).loc[valid[valid==3].index].reset_index()
    paired.to_csv(OUT/'budget_paired.csv',index=False)
    eligible=paired[paired.selector=='validation_eligible_training_usage']
    g=eligible.groupby('J').gap.agg(['mean','std'])
    signs=np.sign(g['mean'].to_numpy())
    crossings=[int(j) for j in g.index[1:][signs[1:]!=signs[:-1]]]
    report={'methods':summary,'common_absolute_budget':[int(g.index.min()),int(g.index.max())],
            'conditional_sign_change_J':crossings,
            'gap_at_budgets':{str(j):float(g.loc[j,'mean']) for j in [8,32,64,80] if j in g.index},
            'null_gap_primary':summary['ncl']['mean_stratified_excess']['mean']-summary['simclr']['mean_stratified_excess']['mean'],
            'null_gap_sensitivity':{str(n):float(per[per.method=='ncl'][f'mean_stratified{n}_excess'].mean()-per[per.method=='simclr'][f'mean_stratified{n}_excess'].mean()) for n in [3,5]},
            'scope':'Exploratory validation; curves have equal replication only on the common plotted domain; sign changes are descriptive, not change-point estimates.'}
    (OUT/'AUDIT_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    fig,axes=plt.subplots(2,2,figsize=(10.2,6.2))
    ax=axes[0,0]
    raw=bf[bf.selector=='validation_eligible_training_usage']
    for method,color in COLORS.items():
        group=raw[raw.method==method]
        for _,f in group.groupby(['encoder_seed','sae_seed']):
            ax.plot(f.J,100*f.purity_conditional,color=color,alpha=.14,lw=.7)
        means=budget[(budget.method==method)&(budget.selector=='validation_eligible_training_usage')]
        count=means.groupby('J').encoder_seed.nunique()
        means=means[means.J.isin(count[count==3].index)]
        s=means.groupby('J').purity_conditional.mean()
        ax.plot(s.index,100*s,color=color,lw=1.8,label=LABELS[method])
    ax.set(xlabel='Usage-ranked eligible features J',ylabel='Top-10 label purity (%)',
           title='A  Curves extend to each dictionary endpoint')
    ax.legend(fontsize=8,frameon=False)
    ax=axes[0,1]
    ax.axhline(0,color='black',lw=.8)
    boot_path=OUT/'patient_budget_bootstrap/curve_intervals.csv'
    if boot_path.exists():
        boot=pd.read_csv(boot_path).set_index('J').reindex(g.index)
        assert np.allclose(boot.point_gap,g['mean'],atol=1e-12)
        ax.fill_between(g.index,100*boot.outer_percentile_lower,100*boot.outer_percentile_upper,
                        color='#697aac',alpha=.20,label='Patient bootstrap + support bounds')
        report['patient_budget_bootstrap']=json.loads((boot_path.parent/'SUMMARY.json').read_text())
        (OUT/'AUDIT_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    else:
        ax.fill_between(g.index,100*(g['mean']-g['std']),100*(g['mean']+g['std']),color='#697aac',alpha=.16,label='Encoder SD')
    ax.plot(g.index,100*g['mean'],color='#40578b',lw=1.6)
    ax.scatter([8,32,64,80],100*g.loc[[8,32,64,80],'mean'],color='#40578b',s=17)
    ax.set(xlabel='Common absolute feature budget J',ylabel='NCL − SimCLR purity (pp)',
           title=('B  Paired patient-bootstrap envelope' if boot_path.exists() else 'B  Equal-budget gap and encoder SD'))
    ax.legend(fontsize=7,frameon=False,loc='lower right')
    ax=axes[1,0]
    t=paired[paired.selector=='training_only_usage'].groupby('J')[['gap','lower','upper']].mean()
    ax.axhline(0,color='black',lw=.8)
    ax.fill_between(t.index,100*t.lower,100*t.upper,color='#16866b',alpha=.19,label='Unscorable-feature bounds')
    ax.plot(t.index,100*t.gap,color='#16866b',lw=1.3,ls='--',label='Scorable-only mean')
    ax.set(xlabel='Training-selected feature budget J',ylabel='NCL − SimCLR purity (pp)',
           title='C  Missing support limits training-only rankings')
    ax.legend(fontsize=7,frameon=False)
    ax=axes[1,1]
    ax.axhline(0,color='black',lw=.8)
    for method,color in COLORS.items():
        b=budget[(budget.selector=='validation_eligible_training_usage')&(budget.method==method)&(budget.J<=g.index.max())]
        s=b.groupby('J').usage_minus_random.agg(['mean','std'])
        # Reconstruct the exact independent-per-dictionary random draws, then
        # aggregate draws in the same order as reported macro-averages.
        draws=[];js=np.arange(int(g.index.min()),int(g.index.max())+1)
        for folder in sorted((OUT/'fits'/method).glob('encoder_*/sae_*')):
            feature=pd.read_csv(folder/'features.csv');pool=np.flatnonzero(feature.purity.notna())
            enc=int(folder.parent.name.split('_')[-1]);seed=int(folder.name.split('_')[-1])
            rng=np.random.default_rng(783013+enc*100+seed)
            values=[]
            for _ in range(1000):
                order=rng.permutation(pool)
                values.append(np.cumsum(feature.purity.to_numpy()[order])[js-1]/js)
            draws.append(np.asarray(values))
        assert len(draws)==9
        drawmean=np.mean(np.stack(draws).reshape(3,3,1000,len(js)),axis=1).mean(0)
        usage=b.groupby('J').purity_conditional.mean().reindex(js).to_numpy()
        q025,q975=np.quantile(drawmean,[.025,.975],axis=0)
        assert np.allclose(usage-drawmean.mean(0),s['mean'].reindex(js),atol=1e-12)
        ax.fill_between(js,100*(usage-q975),100*(usage-q025),color=color,alpha=.12)
        np.savez_compressed(OUT/f'random_reference_means_{method}.npz',J=js,random_macro_means=drawmean,
                            usage_macro_mean=usage,q025=q025,q975=q975)
        ax.plot(s.index,100*s['mean'],color=color,lw=1.6,label=LABELS[method])
    ax.set(xlabel='Feature budget J',ylabel='Usage-ranked − random mean (pp)',
           title='D  Selection rule changes the gallery average')
    ax.legend(fontsize=8,frameon=False)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.tight_layout(pad=1.1)
    fig.savefig(OUT/'feature_budget_audit.pdf',bbox_inches='tight')
    fig.savefig(OUT/'feature_budget_audit.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (OUT/'RESULTS_README.md').write_text('''# Feature-population audit results

`audit_protocol.json` freezes analysis settings before this run. All results reuse
the historical 37-patient validation cohort. NCL/SimCLR use the archived baseline
SAEs; the supervised reference adds nine freshly fitted SAEs on saved supervised
encoder embeddings, using the historical baseline initialization/optimizer recipe.
No encoder was retrained and no test images or new patients were evaluated.

`fit_summary.csv`, `per_encoder.csv` and `summary.csv` report feature-level Spearman
correlations and global/stratified null excess. Orientations use training tumour
labels only. Within-feature utility is univariate binary AUROC, not causal importance,
ablation utility, or evidence about interactions. Feature-level p-values are not used.

Stratification explicitly preserves zero tumour area and positive-area quantile bins
crossed with axial-index bins. It conditions on segmentation-derived covariates;
this is not a deployable label-free adjustment or a causal confound correction.
`strata_definition.json` exposes cutpoints, cell sizes, patients and class composition.
Three/five-bin sensitivities accompany the primary four-bin reference. Both nulls
use 1,000 fixed-seed permutations; the original paper used 100 global permutations.

`budget_fits.csv` extends each dictionary's curve to its own eligible endpoint.
Equal-J paired plots restrict to budgets with all 3 SAE seeds in all 3 encoder seeds
for both NCL/SimCLR. Beyond that domain, there is no fully replicated equal-J method
comparison. Method-specific full-set endpoints are not matched-budget comparisons.
Bounds preserve frozen training-selected IDs and assign unscorable values [0,1].

Random subsets use 1,000 uniform random permutations without replacement from the
same candidate pool; subsets are nested as J increases. The CSV contains per-fit
reference means, 2.5/97.5% quantiles and descriptive usage percentiles. These are
conditional feature-selection references, not patient confidence intervals/p-values.

Means average SAE seeds within encoder first; SD is across three encoder means.
Faint Figure A lines are individual dictionaries; thick lines have full replication
only. Figure B now uses the paired patient-bootstrap/support envelope when its outputs exist; otherwise it shows encoder SD. C shading is deterministic support bounds, not patient uncertainty.
Figure D shading uses 2.5/97.5% quantiles after aggregating 1,000 random-draw means
over the nine dictionaries in each method. It is a conditional random-selection
reference for the macro-average, not patient/model-training uncertainty.
Sign changes are descriptive crossing locations, not estimated change points.
''',encoding='utf-8')
    if boot_path.exists():
        readme=OUT/'RESULTS_README.md'
        text=readme.read_text(encoding='utf-8')
        text += "\n## Paired patient-budget bootstrap\n\n2,000 common draws resample whole validation patients with replacement, retaining all slices and duplicate multiplicity. Models and original selected IDs stay fixed. Top-ten lists are recomputed. Figure B uses the 2.5th percentile of the gap lower bound and 97.5th percentile of its upper bound when missing support makes purity undefined. This pointwise conditional envelope is not encoder SD, a simultaneous band or guaranteed nominal coverage. Strict complete-case intervals require at least 95% fully scorable draws. Null and correlation patient intervals are reported separately when patient_null_correlation_bootstrap/intervals.csv exists. Outputs, fixed IDs and draw counts are under patient_budget_bootstrap/. Reproduce with `python scripts/bootstrap_feature_budget_patients.py` before rebuilding the figure.\n"
        readme.write_text(text,encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
