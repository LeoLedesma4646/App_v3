"""Auditoría reproducible: controles, efectos fijos y estabilidad temporal.

No modifica fuentes ni coeficientes operativos. FWL y covarianza agrupada
reproducen OLS mediante NumPy, independientemente del ajuste statsmodels.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import t
from modelos import DATA,fc,ROOT
from elasticidad import CONTROLS

OUT=Path(__file__).resolve().parent

def fit(x,spec):
    controls=list(CONTROLS)
    if 'sin_stock' in spec:controls.remove('stock_available')
    if spec.startswith('simple'):controls=[]
    w=x[controls].astype(float).copy()
    if not spec.startswith('simple'):
        if spec.startswith('original'):
            cats=x[['channel','region']].astype(str)
        else:cats=pd.DataFrame({'cell':x.channel.astype(str)+'|'+x.region.astype(str)},index=x.index)
        w=pd.concat([w,pd.get_dummies(cats,drop_first=True,dtype=float)],axis=1)
    if spec.startswith('tendencia'):
        tau=(x.week-x.week.min()).dt.days/7
        w['trend']=tau/52
        for k in [1,2]:
            w[f'sin{k}']=np.sin(2*np.pi*k*tau/52)
            w[f'cos{k}']=np.cos(2*np.pi*k*tau/52)
    if spec.startswith('semana'):
        w=pd.concat([w,pd.get_dummies(x.week.astype(str),drop_first=True,dtype=float)],axis=1)
    for c in w:w[c]=(w[c]-w[c].mean())/(w[c].std(ddof=0) or 1)
    W=np.column_stack([np.ones(len(x)),w.to_numpy()])
    # SVD garantiza proyección única aunque calendario/FE sean redundantes.
    U,s,_=np.linalg.svd(W,full_matrices=False)
    rank=int((s>s[0]*max(W.shape)*np.finfo(float).eps).sum())
    U=U[:,:rank]
    p=np.log(x.price_unit.to_numpy());y=np.log(x.units_sold.to_numpy())
    rp=p-U@(U.T@p);ry=y-U@(U.T@y)
    denom=float(rp@rp);beta=float(rp@ry/denom)
    residual=ry-beta*rp
    groups=x.week.factorize()[0];G=int(groups.max()+1);n=len(x);k=rank+1
    scores=np.bincount(groups,weights=rp*residual)
    se=float(np.sqrt((scores@scores)/(denom**2)*G/(G-1)*(n-1)/(n-k)))
    critical=t.ppf(.975,G-1)
    return {'spec':spec,'epsilon':beta,'se_cluster':se,'ci_low':beta-critical*se,'ci_high':beta+critical*se,'p_cluster':float(2*t.sf(abs(beta/se),G-1)),'n':n,'weeks':G,'rank':k,'partial_price_log_sd':float(np.std(rp)),'partial_price_share':float(np.var(rp)/np.var(p))}

def main():
    data=fc.load_data(DATA)
    quality={'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'rows':len(data),'sku_count':int(data.sku.nunique()),'duplicates_sku_week_channel_region':int(data.duplicated(['sku','week','channel','region']).sum()),'missing_raw_units_recovered':data.attrs.get('recovered_sales',[]),'nonpositive_units':int((data.units_sold<=0).sum()),'nonpositive_price':int((data.price_unit<=0).sum()),'missing':data[['units_sold','price_unit',*CONTROLS]].isna().sum().astype(int).to_dict()}
    rows=[];signals=[]
    for sku,g in data.groupby('sku'):
        hist=fc.aggregate_sku(data,sku)
        x=g[g.week.between(hist.week.min(),hist.week.max())].copy()
        for c in ['units_sold','price_unit',*CONTROLS]:x[c]=pd.to_numeric(x[c],errors='coerce').astype(float)
        x=x[(x.units_sold>0)&(x.price_unit>0)].dropna(subset=['units_sold','price_unit',*CONTROLS])
        for spec in ['simple','original','original_sin_stock','tendencia','tendencia_sin_stock','semana','semana_sin_stock']:
            rows.append({'sku':sku,'sample':'total',**fit(x,spec)})
        median_week=sorted(x.week.unique())[x.week.nunique()//2]
        for label,part in [('primera_mitad',x[x.week<median_week]),('segunda_mitad',x[x.week>=median_week])]:
            rows.append({'sku':sku,'sample':label,**fit(part,'tendencia_sin_stock')})
        signals.append({'sku':sku,'n':len(x),'weeks':x.week.nunique(),'corr_log_price_units':np.corrcoef(np.log(x.price_unit),np.log(x.units_sold))[0,1],'corr_stock_units':x.stock_available.corr(x.units_sold),'corr_price_promo':x.price_unit.corr(x.promotion_flag),'corr_price_week':x.price_unit.corr((x.week-x.week.min()).dt.days),'mean_price_promo':x.loc[x.promotion_flag>0,'price_unit'].mean(),'mean_price_no_promo':x.loc[x.promotion_flag==0,'price_unit'].mean()})
    results=pd.DataFrame(rows);results.to_csv(OUT/'especificaciones.csv',index=False)
    pd.DataFrame(signals).to_csv(OUT/'senales-datos.csv',index=False)
    old=pd.read_csv(ROOT/'artifacts/elasticity.csv').set_index('sku')
    original=results.query("spec=='original' and sample=='total'").set_index('sku')
    quality['independent_max_abs_epsilon_error']=float((original.epsilon-old.elasticity).abs().max())
    quality['independent_max_abs_se_error']=float((original.se_cluster-old.se_cluster).abs().max())
    quality['raw_positive_sku_count']=int((old.elasticity>0).sum())
    quality['raw_cluster_ci_includes_zero_count']=int(((old.ci_low<=0)&(old.ci_high>=0)).sum())
    summaries=results.query("sample=='total'").groupby('spec').agg(positive=('epsilon',lambda s:int((s>0).sum())),negative=('epsilon',lambda s:int((s<0).sum())),significantly_negative=('ci_high',lambda s:int((s<0).sum())),significantly_positive=('ci_low',lambda s:int((s>0).sum())),median_epsilon=('epsilon','median'))
    quality['spec_summary']=summaries.reset_index().to_dict('records')
    halves=results.query("sample!='total'").pivot(index='sku',columns='sample',values='epsilon')
    quality['split_sign_changes']=int((np.sign(halves.primera_mitad)!=np.sign(halves.segunda_mitad)).sum())
    quality['ju021']=results[results.sku.eq('JU-021')].to_dict('records')
    (OUT/'auditoria.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in quality.items() if k not in ['missing','missing_raw_units_recovered']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
