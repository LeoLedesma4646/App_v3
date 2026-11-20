"""Contrasta exposición semanal y sesgo de agregación; conserva la base recibida."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from modelos import DATA,fc
from elasticidad import CONTROLS

OUT=Path(__file__).resolve().parent
KEY=['sku','week','channel','region']

def design(g,exposure=False,geometric=False,calendar=True):
    X=pd.DataFrame(index=g.index)
    X['ln_price']=g.price_available_logmean if geometric else np.log(g.price_unit)
    if exposure:
        X['promotion_share']=g.promotion_available_share
    else:X['promotion_flag']=g.promotion_flag.astype(float)
    if calendar:
        tau=(g.week-g.week.min()).dt.days/365.25
        X['trend']=tau
        for k in [1,2]:X[f'sin{k}']=np.sin(2*np.pi*k*tau);X[f'cos{k}']=np.cos(2*np.pi*k*tau)
    cats=pd.DataFrame({'cell':g.channel+'|'+g.region},index=g.index)
    X=pd.concat([X,pd.get_dummies(cats,drop_first=True,dtype=float)],axis=1)
    return sm.add_constant(X).astype(float)

def record(sku,spec,result,X,g):
    G=g.week.nunique();b=float(result.params['ln_price']);se=float(result.bse['ln_price']);c=t.ppf(.975,G-1)
    return {'sku':sku,'specification':spec,'epsilon':b,'se':se,'ci_low':b-c*se,'ci_high':b+c*se,'p':float(2*t.sf(abs(b/se),G-1)),'n':len(g),'weeks':G}

def main():
    panel=fc.load_data(DATA)
    feat=pd.read_csv(OUT/'exposicion-semanal-reconstruida.csv',parse_dates=['week'])
    data=panel.merge(feat,on=KEY,validate='one_to_one')
    rows=[];assoc=[]
    for sku,all_g in data.groupby('sku'):
        hist=fc.aggregate_sku(panel,sku)
        g=all_g[all_g.week.between(hist.week.min(),hist.week.max())&all_g.in_stock_days.gt(0)].copy()
        lnP=np.log(g.price_unit);lnQ=np.log(g.units_sold)
        assoc.append({'sku':sku,'corr_log_price_days':lnP.corr(g.observed_days),'corr_log_units_days':lnQ.corr(g.observed_days),'corr_logprice_logmeanprice':lnP.corr(g.price_available_logmean),'mean_days':g.observed_days.mean()})
        groups=g.week.factorize()[0]
        original=g[CONTROLS].copy();original.insert(0,'ln_price',lnP)
        original=pd.concat([original,pd.get_dummies(g[['channel','region']].astype(str),drop_first=True,dtype=float)],axis=1)
        original=sm.add_constant(original).astype(float)
        for spec,X,offset in [('ols_original_plus_exposure',original.assign(ln_observed_days=np.log(g.observed_days)),None),('ols_rate_actual_promotion',design(g,True),np.log(g.in_stock_days)),('ppml_weekly_total',design(g),None),('ppml_rate_arithmetic',design(g,True),np.log(g.in_stock_days)),('ppml_rate_geometric',design(g,True,True),np.log(g.in_stock_days))]:
            if spec.startswith('ols'):
                y=lnQ if offset is None else lnQ-offset
                r=sm.OLS(y,X).fit(cov_type='cluster',cov_kwds={'groups':groups})
            else:r=sm.GLM(g.units_sold,X,offset=offset,family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':groups})
            rows.append(record(sku,spec,r,X,g))
    pd.DataFrame(rows).to_csv(OUT/'contrastes-exposicion.csv',index=False)
    pd.DataFrame(assoc).to_csv(OUT/'asociaciones-exposicion.csv',index=False)
    # Simulación de medición: precio independiente y venta diaria constante.
    # La distribución de días es la observada. No genera ni sustituye datos fuente.
    rng=np.random.default_rng(805);n=data.observed_days.to_numpy(int)
    slopes=[];correlations=[]
    for _ in range(200):
        p=rng.uniform(1.5,9.,(len(n),7));pbar=(p*(np.arange(7)[None,:]<n[:,None])).sum(axis=1)/n
        xp=np.log(pbar);y=np.log(n);slopes.append(float(np.cov(xp,y,ddof=0)[0,1]/np.var(xp)));correlations.append(float(np.corrcoef(xp,y)[0,1]))
    summaries={spec:{'median_epsilon':float(x.epsilon.median()),'positive':int((x.epsilon>0).sum()),'ci_excludes_zero_negative':int((x.ci_high<0).sum()),'ci_excludes_zero_positive':int((x.ci_low>0).sum())} for spec,x in pd.DataFrame(rows).groupby('specification')}
    result={'scope':'Contrastes de especificación y medición, no identificación causal. Offset normaliza exposición registrada, no imputa días ausentes ni demanda perdida. Simulación independiente ilustra un mecanismo, no prueba el generador.', 'models':summaries,'ju021':[r for r in rows if r['sku']=='JU-021'],'price_day_association':{'median_corr_logprice_days':float(pd.DataFrame(assoc).corr_log_price_days.median()),'median_corr_logunits_days':float(pd.DataFrame(assoc).corr_log_units_days.median())},'independent_price_simulation':{'replications':len(slopes),'price_range':[1.5,9.0],'daily_sales_constant':1,'ols_log_total_on_log_meanprice_mean':float(np.mean(slopes)),'ols_slope_95_simulation_quantiles':[float(v) for v in np.quantile(slopes,[.025,.975])],'mean_correlation':float(np.mean(correlations))}}
    (OUT/'contrastes-exposicion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
