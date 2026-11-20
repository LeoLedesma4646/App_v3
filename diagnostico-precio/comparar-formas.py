"""Pruebas adicionales de forma funcional y dinámica, sin cambiar la app."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum
from statsmodels.stats.multitest import multipletests
from scipy.linalg import qr
from scipy.stats import t
from modelos import DATA,fc
from elasticidad import CONTROLS

OUT=Path(__file__).resolve().parent

def controls(x,stock=False):
    cols=[c for c in CONTROLS if stock or c!='stock_available']
    W=x[cols].astype(float).copy()
    tau=(x.week-x.week.min()).dt.days/7
    W['trend']=tau/52
    for k in [1,2]:
        W[f'sin{k}']=np.sin(2*np.pi*k*tau/52)
        W[f'cos{k}']=np.cos(2*np.pi*k*tau/52)
    return W

def design(x,W,price_columns,cell=True):
    if cell:W=pd.concat([W,pd.get_dummies(x.cell,drop_first=True,dtype=float)],axis=1)
    for c in W:W[c]=(W[c]-W[c].mean())/(W[c].std(ddof=0) or 1.)
    W=sm.add_constant(W,has_constant='add')
    _,R,piv=qr(W.to_numpy(),mode='economic',pivoting=True)
    rank=int((np.abs(np.diag(R))>np.abs(R[0,0])*max(W.shape)*np.finfo(float).eps).sum())
    W=W.iloc[:,sorted(piv[:rank])]
    X=pd.concat([x[price_columns].astype(float),W],axis=1)
    assert np.linalg.matrix_rank(X.to_numpy())==X.shape[1]
    return X

def record(sku,method,result,X,groups,contrast):
    c=np.array([contrast.get(name,0.) for name in X.columns])
    beta=float(c@np.asarray(result.params));se=float(np.sqrt(c@np.asarray(result.cov_params())@c))
    G=int(pd.Series(groups).nunique());critical=t.ppf(.975,G-1)
    row={'sku':sku,'method':method,'epsilon':beta,'se_cluster':se,'ci_low':beta-critical*se,'ci_high':beta+critical*se,'p_cluster':float(2*t.sf(abs(beta/se),G-1)),'n':len(X),'weeks':G,'converged':bool(getattr(result,'converged',True))}
    time=pd.Series(groups).factorize(sort=True)[0]
    cov0=cov_nw_groupsum(result,0,time,use_correction='cluster')
    np.testing.assert_allclose(cov0,np.asarray(result.cov_params()),rtol=1e-7,atol=1e-10)
    for lag in [4,8]:
        cov=cov_nw_groupsum(result,lag,time,use_correction='cluster')
        dkse=float(np.sqrt(c@cov@c))
        row.update({f'se_dk{lag}':dkse,f'ci_low_dk{lag}':beta-critical*dkse,f'ci_high_dk{lag}':beta+critical*dkse,f'p_dk{lag}':float(2*t.sf(abs(beta/dkse),G-1))})
    return row

def main():
    data=fc.load_data(DATA);rows=[];measurement=[]
    for sku,g in data.groupby('sku'):
        hist=fc.aggregate_sku(data,sku)
        x=g[g.week.between(hist.week.min(),hist.week.max())].copy()
        x['cell']=x.channel.astype(str)+'|'+x.region.astype(str)
        x=x.sort_values(['cell','week'])
        x['ln_price']=np.log(x.price_unit);x['ln_units']=np.log(x.units_sold)
        for stock in [False,True]:
            X=design(x,controls(x,stock),['ln_price'])
            model=sm.GLM(x.units_sold.astype(float),X,family=sm.families.Poisson())
            result=model.fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':x.week.factorize()[0]})
            rows.append(record(sku,'PPML_'+('con_stock' if stock else 'sin_stock'),result,X,x.week,{'ln_price':1.}))
        # Solo variación entre semanas consecutivas de la misma celda.
        W=controls(x);all_values=pd.concat([x[['ln_price','ln_units']],W],axis=1)
        diffs=all_values.groupby(x.cell).diff()
        valid=x.groupby('cell').week.diff().dt.days.eq(7)
        xx=x.loc[valid].copy();xx['ln_price']=diffs.loc[valid,'ln_price']
        XX=design(xx,diffs.loc[valid,W.columns].copy(),['ln_price'],cell=False)
        rr=sm.OLS(diffs.loc[valid,'ln_units'],XX).fit(cov_type='cluster',cov_kwds={'groups':xx.week.factorize()[0]})
        rows.append(record(sku,'diferencias_semanales',rr,XX,xx.week,{'ln_price':1.}))
        # Respuesta acumulada a un cambio permanente, con dos rezagos de precio.
        x['ln_price_lag1']=x.groupby('cell').ln_price.shift(1)
        x['ln_price_lag2']=x.groupby('cell').ln_price.shift(2)
        valid=x.groupby('cell').week.diff(2).dt.days.eq(14)
        xx=x.loc[valid].copy();pcols=['ln_price','ln_price_lag1','ln_price_lag2']
        XX=design(xx,controls(xx),pcols)
        rr=sm.OLS(xx.ln_units,XX).fit(cov_type='cluster',cov_kwds={'groups':xx.week.factorize()[0]})
        rows.append(record(sku,'precio_actual_y_2_rezagos_acumulado',rr,XX,xx.week,{c:1. for c in pcols}))
        # Pendientes distintas con y sin promoción; no presupone un único epsilon.
        xx=x.copy();xx['ln_price_promo']=xx.ln_price*xx.promotion_flag
        XX=design(xx,controls(xx),['ln_price','ln_price_promo'])
        rr=sm.OLS(xx.ln_units,XX).fit(cov_type='cluster',cov_kwds={'groups':xx.week.factorize()[0]})
        for promo in [False,True]:
            rows.append(record(sku,'pendiente_'+('con_promo' if promo else 'sin_promo'),rr,XX,xx.week,{'ln_price':1.,'ln_price_promo':float(promo)}))
        weekly=x.assign(revenue=x.units_sold*x.price_unit).groupby('week').agg(units=('units_sold','sum'),revenue=('revenue','sum'),simple_price=('price_unit','mean'))
        weekly['unit_value']=weekly.revenue/weekly.units
        measurement.append({'sku':sku,'relative_price_definition_difference_recent_pct':100*(weekly.unit_value.tail(4).mean()/weekly.simple_price.tail(4).mean()-1),'simple_reference_recent':weekly.simple_price.tail(4).mean(),'unit_value_recent':weekly.unit_value.tail(4).mean(),'weeks':len(weekly)})
    result=pd.DataFrame(rows)
    for pcol in ['p_cluster','p_dk4','p_dk8']:
        result[pcol+'_holm']=multipletests(result[pcol],method='holm')[1]
    result.to_csv(OUT/'formas-adicionales.csv',index=False)
    pd.DataFrame(measurement).to_csv(OUT/'definiciones-precio.csv',index=False)
    stats=result.groupby('method').agg(negative=('epsilon',lambda s:int((s<0).sum())),negative_ci=('ci_high',lambda s:int((s<0).sum())),positive_ci=('ci_low',lambda s:int((s>0).sum())),negative_ci_dk4=('ci_high_dk4',lambda s:int((s<0).sum())),negative_ci_dk8=('ci_high_dk8',lambda s:int((s<0).sum())),converged=('converged','sum')).reset_index().to_dict('records')
    receipt={'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'scope':'Diagnósticos exploratorios; ninguna variante resuelve por sí sola la endogeneidad. IC individuales agrupados por semana y sensibilidad Driscoll-Kraay (Bartlett, 4/8 rezagos). Se añaden p Holm para la familia exploratoria de 180 contrastes, separadamente por cada covarianza.','lag0_matches_cluster':True,'holm_rejections':{pcol:int((result[pcol+'_holm']<.05).sum()) for pcol in ['p_cluster','p_dk4','p_dk8']},'summary':stats,'ju021':result[result.sku.eq('JU-021')].to_dict('records'),'yo009':result[result.sku.eq('YO-009')].to_dict('records'),'ju021_price_definitions':next(r for r in measurement if r['sku']=='JU-021')}
    (OUT/'formas-adicionales.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
