"""Pruebas de recuperación de efectos conocidos y agregación independiente."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from elasticidad_semanal import build,fit,predict,mean_jacobian

def synthetic(epsilon,seed=805):
    rng=np.random.default_rng(seed);rows=[]
    for w in pd.date_range('2022-01-03',periods=90,freq='W-MON'):
        for channel in ['A','B']:
            for day in range(7):
                if rng.random()<.15:continue
                price=rng.uniform(1.5,9.);promo=int(rng.random()<.15);stock=0 if rng.random()<.02 else 10000
                mu=150*(1.2 if channel=='B' else 1)*np.exp(.6*promo)*(price/5.25)**epsilon
                rows.append({'sku':'TEST','week':w,'date':w+pd.Timedelta(days=day),'channel':channel,'region':'R','pack_type':'Single','price_unit':price,'promotion_flag':promo,'stock_available':stock,'units_sold':int(rng.poisson(mu)) if stock else 0})
    d=pd.DataFrame(rows);w=d.groupby(['sku','week','channel','region']).units_sold.sum().reset_index()
    return w,d

def main():
    evidence=[]
    for truth in [-1.,0.7]:
        w,d=synthetic(truth);design=build(w,d)
        free=fit(design,'free');constrained=fit(design,'nonpositive')
        assert abs(free['epsilon']-truth)<.08
        assert constrained['epsilon']<=0
        if truth<0:assert abs(constrained['epsilon']-truth)<.08
        else:assert abs(constrained['epsilon'])<1e-8
        assert np.isfinite(free['covariance']).all() and np.diag(free['covariance']).min()>0
        # Derivadas contrastadas por diferencias finitas sobre mu.
        p=free['parameters'];mu,J,_=mean_jacobian(p,design);errors=[]
        for j in range(len(p)):
            step=np.zeros(len(p));step[j]=1e-5
            plus=mean_jacobian(p+step,design)[0];minus=mean_jacobian(p-step,design)[0]
            errors.append(float(np.max(abs((np.log(plus)-np.log(minus))/(2e-5)-J[:,j]))))
        assert max(errors)<1e-7
        # Una subida común de TODOS los precios diarios debe escalar el total
        # semanal exactamente por ratio**epsilon, aunque falten días y cambie promo.
        shifted=d.copy();shifted.price_unit*=1.1
        alt=build(w,shifted);changed=predict(constrained,alt);original=predict(constrained,design)
        assert np.allclose(changed,original*1.1**constrained['epsilon'],rtol=1e-11)
        assert (changed<=original+1e-8).all()
        evidence.append({'true_epsilon':truth,'recovered_free':free['epsilon'],'recovered_nonpositive':constrained['epsilon'],'maximum_derivative_error':max(errors),'weekly_cells':len(w),'result':'pass'})
    output=Path(__file__).resolve().parent/'verificacion-modelo-semanal.json'
    output.write_text(json.dumps({'synthetic_only_for_tests':True,'source_dataset_unchanged':True,'tests':evidence},indent=2),encoding='utf8')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
