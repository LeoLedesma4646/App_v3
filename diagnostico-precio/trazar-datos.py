"""Reconstruye agregación desde la fuente pública; estima al grano diario.

No altera fuente recibida, forecast ni registros de elasticidad existentes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,zipfile,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from modelos import DATA,fc

OUT=Path(__file__).resolve().parent
KEY=['sku','week','channel','region']

def main():
    archive=OUT/'fuentes-publicas/dataset-v16.zip'
    with zipfile.ZipFile(archive) as z:
        public=pd.read_csv(z.open('weekly_df_final_for_modeling.csv'),parse_dates=['week'])
        daily=pd.read_csv(z.open('FMCG_2022_2024.csv'),parse_dates=['date'])
        member_hashes={name:hashlib.sha256(z.read(name)).hexdigest() for name in ['weekly_df_final_for_modeling.csv','FMCG_2022_2024.csv']}
    local=fc.load_data(DATA)
    public=public.sort_values(KEY).reset_index(drop=True);local=local.sort_values(KEY).reset_index(drop=True)
    assert public[KEY].equals(local[KEY])
    comparisons=[]
    for c in public.columns:
        if c in KEY:continue
        if pd.api.types.is_bool_dtype(public[c]):
            same=public[c].astype(bool).eq(local[c].astype(bool));delta=0.
        elif pd.api.types.is_numeric_dtype(public[c]):
            a=pd.to_numeric(public[c],errors='coerce').astype(float);b=pd.to_numeric(local[c],errors='coerce').astype(float)
            same=pd.Series(np.isclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True));delta=float(np.nanmax(abs(a-b)))
        else:same=public[c].astype(str).eq(local[c].astype(str));delta=None
        comparisons.append({'column':c,'different_rows':int((~same).sum()),'maximum_numeric_difference':delta})
    assert not any(r['different_rows'] for r in comparisons)
    daily['week']=daily.date.dt.to_period('W').dt.start_time
    daily['in_stock']=daily.stock_available.gt(0).astype(int)
    daily['revenue']=daily.units_sold*daily.price_unit
    reconstructed=daily.groupby(KEY).agg(units_sold=('units_sold','sum'),price_unit=('price_unit','mean'),promotion_flag=('promotion_flag','max'),stock_available=('stock_available','mean'),delivery_days=('delivery_days','mean'),observed_days=('date','nunique'),in_stock_days=('in_stock','sum'),promotion_days=('promotion_flag','sum'),promotion_share=('promotion_flag','mean'),revenue=('revenue','sum')).reset_index()
    enriched=public.merge(reconstructed,on=KEY,how='left',suffixes=('','_daily'),validate='one_to_one')
    reconstructed_checks=[]
    for c in ['units_sold','price_unit','promotion_flag','stock_available','delivery_days']:
        delta=(enriched[c].astype(float)-enriched[c+'_daily'].astype(float)).abs()
        reconstructed_checks.append({'column':c,'different_rows':int((delta>1e-9).sum()),'maximum_difference':float(delta.max())})
    assert all(r['different_rows']==0 for r in reconstructed_checks)
    features=enriched[KEY+['observed_days','in_stock_days','promotion_days','promotion_share','revenue']].copy()
    features['nonpromotion_days']=features.observed_days-features.promotion_days
    features['stockout_days']=features.observed_days-features.in_stock_days
    available=daily[daily.in_stock.eq(1)].groupby(KEY).agg(price_available_mean=('price_unit','mean'),promotion_available_share=('promotion_flag','mean'),price_available_logmean=('price_unit',lambda p:np.log(p).mean()))
    features=features.merge(available,on=KEY,how='left',validate='one_to_one')
    features.to_csv(OUT/'exposicion-semanal-reconstruida.csv',index=False)
    estimates=[]
    for sku,g in daily.groupby('sku'):
        hist=fc.aggregate_sku(local,sku)
        x=g[g.week.between(hist.week.min(),hist.week.max())&g.in_stock.eq(1)].copy()
        tau=(x.date-x.date.min()).dt.days/365.25
        X=pd.DataFrame({'ln_price':np.log(x.price_unit),'promotion_flag':x.promotion_flag.astype(float),'trend':tau},index=x.index)
        for k in [1,2]:X[f'sin{k}']=np.sin(2*np.pi*k*tau);X[f'cos{k}']=np.cos(2*np.pi*k*tau)
        cats=pd.DataFrame({'cell':x.channel.astype(str)+'|'+x.region.astype(str),'pack':x.pack_type.astype(str),'weekday':x.date.dt.dayofweek.astype(str)},index=x.index)
        X=pd.concat([X,pd.get_dummies(cats,drop_first=True,dtype=float)],axis=1)
        X=sm.add_constant(X,has_constant='add')
        assert np.linalg.matrix_rank(X.to_numpy())==len(X.columns)
        result=sm.GLM(x.units_sold.astype(float),X,family=sm.families.Poisson()).fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':x.week.factorize()[0]})
        beta=float(result.params['ln_price']);se=float(result.bse['ln_price']);G=int(x.week.nunique());critical=t.ppf(.975,G-1)
        estimates.append({'sku':sku,'epsilon':beta,'se_cluster':se,'ci_low':beta-critical*se,'ci_high':beta+critical*se,'p_cluster':float(2*t.sf(abs(beta/se),G-1)),'n_daily':len(x),'weeks':G,'promotion_coefficient':float(result.params['promotion_flag']),'converged':bool(result.converged)})
    est=pd.DataFrame(estimates);est['p_holm']=multipletests(est.p_cluster,method='holm')[1];est.to_csv(OUT/'elasticidad-diaria-contraste.csv',index=False)
    receipt={'source_xlsx_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'public_zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_sha256':member_hashes,'weekly_public_matches_local_all_24_columns':True,'comparison':comparisons,'daily_reconstruction_matches_weekly':reconstructed_checks,'daily_rows':len(daily),'weekly_received_rows':len(local),'daily_zero_stock_rate':float(daily.stock_available.eq(0).mean()),'daily_zero_units_rate':float(daily.units_sold.eq(0).mean()),'daily_price_promo_correlation':float(daily.price_unit.corr(daily.promotion_flag)),'daily_price_units_correlation':float(daily.price_unit.corr(daily.units_sold)),'daily_promo_share':float(daily.promotion_flag.mean()),'weekly_any_promo_share':float(public.promotion_flag.mean()),'weekly_promo_share_when_flag1':float(enriched.loc[enriched.promotion_flag.eq(1),'promotion_share'].mean()),'weeks_with_hidden_stockout':int((features.stockout_days>0).sum()),'weeks_partial_under7_days':int((features.observed_days<7).sum()),'weekly_observed_days_distribution':features.observed_days.value_counts().sort_index().to_dict(),'ppml_daily_summary':{'positive':int((est.epsilon>0).sum()),'negative':int((est.epsilon<0).sum()),'significantly_negative_individual':int((est.ci_high<0).sum()),'significantly_positive_individual':int((est.ci_low>0).sum()),'holm_significant':int((est.p_holm<.05).sum()),'median_epsilon':float(est.epsilon.median()),'max_abs_epsilon':float(est.epsilon.abs().max())},'ju021_daily':est[est.sku.eq('JU-021')].to_dict('records'),'scope':'Comparación observacional; filtrar disponibilidad no demuestra ignorabilidad. La fuente diaria permite verificar agregación, no el generador original.'}
    (OUT/'trazabilidad.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['comparison','member_sha256']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
