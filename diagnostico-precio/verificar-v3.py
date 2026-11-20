"""Comprueba modelo semanal, integración, UI y trazabilidad de la política v3."""
from pathlib import Path
import sys,json,hashlib,zipfile,io
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from motor import Scenario,ROOT,DATA,PRICE_POLICY,PRICE_BASE_POLICY,load_registry,load_data,load_price_response,forecast_base,calculate,validate,bundle,save,restore,export_portfolio

def main():
    out=Path(__file__).resolve().parent
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'artifacts/elasticity.csv',*sorted((ROOT/'models').glob('*.pkl'))]}
    source_hash=hashlib.sha256(DATA.read_bytes()).hexdigest()
    pr=load_price_response();assert len(pr)==30 and pr.sku.nunique()==30
    assert pr.elasticity_used.le(0).all() and pr.converged.all()
    assert not pr.price_supported.any()  # Resultado de ESTA versión de fuente, no regla universal.
    validation=pd.read_csv(out/'validacion-respuesta-semanal.csv',parse_dates=['validation_end'])
    holdout=pd.read_csv(out/'holdout-respuesta-semanal.csv',parse_dates=['week'])
    for sku in pr.sku:
        assert validation[validation.sku.eq(sku)].validation_end.max()<holdout[holdout.sku.eq(sku)].week.min()
    data=load_data();count=0;reference_count=0;rows=[];weekly=[];iv=[];configs=[];example=None
    for sku in sorted(load_registry().sku):
        c=Scenario(sku=sku,samples=1000);base=forecast_base(c,data)
        for policy in ['proportional','absolute']:
            for delta in [-10.,10.]:
                auto=calculate(replace(c,price_change_pct=delta,error_policy=policy),base)
                np.testing.assert_array_equal(auto['weekly'].base_demand,auto['weekly'].scenario_demand)
                np.testing.assert_array_equal(auto['inventory'].iloc[0,1:].to_numpy(),auto['inventory'].iloc[1,1:].to_numpy())
                assert auto['summary']['elasticity']==0 and auto['info']['price_supported'] is False
                assert auto['info']['price_policy_version']==PRICE_POLICY
                assert any('no demuestra' in w for w in auto['warnings'])
                manual=calculate(replace(c,price_change_pct=delta,error_policy=policy,elasticity_override=-1.,elasticity_se_override=.2),base)
                np.testing.assert_allclose(manual['weekly'].scenario_demand,manual['weekly'].base_demand/(1+delta/100))
                for bound in ['low','high']:
                    diff=manual['weekly'][f'scenario_{bound}']-manual['weekly'][f'base_{bound}']
                    assert (diff<=1e-8).all() if delta>0 else (diff>=-1e-8).all()
                count+=2
        anchor=base[2]['forecast_price_anchor']
        for policy in ['proportional','absolute']:
            for ref_ratio in [.9,1.1]:
                for epsilon in [None,-1.]:
                    cfg=replace(c,reference_price=anchor*ref_ratio,price_change_pct=10.,error_policy=policy,elasticity_override=epsilon,elasticity_se_override=.2 if epsilon is not None else 0.)
                    anchored=forecast_base(cfg,data)
                    np.testing.assert_array_equal(anchored[0].anchor_demand,base[0].anchor_demand)
                    result=calculate(cfg,anchored)
                    total_ratio=ref_ratio*1.1
                    direct=calculate(replace(cfg,reference_price=None,price_change_pct=(total_ratio-1)*100),base)
                    np.testing.assert_allclose(result['weekly'].scenario_demand,direct['weekly'].scenario_demand,rtol=1e-12)
                    for bound in ['low','high']:np.testing.assert_allclose(result['weekly'][f'scenario_{bound}'],direct['weekly'][f'scenario_{bound}'],rtol=1e-11,atol=1e-8)
                    np.testing.assert_allclose(result['inventory'].iloc[1,1:].astype(float),direct['inventory'].iloc[1,1:].astype(float),rtol=1e-11,atol=1e-8)
                    reference_count+=1
        chosen=calculate(replace(c,price_change_pct=10.),base)
        if sku=='JU-021':example=chosen
        provenance={k:chosen['info'][k] for k in ['elasticity_source','price_response_status','price_policy_version','base_price_policy','price_supported']}
        rows.append({'sku':sku,**chosen['summary'],**provenance});weekly.append(chosen['weekly'].assign(sku=sku,**provenance));iv.append(chosen['inventory'].assign(sku=sku,**provenance));configs.append(chosen['config'])
    for bad in [{'elasticity_override':.02},{'elasticity_override':0.,'elasticity_se_override':.2}]:
        try:validate(Scenario('JU-021',**bad))
        except ValueError:pass
        else:raise AssertionError('Caso inválido aceptado')
    saved=save(example,'Modelo semanal validado: JU-021 +10%, respuesta no respaldada')
    restored=restore(saved.name);assert restored['summary']==example['summary']
    assert json.loads(zipfile.ZipFile(io.BytesIO(bundle(example))).read('manifest.json'))['info']['corrected_elasticity_free']<0
    exported=export_portfolio({'summary':pd.DataFrame(rows),'weekly':pd.concat(weekly),'inventory':pd.concat(iv),'configurations':configs,'price_policy_version':PRICE_POLICY,'base_price_policy':PRICE_BASE_POLICY})
    assert pd.read_csv(exported/'summary.csv').price_policy_version.eq(PRICE_POLICY).all()
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run();assert not at.exception
    slider=next(x for x in at.slider if x.label=='Cambio de precio (%)');assert not slider.disabled
    slider.set_value(10.).run();assert not at.exception
    assert next(x for x in at.metric if x.label=='Ventas previstas del periodo').delta=='+0.00%'
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').check().run()
    next(x for x in at.number_input if x.label=='Elasticidad del escenario').set_value(-1.).run()
    assert next(x for x in at.metric if x.label=='Ventas previstas del periodo').delta=='-9.09%'
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').uncheck().run()
    assert next(x for x in at.slider if x.label=='Cambio de precio (%)').value==10.
    assert next(x for x in at.metric if x.label=='Ventas previstas del periodo').delta=='+0.00%'
    pages=[]
    for page in ['Escenario','Inventarios','Modelos y elasticidad','Escenarios guardados','Resumen de 30 SKU','Método y fuentes']:
        at.radio[0].set_value(page).run();assert not at.exception;pages.append(page)
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==source_hash
    for name,h in before.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    evidence={'passed':True,'policy':PRICE_POLICY,'base_price_policy':PRICE_BASE_POLICY,'sku_count':30,'scenario_checks':count,'reference_price_checks':reference_count,'two_price_paths_apply_once':True,'temporal_validation_precedes_holdout':True,'source_sha256':source_hash,'source_original_ols_and_forecast_models_unchanged':True,'pages':pages,'ui_automatic_plus10_demand_change':'0.00%, respuesta no respaldada; no prueba inelasticidad','manual_example':'epsilon=-1, precio +10%: -9.09%; supuesto, no estimación','saved_run':saved.name,'export_run':exported.name,'ju021':{'info':example['info'],'summary':example['summary']}}
    (out/'verificacion-v3.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
