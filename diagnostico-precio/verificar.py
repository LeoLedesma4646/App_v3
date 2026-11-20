"""Verifica reparación, modelos reales, cuantiles, inventarios y exportación."""
from pathlib import Path
import sys,json,hashlib,zipfile,io
from dataclasses import asdict,replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from motor import Scenario,ROOT,DATA,PRICE_POLICY,load_registry,load_data,load_elasticity,forecast_base,calculate,validate,bundle,save,restore,export_portfolio

OUT=Path(__file__).resolve().parent
def main():
    source_before=hashlib.sha256(DATA.read_bytes()).hexdigest()
    ols_before=hashlib.sha256((ROOT/'artifacts/elasticity.csv').read_bytes()).hexdigest()
    data=load_data();rows=[];week=[];iv=[];configs=[];count=0;example=None
    for sku in sorted(load_registry().sku):
        c=Scenario(sku=sku,samples=1000)
        base=forecast_base(c,data)
        r0=calculate(c,base)
        assert r0['info']['elasticity_source']=='Sin respuesta estimada'
        np.testing.assert_array_equal(r0['weekly'].base_demand,r0['weekly'].scenario_demand)
        for policy in ['proportional','absolute']:
            for delta in [-10.,10.]:
                config=replace(c,price_change_pct=delta,elasticity_override=-1.,elasticity_se_override=.2,error_policy=policy)
                r=calculate(config,base);factor=1/(1+delta/100)
                np.testing.assert_allclose(r['weekly'].scenario_demand,r['weekly'].base_demand*factor,rtol=1e-12)
                for bound in ['low','high']:
                    difference=r['weekly'][f'scenario_{bound}']-r['weekly'][f'base_{bound}']
                    assert (difference<=1e-9).all() if delta>0 else (difference>=-1e-9).all()
                assert r['info']['elasticity_source']=='Supuesto manual'
                assert not r['weekly'].scenario_demand.isna().any()
                assert (r['inventory'].target_stock>=r['inventory'].expected_protection_demand).all()
                assert (r['inventory'].suggested_order>=0).all()
                count+=1
        chosen=calculate(replace(c,price_change_pct=10.,elasticity_override=-1.,include_elasticity_uncertainty=False),base) if sku=='JU-021' else r0
        if sku=='JU-021':example=chosen
        provenance={k:chosen['info'][k] for k in ['elasticity_source','price_response_status','price_policy_version']}
        rows.append({'sku':sku,**chosen['summary'],**provenance,'target_stock':int(chosen['inventory'].iloc[1].target_stock)})
        week.append(chosen['weekly'].assign(sku=sku,**provenance));iv.append(chosen['inventory'].assign(sku=sku,**provenance));configs.append(chosen['config'])
    rejected=[]
    for kwargs in [{'price_change_pct':10.},{'elasticity_override':.023},{'elasticity_override':0.,'elasticity_se_override':.2}]:
        try:validate(Scenario('JU-021',**kwargs))
        except ValueError as exc:rejected.append(str(exc))
        else:raise AssertionError('Se permitió un caso metodológicamente inválido')
    archive=zipfile.ZipFile(io.BytesIO(bundle(example)))
    manifest=json.loads(archive.read('manifest.json'))
    assert manifest['info']['price_policy_version']==PRICE_POLICY
    saved=save(example,'Verificación: JU-021 +10%, ε=-1 supuesto ilustrativo')
    recovered=restore(saved.name)
    assert recovered['summary']==example['summary']
    pf={'summary':pd.DataFrame(rows),'weekly':pd.concat(week),'inventory':pd.concat(iv),'configurations':configs,'price_policy_version':PRICE_POLICY}
    exported=export_portfolio(pf)
    for table in ['summary','weekly','inventory']:
        out=pd.read_csv(exported/f'{table}.csv')
        assert out.sku.nunique()==30 and out.price_policy_version.eq(PRICE_POLICY).all()
    assert json.loads((ROOT/'exports/latest.json').read_text())['run_id']==exported.name
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not at.exception
    slider=next(x for x in at.slider if x.label=='Cambio de precio (%)')
    assert slider.disabled and slider.value==0.
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').check().run()
    next(x for x in at.number_input if x.label=='Elasticidad del escenario').set_value(-1.).run()
    next(x for x in at.slider if x.label=='Cambio de precio (%)').set_value(10.).run()
    assert not at.exception
    demand=next(x for x in at.metric if x.label=='Ventas previstas del periodo')
    assert demand.delta=='-9.09%'
    pages=[]
    for page in ['Escenario','Inventarios','Modelos y elasticidad','Escenarios guardados','Resumen de 30 SKU','Método y fuentes']:
        at.radio[0].set_value(page).run();assert not at.exception;pages.append(page)
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').uncheck().run()
    assert next(x for x in at.slider if x.label=='Cambio de precio (%)').value==0.
    assert not at.exception
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==source_before
    assert hashlib.sha256((ROOT/'artifacts/elasticity.csv').read_bytes()).hexdigest()==ols_before
    receipt={'passed':True,'policy':PRICE_POLICY,'sku_count':30,'monotonic_scenario_checks':count,'invalid_cases_rejected':rejected,'app_pages':pages,'ui_price_plus10_delta':'-9.09%','source_sha256':source_before,'ols_sha256':ols_before,'source_and_ols_unchanged':True,'saved_run':saved.name,'export_run':exported.name,'example_ju021':{'summary':example['summary'],'inventory':example['inventory'].to_dict('records')},'example_scope':'epsilon=-1 es una hipótesis ilustrativa, no un resultado estimado'}
    (OUT/'verificacion.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
