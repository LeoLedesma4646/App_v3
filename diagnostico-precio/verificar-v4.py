"""Prueba la respuesta automática, su significado y el tránsito a inventarios."""
from pathlib import Path
import sys,json,hashlib,zipfile,io,math
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.integrate import quad
from streamlit.testing.v1 import AppTest
from respuesta_precio import distribution
from motor import Scenario,ROOT,DATA,PRICE_POLICY,PRICE_BASE_POLICY,load_registry,load_data,load_price_response,forecast_base,calculate,validate,bundle,save,restore,export_portfolio

def verify_ui():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run();assert not at.exception
    slider=lambda:next(x for x in at.slider if x.label=='Cambio de precio (%)')
    demand=lambda:next(x for x in at.metric if x.label=='Ventas previstas del periodo')
    slider().set_value(10.).run();assert demand().delta=='-0.29%'
    slider().set_value(-10.).run();assert demand().delta=='+0.32%'
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').check().run()
    next(x for x in at.number_input if x.label=='Elasticidad del escenario').set_value(-1.).run()
    assert demand().delta=='+11.11%'
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').uncheck().run()
    assert demand().delta=='+0.32%'
    slider().set_value(10.).run()
    next(x for x in at.checkbox if x.label=='Definir precio de referencia').check().run()
    next(x for x in at.number_input if x.label=='Precio de referencia por unidad').set_value(5.815335515873017).run()
    assert demand().value=='8,228.0 u.' and demand().delta=='-0.29%'
    next(x for x in at.checkbox if x.label=='Definir precio de referencia').uncheck().run()
    assert demand().value=='8,251.9 u.'
    pages=[]
    for page in ['Escenario','Inventarios','Modelos y elasticidad','Escenarios guardados','Resumen de 30 SKU','Método y fuentes']:
        at.radio[0].set_value(page).run();assert not at.exception;pages.append(page)
    receipt={'passed':True,'policy':PRICE_POLICY,'pages':pages,'slider_plus10':'-0.29%','slider_minus10':'+0.32%','manual_reference_plus10_and_slider_plus10':'8,228.0 u.','reference_cache_uses_same_historical_anchor':True,'manual_automatic_transition_preserves_price':True}
    (Path(__file__).resolve().parent/'verificacion-ui-v4.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    return pages

def main():
    out=Path(__file__).resolve().parent
    preserved=[ROOT/'artifacts/elasticity.csv',ROOT/'artifacts/price_response.csv',ROOT/'artifacts/price_response_manifest.json',*sorted((ROOT/'models').glob('*.pkl'))]
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in preserved}
    source_hash=hashlib.sha256(DATA.read_bytes()).hexdigest()
    pr=load_price_response();assert len(pr)==30 and pr.sku.nunique()==30
    assert pr.elasticity_used.lt(0).all() and pr.converged.all()
    # Independencia: integrar la densidad proporcional del modelo resumido
    # sin emplear truncnorm ni su fórmula de momentos.
    integration=[]
    for b,s in [(-.009087088,.033646567),(.061965,.028763),(-.8,.02),(0.,.1)]:
        upper=-b/s
        density=lambda z:math.exp(-z*z/2)/math.sqrt(2*math.pi)
        lower=min(-12.,upper-12.);upper=min(12.,upper)
        mass=quad(density,lower,upper,epsabs=1e-12)[0]
        first=quad(lambda z:(b+s*z)*density(z),lower,upper,epsabs=1e-12)[0]/mass
        d=distribution(b,s);np.testing.assert_allclose(d.mean(),first,atol=1e-10)
        draws=d.rvs(size=10000,random_state=123);assert (draws<0).all()
        integration.append({'estimate':b,'se':s,'integrated_mean':first})
    assert abs(distribution(.1,.005).mean())<.001 # Evidencia positiva precisa aproxima la frontera, no una elasticidad fija.
    validation=pd.read_csv(out/'validacion-respuesta-semanal.csv',parse_dates=['validation_end'])
    holdout=pd.read_csv(out/'holdout-respuesta-semanal.csv',parse_dates=['week'])
    for sku in pr.sku:
        assert validation[validation.sku.eq(sku)].validation_end.max()<holdout[holdout.sku.eq(sku)].week.min()
    data=load_data();count=0;reference_count=0;rows=[];weekly=[];iv=[];configs=[];example=None
    for sku in sorted(load_registry().sku):
        c=Scenario(sku=sku,samples=1000);base=forecast_base(c,data)
        neutral=calculate(c,base)
        np.testing.assert_array_equal(neutral['weekly'].base_demand,neutral['weekly'].scenario_demand)
        np.testing.assert_array_equal(neutral['inventory'].iloc[0,1:].to_numpy(),neutral['inventory'].iloc[1,1:].to_numpy())
        for policy in ['proportional','absolute']:
            for delta in [-40.,-10.,10.,40.]:
                auto=calculate(replace(c,price_change_pct=delta,error_policy=policy),base)
                eps=float(pr.set_index('sku').loc[sku,'elasticity_used']);factor=math.exp(eps*math.log1p(delta/100))
                np.testing.assert_allclose(auto['weekly'].scenario_demand,auto['weekly'].base_demand*factor)
                diff=auto['weekly'].scenario_demand-auto['weekly'].base_demand
                assert (diff<0).all() if delta>0 else (diff>0).all()
                for bound in ['low','high']:
                    diff=auto['weekly'][f'scenario_{bound}']-auto['weekly'][f'base_{bound}']
                    assert (diff<=1e-8).all() if delta>0 else (diff>=-1e-8).all()
                q=auto['inventory'].expected_protection_demand
                assert q.iloc[1]<q.iloc[0] if delta>0 else q.iloc[1]>q.iloc[0]
                assert auto['info']['elasticity_uncertainty_included'] and auto['summary']['elasticity']<0
                assert auto['info']['price_policy_version']==PRICE_POLICY
                assert any('supuesto' in w for w in auto['warnings'])
                manual=calculate(replace(c,price_change_pct=delta,error_policy=policy,elasticity_override=-1.,elasticity_se_override=.2),base)
                np.testing.assert_allclose(manual['weekly'].scenario_demand,manual['weekly'].base_demand/(1+delta/100))
                count+=2
        anchor=base[2]['forecast_price_anchor']
        for policy in ['proportional','absolute']:
            for ref_ratio in [.9,1.1]:
                for epsilon in [None,-1.]:
                    cfg=replace(c,reference_price=anchor*ref_ratio,price_change_pct=10.,error_policy=policy,elasticity_override=epsilon,elasticity_se_override=.2 if epsilon is not None else 0.)
                    anchored=forecast_base(cfg,data)
                    np.testing.assert_array_equal(anchored[0].anchor_demand,base[0].anchor_demand)
                    result=calculate(cfg,anchored)
                    direct=calculate(replace(cfg,reference_price=None,price_change_pct=(ref_ratio*1.1-1)*100),base)
                    np.testing.assert_allclose(result['weekly'].scenario_demand,direct['weekly'].scenario_demand,rtol=1e-12)
                    for bound in ['low','high']:np.testing.assert_allclose(result['weekly'][f'scenario_{bound}'],direct['weekly'][f'scenario_{bound}'],rtol=1e-11,atol=1e-8)
                    np.testing.assert_allclose(result['inventory'].iloc[1,1:].astype(float),direct['inventory'].iloc[1,1:].astype(float),rtol=1e-11,atol=1e-8)
                    reference_count+=1
        chosen=calculate(replace(c,price_change_pct=10.),base)
        if sku=='JU-021':example=chosen
        provenance={k:chosen['info'][k] for k in ['elasticity_source','price_response_status','price_policy_version','base_price_policy','price_supported']}
        rows.append({'sku':sku,**chosen['summary'],**provenance});weekly.append(chosen['weekly'].assign(sku=sku,**provenance));iv.append(chosen['inventory'].assign(sku=sku,**provenance));configs.append(chosen['config'])
    fixed=calculate(Scenario('JU-021',price_change_pct=10.,include_elasticity_uncertainty=False),forecast_base(Scenario('JU-021'),data))
    assert not fixed['info']['elasticity_uncertainty_included']
    np.testing.assert_allclose(fixed['weekly'].scenario_high,fixed['weekly'].base_high*1.1**fixed['summary']['elasticity'])
    for bad in [{'elasticity_override':.02},{'elasticity_override':0.,'elasticity_se_override':.2}]:
        try:validate(Scenario('JU-021',**bad))
        except ValueError:pass
        else:raise AssertionError('Caso inválido aceptado')
    saved=save(example,'JU-021 +10%: sensibilidad condicionada v4')
    assert restore(saved.name)['summary']==example['summary']
    assert json.loads(zipfile.ZipFile(io.BytesIO(bundle(example))).read('manifest.json'))['summary']['demand_change_pct']<0
    exported=export_portfolio({'summary':pd.DataFrame(rows),'weekly':pd.concat(weekly),'inventory':pd.concat(iv),'configurations':configs,'price_policy_version':PRICE_POLICY,'base_price_policy':PRICE_BASE_POLICY})
    assert pd.read_csv(exported/'summary.csv').demand_change_pct.lt(0).all()
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run();assert not at.exception
    next(x for x in at.slider if x.label=='Cambio de precio (%)').set_value(10.).run();assert not at.exception
    assert next(x for x in at.metric if x.label=='Ventas previstas del periodo').delta=='-0.29%'
    next(x for x in at.slider if x.label=='Cambio de precio (%)').set_value(-10.).run();assert not at.exception
    assert next(x for x in at.metric if x.label=='Ventas previstas del periodo').delta=='+0.32%'
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').check().run()
    next(x for x in at.number_input if x.label=='Elasticidad del escenario').set_value(-1.).run()
    assert next(x for x in at.metric if x.label=='Ventas previstas del periodo').delta=='+11.11%'
    next(x for x in at.checkbox if x.label=='Usar elasticidad manual').uncheck().run()
    assert next(x for x in at.metric if x.label=='Ventas previstas del periodo').delta=='+0.32%'
    pages=[]
    for page in ['Escenario','Inventarios','Modelos y elasticidad','Escenarios guardados','Resumen de 30 SKU','Método y fuentes']:
        at.radio[0].set_value(page).run();assert not at.exception;pages.append(page)
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==source_hash
    for name,h in before.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    evidence={'passed':True,'policy':PRICE_POLICY,'base_price_policy':PRICE_BASE_POLICY,'sku_count':30,'scenario_checks':count,'reference_price_checks':reference_count,'automatic_response_nonzero_and_monotonic_all30':True,'neutral_price_identical_inventory_all30':True,'two_price_paths_apply_once':True,'temporal_evidence_preserved':True,'independent_normal_density_integration':integration,'source_sha256':source_hash,'source_original_ols_v3_estimates_and_forecast_models_unchanged':True,'pages':pages,'saved_run':saved.name,'export_run':exported.name,'ju021':{'summary':example['summary'],'inventory':example['inventory'].to_dict('records'),'info':example['info']}}
    (out/'verificacion-v4.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if '--ui-only' in sys.argv:print(verify_ui())
    else:main()
