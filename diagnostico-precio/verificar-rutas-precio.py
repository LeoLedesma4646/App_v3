"""Reproduce la vía anterior del regresor de forecast y prueba su corrección."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from motor import Scenario,PRICE_BASE_POLICY,load_data,forecast_base,calculate,load_model
from modelos import fc,predict

data=load_data();sku='JU-021';base=forecast_base(Scenario(sku,samples=100),data)
frame,meta,info=base;anchor=info['forecast_price_anchor'];hist=fc.aggregate_sku(data,sku)
name=meta['winner'];model=load_model(sku,name);rows=[]
for ratio in [.9,1.,1.1]:
    # Solo consulta directa al forecast, para reproducir el enrutamiento antiguo.
    # No altera la app, el modelo ni el diseño de la respuesta de precio vigente.
    old=frame.copy();old['price_unit']=anchor*ratio
    old_units=float(predict(name,model,hist,old,meta['params'][name]).sum())
    config=Scenario(sku,samples=100,reference_price=anchor*ratio)
    current=calculate(config,forecast_base(config,data))
    hypoth=Scenario(sku,samples=100,reference_price=anchor*ratio,elasticity_override=-1.)
    sensitivity=calculate(hypoth,forecast_base(hypoth,data))
    assert np.isclose(current['summary']['base_units'],frame.base_demand.sum())
    assert np.isclose(sensitivity['summary']['base_units'],frame.base_demand.sum()/ratio)
    rows.append({'reference_ratio':ratio,'old_predictive_regressor_units':old_units,'corrected_automatic_units':current['summary']['base_units'],'explicit_hypothesis_minus1_units':sensitivity['summary']['base_units']})
assert rows[2]['old_predictive_regressor_units']>rows[1]['old_predictive_regressor_units']
receipt={'passed':True,'sku':sku,'forecast_model':name,'price_anchor':anchor,'base_price_policy':PRICE_BASE_POLICY,'source_sha256':info['data_version'],'old_plus10_base_increase_pct':100*(rows[2]['old_predictive_regressor_units']/rows[1]['old_predictive_regressor_units']-1),'results':rows,'scope':'Reproduce el cálculo predictivo anterior, no un efecto causal. Hipótesis -1 es exclusivamente prueba de integración.'}
(Path(__file__).resolve().parent/'verificacion-rutas-precio.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
