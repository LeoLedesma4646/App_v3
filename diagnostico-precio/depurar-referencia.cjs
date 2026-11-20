const {chromium}=require('C:/Users/ledes/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {expect}=require('C:/Users/ledes/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/test');
(async()=>{
 const b=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const p=await b.newPage({viewport:{width:1440,height:1100}});await p.goto('http://localhost:8510/');
  await p.getByRole('button',{name:'Guardar escenario',exact:true}).waitFor({timeout:90000});
  await p.getByText('Definir precio de referencia',{exact:true}).click();
  await expect(p.locator('body')).toContainText('La referencia manual se compara con el ancla histórica',{timeout:90000});
  const r=p.getByRole('spinbutton',{name:'Precio de referencia por unidad',exact:true});
  console.log('attributes',await r.evaluate(e=>({type:e.type,value:e.value,disabled:e.disabled,readonly:e.readOnly,step:e.step,min:e.min})));
  await r.focus();await expect(r).toHaveValue(/^5\.2866/);
  await r.fill('6.00');console.log('filled',await r.inputValue());
  await r.press('Enter');console.log('enter',await r.inputValue());
  try{await expect(p.locator('[data-testid="stMetric"]').filter({hasText:'Precio propuesto'})).toContainText('6.00 u.m.',{timeout:15000});}catch(e){console.log('metric did not become 6.00');}
  console.log('metrics',await p.locator('[data-testid="stMetric"]').allTextContents());
  console.log('reference_after',await r.inputValue());
 }finally{await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
