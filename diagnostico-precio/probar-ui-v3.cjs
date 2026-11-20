const {chromium}=require('C:/Users/ledes/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {expect}=require('C:/Users/ledes/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/test');
const fs=require('node:fs/promises');
const path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://localhost:8510/',{waitUntil:'domcontentloaded'});
  await page.getByRole('button',{name:'Guardar escenario',exact:true}).waitFor({timeout:90000});
  const price=page.getByRole('slider',{name:'Cambio de precio (%)',exact:true});
  await expect(price).toHaveAttribute('aria-valuenow','0');
  await expect(page.getByRole('checkbox',{name:'Usar elasticidad manual',exact:true})).not.toBeChecked();
  await price.focus();for(let i=0;i<20;i++)await price.press('ArrowRight');
  await expect(price).toHaveAttribute('aria-valuenow','10');
  const metric=page.locator('[data-testid="stMetric"]').filter({hasText:'Precio propuesto'});
  await expect(metric).toContainText('+10.0%',{timeout:90000});
  const demand=page.locator('[data-testid="stMetric"]').filter({hasText:'Ventas previstas del periodo'});
  await expect(demand).toContainText('+0.00%');
  await expect(page.locator('body')).toContainText('esto no demuestra demanda real inelástica');
  // Esperar la descarga real del servidor comprueba el manifiesto publicado,
  // además del motor probado con AppTest.
  const downloaded=page.waitForEvent('download');
  await page.getByText('Descargar resultado completo',{exact:true}).click();
  const download=await downloaded;await download.saveAs(path.join(__dirname,'descarga-navegador-v3.zip'));
  await page.screenshot({path:path.join(__dirname,'ejemplo-modelo-semanal-v3.png'),fullPage:true});
  const scenarioText=await page.locator('body').innerText();
  await page.getByText('Definir precio de referencia',{exact:true}).click();
  const reference=page.getByRole('spinbutton',{name:'Precio de referencia por unidad',exact:true});
  await reference.fill('5.815335515873017');await reference.press('Enter');
  await expect(demand).toContainText('+0.00%',{timeout:90000});
  await expect(page.locator('body')).toContainText('La referencia manual se compara con el ancla histórica');
  await expect(demand).toContainText('8,276 u.');
  await page.getByText('Modelos y elasticidad',{exact:true}).click();
  await page.getByText('Regresión semanal corregida del mismo SKU',{exact:true}).waitFor({timeout:90000});
  await expect(page.locator('body')).toContainText('-0.0091');
  await expect(page.locator('body')).toContainText('[-0.0757, 0.0575]');
  await page.screenshot({path:path.join(__dirname,'modelo-semanal-v3.png'),fullPage:true});
  await page.getByText('Inventarios',{exact:true}).click();
  await page.getByText('Reposición para la misma demanda',{exact:true}).waitFor({timeout:90000});
  await expect(page.getByRole('slider',{name:'Cambio de precio (%)',exact:true})).toHaveAttribute('aria-valuenow','10');
  if(errors.length)throw new Error(JSON.stringify(errors));
  const receipt={passed:true,browser:'Edge headless, teclado/ratón virtual de Playwright; sin captura nativa ni Pointer Lock',url:'http://localhost:8510/',sku:'JU-021',manual:false,price_change_pct:10,manual_reference_plus10_checked:true,demand_change_pct:0,meaning:'Respuesta no respaldada; conserva demanda base, no prueba inelasticidad real',pages:['Escenario','Modelos y elasticidad','Inventarios'],page_errors:errors,downloaded_result:'descarga-navegador-v3.zip',scenario_metric_text:scenarioText.match(/Precio propuesto[\s\S]*?Ingreso potencial del periodo[\s\S]*?\+10.00%/)?.[0]??null};
  await fs.writeFile(path.join(__dirname,'verificacion-navegador-v3.json'),JSON.stringify(receipt,null,2));
  console.log(JSON.stringify(receipt,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
