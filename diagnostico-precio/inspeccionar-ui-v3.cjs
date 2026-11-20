const {chromium}=require('C:/Users/ledes/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 try{
  const page=await browser.newPage({viewport:{width:1360,height:1000}});
  await page.goto('http://localhost:8510/',{waitUntil:'domcontentloaded'});
  await page.getByText('De precio a ventas previstas',{exact:true}).waitFor({timeout:90000});
  console.log((await page.locator('body').innerText()).slice(0,9000));
  console.log('SLIDERS',await page.getByRole('slider').evaluateAll(el=>el.map(x=>({label:x.getAttribute('aria-label'),min:x.getAttribute('aria-valuemin'),max:x.getAttribute('aria-valuemax'),value:x.getAttribute('aria-valuenow')}))));
 } finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
