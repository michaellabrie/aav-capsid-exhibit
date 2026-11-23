/* AAV Capsid Lab — actual browser inference using exported scikit-learn MLP weights.
   Data, labels and this model are synthetic teaching materials, never wet-lab results. */
const $ = (id) => document.getElementById(id);
const pct = (x, digits=1) => `${(100*x).toFixed(digits)}%`;
const clamp = (x,a,b) => Math.min(b,Math.max(a,x));
const init = {serotype:'AAV2',aap:1,contact:.70,stability:.75,ph:7.2,salt:145,concentration:.70};
const state = { model:null, cases:[], guide:'', threshold:.315, controls:{...init}, selectedCase:null, caseMenu:[], engine:null, chatBusy:false };

function features(input){
  return [input.aap,input.contact,input.stability,input.ph,input.salt,input.concentration,
    ...['AAV2','AAV5','AAV8','AAV9'].map((s)=>Number(input.serotype===s))];
}
function predict(input){
  if(!state.model)return 0;
  let a=features(input).map((x,i)=>(x-state.model.mean[i])/state.model.scale[i]);
  for(const layer of state.model.layers){
    const next=layer.bias.map((bias,j)=>{
      let val=bias;
      for(let i=0;i<a.length;i++)val+=a[i]*layer.weights[i][j];
      return layer.activation==='relu'?Math.max(0,val):1/(1+Math.exp(-val));
    });
    a=next;
  }
  return a[0];
}
function getInputs(){return {...state.controls};}
function statsAt(threshold){
  let tp=0,tn=0,fp=0,fn=0;
  for(const c of state.cases){let pred=Number(c.prob>=threshold);
    if(c.actual===1){if(pred)tp++;else fn++;}else{if(pred)fp++;else tn++;}
  }
  const div=(a,b)=>b?a/b:0;
  const accuracy=div(tp+tn,state.cases.length), precision=div(tp,tp+fp),recall=div(tp,tp+fn);
  return {tp,tn,fp,fn,accuracy,precision,recall,f1:div(2*precision*recall,precision+recall)};
}
function updateInputsUI(){
  const c=state.controls;
  $('serotype').value=c.serotype;
  ['contact','stability','ph','salt','concentration'].forEach(k=>$(k).value=c[k]);
  $('aap-absent').classList.toggle('active',c.aap===0);
  $('aap-present').classList.toggle('active',c.aap===1);
  $('aap-absent').setAttribute('aria-pressed',String(c.aap===0));
  $('aap-present').setAttribute('aria-pressed',String(c.aap===1));
  $('contact-value').textContent=c.contact.toFixed(2);$('stability-value').textContent=c.stability.toFixed(2);
  $('ph-value').textContent=c.ph.toFixed(1);$('salt-value').textContent=c.salt.toFixed(0);
  $('concentration-value').textContent=c.concentration.toFixed(2);
}
function renderWorkbench(){
  if(!state.model)return;
  const c=getInputs(),p=predict(c), absent=predict({...c,aap:0}), present=predict({...c,aap:1});
  $('pred-prob').textContent=(p*100).toFixed(1);$('pred-label').textContent=p>=state.threshold?'PREDICTED: ASSEMBLED':'PREDICTED: NOT ASSEMBLED';
  $('pred-label').style.color=p>=state.threshold?'var(--teal)':'var(--orange)';
  $('probability-bar').style.width=pct(p);$('probability-threshold').style.left=pct(state.threshold);
  $('compare-absent').textContent=pct(absent);$('compare-present').textContent=pct(present);
  $('compare-absent-bar').style.width=pct(absent);$('compare-present-bar').style.width=pct(present);
  const delta=(present-absent)*100;
  $('compare-note').textContent=`With this synthetic input, changing only AAP from absent to present ${delta>=0?'raises':'lowers'} the network prediction by ${Math.abs(delta).toFixed(1)} percentage points. This is a model contrast, not a causal experiment.`;
}
function drawChart(){
  const svg=$('trade-chart'); if(!state.cases.length)return;
  const w=580,h=230,left=40,right=15,top=17,bottom=36;
  const xp=(v)=>left+(w-left-right)*(v-.05)/.90;
  const yp=(v)=>top+(h-top-bottom)*(1-v);
  const xvals=Array.from({length:91},(_,i)=>.05+i*.01);
  const getPath=(name)=>xvals.map((x,i)=>`${i?'L':'M'}${xp(x).toFixed(1)},${yp(statsAt(x)[name]).toFixed(1)}`).join(' ');
  const ns='http://www.w3.org/2000/svg';svg.replaceChildren();
  function node(tag,attrs,parent=svg){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));parent.appendChild(e);return e;}
  [0,.25,.5,.75,1].forEach((v)=>{
    node('line',{x1:left,x2:w-right,y1:yp(v),y2:yp(v),stroke:'#315061','stroke-dasharray':'2,6'});
    const t=node('text',{x:7,y:yp(v)+4,fill:'#85a2ae','font-size':'10'});t.textContent=`${Math.round(v*100)}%`;
  });
  [.1,.3,.5,.7,.9].forEach((v)=>{const t=node('text',{x:xp(v)-9,y:h-12,fill:'#85a2ae','font-size':'10'});t.textContent=v.toFixed(1);});
  node('path',{d:getPath('recall'),fill:'none',stroke:'#70f0df','stroke-width':3,'stroke-linecap':'round'});
  node('path',{d:getPath('precision'),fill:'none',stroke:'#a9a3f7','stroke-width':3,'stroke-linecap':'round'});
  node('line',{x1:xp(state.threshold),x2:xp(state.threshold),y1:top,y2:h-bottom,stroke:'#ffc58a','stroke-width':1.5,'stroke-dasharray':'5,4'});
  const m=statsAt(state.threshold);
  node('circle',{cx:xp(state.threshold),cy:yp(m.recall),r:5,fill:'#70f0df',stroke:'#0a101a','stroke-width':2});
  node('circle',{cx:xp(state.threshold),cy:yp(m.precision),r:5,fill:'#a9a3f7',stroke:'#0a101a','stroke-width':2});
}
function renderThreshold(){
  if(!state.model)return;
  const v=state.threshold,s=statsAt(v);
  $('threshold-slider').value=v;$('threshold-big').textContent=v.toFixed(3);
  for(const k of ['accuracy','precision','recall','f1'])$('m-'+k).textContent=pct(s[k]);
  for(const k of ['tn','fp','fn','tp'])$('m-'+k).textContent=s[k];
  drawChart();renderWorkbench();
}
function caseLabel(c){return c.prob>=state.model.metrics.threshold?'Assembled':'Not assembled';}
function actualLabel(c){return c.actual?'Assembled':'Not assembled';}
function pickCases(){
  const t=state.model.metrics.threshold;
  const fp=state.cases.filter(c=>c.actual===0&&c.prob>=t).sort((a,b)=>b.prob-a.prob)[0];
  const fn=state.cases.filter(c=>c.actual===1&&c.prob<t).sort((a,b)=>a.prob-b.prob)[0];
  const boundary=[...state.cases].sort((a,b)=>Math.abs(a.prob-t)-Math.abs(b.prob-t))[0];
  const correct=state.cases.filter(c=>c.actual===1&&c.prob>=t).sort((a,b)=>b.prob-a.prob)[0];
  state.caseMenu=[
    {case:fp,label:'FALSE POSITIVE',title:'Confident false alarm',note:'The network predicted assembly, but the simulated label is negative.'},
    {case:fn,label:'FALSE NEGATIVE',title:'Missed positive',note:'The network predicted no assembly, but the simulated label is positive.'},
    {case:boundary,label:'BORDERLINE CASE',title:'Close to the boundary',note:'This prediction lies near the validation-selected threshold. A slight threshold change may flip the class.'},
    {case:correct,label:'TRUE POSITIVE',title:'Correct positive',note:'The network correctly identifies this positive simulated test case.'}
  ].filter(x=>x.case);
  $('case-buttons').replaceChildren();
  state.caseMenu.forEach((item,i)=>{
    const btn=document.createElement('button');btn.className='case-choice';btn.type='button';btn.innerHTML=`<div><small>${String(i+1).padStart(2,'0')} / ${item.label}</small><strong>${item.title}</strong></div><span aria-hidden="true">↗</span>`;
    btn.addEventListener('click',()=>showCase(i));$('case-buttons').appendChild(btn);
  });
  showCase(0);
}
function showCase(i){
  const {case:c,label,note}=state.caseMenu[i];state.selectedCase=c;
  [...$('case-buttons').children].forEach((btn,j)=>btn.classList.toggle('active',i===j));
  $('case-id').textContent=`${c.id} / HELD-OUT TEST`;$('case-kind').textContent=label;
  $('case-pred').textContent=caseLabel(c);$('case-actual').textContent=actualLabel(c);
  $('case-prob').textContent=pct(c.prob);
  const vals=[['SEROTYPE',c.serotype],['AAP',c.aap?'PRESENT':'ABSENT'],['INTERFACE',c.contact.toFixed(2)],['VP STABILITY',c.stability.toFixed(2)],['pH',c.ph.toFixed(1)],['IONIC STRENGTH',`${Math.round(c.salt)} mM`],['VP INDEX',c.concentration.toFixed(2)]];
  $('case-facts').replaceChildren();
  for(const [title,value] of vals){let d=document.createElement('div'),s=document.createElement('span'),b=document.createElement('strong');s.textContent=title;b.textContent=value;d.append(s,b);$('case-facts').appendChild(d);}
  const nextStep=(c.actual!==Number(c.prob>=state.model.metrics.threshold))?' What might help: repeated synthetic observations at similar inputs could estimate the underlying probability more reliably; additional measured features might help only if they contain genuinely new information.':' This shows a correct prediction, but a single success does not validate a model.';
  $('case-explainer').textContent=note+' Synthetic labels contain randomized variation, so this case does not establish a molecular mechanism.'+nextStep;
}
function syncAll(){updateInputsUI();renderWorkbench();}
function bindControls(){
  $('serotype').addEventListener('change',e=>{state.controls.serotype=e.target.value;syncAll();});
  $('aap-present').addEventListener('click',()=>{state.controls.aap=1;syncAll();});
  $('aap-absent').addEventListener('click',()=>{state.controls.aap=0;syncAll();});
  for(const id of ['contact','stability','ph','salt','concentration'])$(id).addEventListener('input',e=>{state.controls[id]=Number(e.target.value);syncAll();});
  $('reset').addEventListener('click',()=>{state.controls={...init};syncAll();});
  $('threshold-slider').addEventListener('input',e=>{state.threshold=Number(e.target.value);renderThreshold();});
  $('restore-threshold').addEventListener('click',()=>{state.threshold=state.model.metrics.threshold;renderThreshold();});
  $('load-case').addEventListener('click',()=>{const c=state.selectedCase;if(!c)return;state.controls={serotype:c.serotype,aap:c.aap,contact:c.contact,stability:c.stability,ph:c.ph,salt:c.salt,concentration:c.concentration};syncAll();$('workbench').scrollIntoView({behavior:'smooth'});});
  document.querySelectorAll('[data-question]').forEach(btn=>btn.addEventListener('click',()=>{
    const question=btn.dataset.question;
    if(state.engine){$('chat-input').value=question;$('chat-input').focus();}
    else{$('assistant').scrollIntoView({behavior:'smooth'});$('chat-progress').textContent='Enable chat first, then choose this question. Chat only runs when a WebGPU model is loaded.';}
  }));
  $('enable-chat').addEventListener('click',enableChat);
  $('chat-form').addEventListener('submit',sendChat);
}
function chatBubble(role,message){
  const d=document.createElement('div');d.className=`chat-bubble ${role==='USER'?'user-bubble':'assistant-bubble'}`;
  const header=document.createElement('span');header.textContent=role;
  const text=document.createTextNode(message);d.append(header,text);$('chat-log').appendChild(d);$('chat-log').scrollTop=$('chat-log').scrollHeight;return d;
}
function setStatus(status,desc){$('chat-state').textContent=status;if(desc)$('chat-progress').textContent=desc;}
async function enableChat(){
  const button=$('enable-chat');button.disabled=true;
  setStatus('CHECKING WEBGPU','Checking the browser for hardware-accelerated inference…');
  try{
    if(!('gpu' in navigator))throw new Error('WebGPU is unavailable in this browser. Use desktop Chrome or Edge with supported GPU hardware.');
    const adapter=await navigator.gpu.requestAdapter();
    if(!adapter)throw new Error('No supported WebGPU adapter was found. The rest of this exhibit still works.');
    setStatus('LOADING MODEL','Downloading the small language model into this browser; first run may be substantial.');
    $('chat-progressbar').hidden=false;
    // Import only after explicit opt-in; model weights are not loaded on ordinary page visits.
    const webllm=await import('https://esm.run/@mlc-ai/web-llm');
    const preferred=['SmolLM2-360M-Instruct-q4f16_1-MLC','Qwen2.5-0.5B-Instruct-q4f16_1-MLC','Llama-3.2-1B-Instruct-q4f16_1-MLC'];
    const available=new Set(webllm.prebuiltAppConfig.model_list.map(x=>x.model_id));
    const modelId=preferred.find(x=>available.has(x));
    if(!modelId)throw new Error('No compatible small WebLLM model was found in this WebLLM version.');
    state.engine=await webllm.CreateMLCEngine(modelId,{initProgressCallback:(p)=>{
      const progress=typeof p.progress==='number'?clamp(p.progress,0,1):0;
      $('chat-progressbar').value=progress;
      $('chat-progress').textContent=(p.text||'Loading local model…').slice(0,250);
    }});
    $('chat-progressbar').hidden=true;$('chat-start').hidden=true;$('chat-form').hidden=false;
    setStatus('READY');chatBubble('LAB ASSISTANT',`Local WebLLM (${modelId}) is ready. Ask me about this synthetic dataset, the network or its test results.`);
    $('chat-input').focus();
  }catch(err){
    state.engine=null;button.disabled=false;$('chat-progressbar').hidden=true;
    setStatus('UNAVAILABLE',`Chat could not load: ${err?.message||String(err)} The prediction workbench and evaluation still work.`);
    chatBubble('LAB ASSISTANT','Chat is unavailable on this device. Please review the exhibit explanation and the exported metrics instead.');
  }
}
function contextGuide(){
  const s=statsAt(state.threshold),m=state.model.metrics;
  return `${state.guide}\nCURRENT LIVE THRESHOLD: ${state.threshold.toFixed(3)} (the original validation-chosen threshold was ${m.threshold.toFixed(3)}). Held-out confusion matrix at current live threshold: true positives ${s.tp}, true negatives ${s.tn}, false positives ${s.fp}, false negatives ${s.fn}; test accuracy ${pct(s.accuracy)}, precision ${pct(s.precision)}, recall ${pct(s.recall)}. In the current workbench, synthetic predicted assembly probability is ${pct(predict(state.controls))}; current scenario ${JSON.stringify(state.controls)}. This may change as user changes controls.`;
}
async function sendChat(event){
  event.preventDefault();if(!state.engine||state.chatBusy)return;
  const input=$('chat-input'),question=input.value.trim();if(!question)return;
  input.value='';chatBubble('USER',question);state.chatBusy=true;$('chat-send').disabled=true;
  const placeholder=chatBubble('LAB ASSISTANT','Thinking locally…');
  try{
    const response=await state.engine.chat.completions.create({
      messages:[{role:'system',content:`You are a concise museum-style guide for an educational neural network exhibit. Answer questions based ONLY on the following exhibit guide and current live results. If not covered, say "The exhibit guide does not provide that information." All dataset labels are SYNTHETIC and NOT real experimental/clinical evidence. Do not make causal or medical claims. Answer in 2-5 short sentences.\n\n${contextGuide()}`},{role:'user',content:question}],temperature:.15,max_tokens:230
    });
    const answer=response?.choices?.[0]?.message?.content||'The model returned no answer.';
    placeholder.replaceChildren();const title=document.createElement('span');title.textContent='LAB ASSISTANT';placeholder.append(title,document.createTextNode(answer));
  }catch(err){placeholder.replaceChildren();const title=document.createElement('span');title.textContent='LAB ASSISTANT';placeholder.append(title,document.createTextNode(`Local inference error: ${err?.message||String(err)}`));}
  finally{state.chatBusy=false;$('chat-send').disabled=false;input.focus();$('chat-log').scrollTop=$('chat-log').scrollHeight;}
}
async function start(){
  bindControls();
  try{
    const [modelRes,casesRes,guideRes]=await Promise.all([fetch('model/network.json'),fetch('data/heldout_test.json'),fetch('guide.txt')]);
    if(!modelRes.ok||!casesRes.ok||!guideRes.ok)throw new Error(`Missing model file, test cases or guide (${modelRes.status}, ${casesRes.status}, ${guideRes.status}).`);
    state.model=await modelRes.json();state.cases=await casesRes.json();state.guide=await guideRes.text();
    const m=state.model.metrics;
    if(state.cases.length!==m.test_count)throw new Error('Test data size does not match model metadata.');
    state.threshold=m.threshold;
    $('stat-samples').textContent=m.sample_count.toLocaleString();$('stat-test').textContent=m.test_count.toLocaleString();
    $('stat-accuracy').textContent=pct(m.accuracy);$('stat-baseline').textContent=pct(m.baseline_accuracy);
    $('split-text').textContent=`${m.train_count.toLocaleString()} training, ${m.validation_count.toLocaleString()} validation, ${m.test_count.toLocaleString()} test.`;
    $('baseline-copy').textContent=`On ${m.test_count} held-out test scenarios, the neural network achieved ${pct(m.accuracy)} accuracy versus ${pct(m.baseline_accuracy)} for a majority-class baseline fit to the training split. The network achieved ${pct(m.recall)} recall and ${pct(m.precision)} precision at the validation-chosen threshold.`;
    // Use exported test probabilities, not fabricated examples.
    const maxDiff=Math.max(...state.cases.slice(0,30).map(c=>Math.abs(predict({serotype:c.serotype,aap:c.aap,contact:c.contact,stability:c.stability,ph:c.ph,salt:c.salt,concentration:c.concentration})-c.prob)));
    if(maxDiff>.0001)throw new Error('Exported model does not match the held-out test predictions.');
    syncAll();renderThreshold();pickCases();
  }catch(err){
    console.error('Exhibit startup failed',err);
    $('pred-label').textContent='Model unavailable';$('chat-progress').textContent=`Exhibit initialization failed: ${err.message}. Please use a local web server, not file://`;
  }
}
start();
