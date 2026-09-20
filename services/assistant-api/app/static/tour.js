"use strict";
/* Guided tour: presentation only. Never launches a vehicle scenario automatically. */
(() => {
  const SEEN = 'aurora-tour-v1';
  const steps = [
    {tag:'WELCOME ABOARD', title:'Meet your driving companion.', text:'AURORA connects a simulated vehicle, energy predictions and an assistant. Take a quick look around, then try it yourself. This is a simulator, not a self-driving system.', target:'.screen'},
    {tag:'01 · LIVE READINGS', title:'Your car, at a glance.', text:'Speed, battery charge and estimated range update as the simulator runs. Dimmed readings mean the feed is stale. Range is an estimate, not a guarantee.', target:'.instrument'},
    {tag:'02 · ASK NATURALLY', title:'Start with a real question.', text:'Type “What is my speed?” or “Which tyre is lowest?” and press Enter. The answer stays short. Ask “in miles?” after a range question, or “tell me more” for the full answer.', target:'#form', example:'What is my speed?'},
    {tag:'03 · VOICE', title:'Speak, repeat, or interrupt.', text:'Tap the orb to start a voice conversation, or hold the microphone for a quick question. Allow microphone access if your browser asks. Stop ends listening and speech. Typed questions always work.', target:'.orbwrap'},
    {tag:'04 · GO DEEPER', title:'Short answer. More when needed.', text:'Repeat speaks the last answer. Details opens its explanation and sources. Car status, Range and Energy are quick ways to try the system.', target:'.answer-actions'},
    {tag:'05 · EARLY WARNINGS', title:'Warnings have their own space.', text:'A priority warning appears here when the simulator detects a problem. Speak reads it; Why explains the evidence. A warning should not replace the answer to an unrelated question.', target:'#alertSlot', empty:true},
    {tag:'06 · TRY A TRIP', title:'Give the simulation a destination.', text:'Open these controls, enter a distance in kilometres and select Go. Watch the trip and energy estimates change. Climate, cruise and charging controls affect this simulator only.', target:'.vehicle-controls', prepare:'controls'},
    {tag:'07 · DEMO LAB', title:'Make something happen.', text:'The lab contains tyre-leak, cold-highway and battery-heat scenarios. Launch one yourself, then watch the readings and alerts react. The tour does not run scenarios for you.', target:'#director', prepare:'lab'},
    {tag:'08 · UNDER THE HOOD', title:'See what is actually connected.', text:'Engineering shows the answering provider, telemetry source, prediction status and voice options. A model answer and an offline template are different; provider details tell you which one responded.', target:'#engPanel', prepare:'engineering'},
    {tag:'READY TO EXPLORE', title:'Your first three things to try.', text:'Ask about speed. Ask about range, then “in miles?” Finally, try Energy to explain the forecast. Reopen this guide anytime with the Tour button.', target:'.chips', example:'What is my current range?'}
  ];
  let index=0, open=false, saved, lastFocus, narrator=false;
  const shade=document.createElement('div'); shade.id='tourShade'; shade.hidden=true;
  const spot=document.createElement('div'); spot.id='tourSpot'; spot.setAttribute('aria-hidden','true');
  const card=document.createElement('section'); card.id='tourCard'; card.hidden=true;
  card.setAttribute('role','dialog'); card.setAttribute('aria-modal','true'); card.setAttribute('aria-labelledby','tourTitle'); card.setAttribute('aria-describedby','tourText');
  card.innerHTML='<div class="tour-top"><span id="tourTag"></span><button id="tourSkip" aria-label="Close tour">✕</button></div><div class="tour-count" id="tourCount"></div><h2 id="tourTitle"></h2><p id="tourText"></p><button id="tourExample" hidden></button><div id="tourProgress" aria-label="Tour progress"></div><div class="tour-bottom"><button id="tourRead" aria-pressed="false">Read aloud</button><div><button id="tourBack">Back</button><button id="tourNext">Next →</button></div></div><span class="tour-keyhint">Arrow keys to explore · Esc to close</span>';
  document.body.append(shade,spot,card); spot.hidden=true;
  const el=id=>document.getElementById(id);
  function remember(){try{localStorage.setItem(SEEN,'seen')}catch{}}
  function restorePanels(){
    document.body.classList.toggle('eng',saved.eng);
    el('engPanel').classList.toggle('open',saved.panel);
    el('director').classList.toggle('open',saved.lab);
    el('director').setAttribute('aria-hidden',String(!saved.lab));
    document.querySelector('.vehicle-controls').open=saved.controls;
  }
  function close(){
    if(!open)return; open=false; remember(); AuroraVoice.cancel(); narrator=false;
    shade.hidden=card.hidden=spot.hidden=true;
    document.querySelectorAll('[data-tour-inert]').forEach(n=>{n.inert=false;n.removeAttribute('data-tour-inert')});
    restorePanels(); document.body.classList.remove('touring');
    window.scrollTo(saved.scrollX,saved.scrollY); lastFocus?.focus();
  }
  function position(){
    if(!open)return;
    const vw=document.documentElement.clientWidth;
    const step=steps[index]; const target=document.querySelector(step.target);
    const r=target.getBoundingClientRect();
    const pad=8, width=Math.min(vw-24,390);
    const rect={left:Math.max(8,r.left-pad), top:Math.max(8,r.top-pad), width:Math.min(vw-16,r.width+pad*2),height:Math.max(step.empty?82:0,r.height+pad*2)};
    spot.style.cssText=`left:${rect.left}px;top:${rect.top}px;width:${rect.width}px;height:${rect.height}px;`;
    card.style.width=width+'px';
    const h=card.offsetHeight;
    let x=rect.left+rect.width+20,y=rect.top;
    if(x+width>vw-12){x=Math.max(12,Math.min(rect.left,vw-width-12)); y=rect.top+rect.height+18;}
    if(y+h>innerHeight-12)y=rect.top-h-18;
    if(y<12 || h>innerHeight-24)y=Math.max(12,innerHeight-h-12);
    card.style.left=Math.min(vw-width-12,x)+'px'; card.style.top=y+'px';
  }
  function read(){
    AuroraVoice.cancel();
    el('tourRead').setAttribute('aria-pressed',String(narrator));
    if(narrator){const s=steps[index]; AuroraVoice.speak(`${s.title} ${s.text}`,{rate:AuroraVoice.studioState().engine==='studio'?175:1,onEnd:()=>{}});}
  }
  function render(){
    AuroraVoice.cancel(); restorePanels();
    const s=steps[index];
    if(s.prepare==='controls')document.querySelector('.vehicle-controls').open=true;
    if(s.prepare==='engineering'){document.body.classList.add('eng');el('engPanel').classList.add('open'); window.AURORA.refreshEngineering?.();}
    if(s.prepare==='lab'){document.body.classList.add('eng');el('director').classList.add('open');el('director').setAttribute('aria-hidden','false');window.AURORA.refreshDirector?.();}
    el('tourTag').textContent=s.tag;el('tourTitle').textContent=s.title;el('tourText').textContent=s.text;
    el('tourCount').textContent=`${index+1} / ${steps.length}`;
    el('tourNext').textContent=index===steps.length-1?'Start exploring ✓':'Next →';
    el('tourBack').disabled=index===0;
    el('tourExample').hidden=!s.example;el('tourExample').textContent=s.example?`Try: “${s.example}”`:'';
    el('tourExample').onclick=()=>{close();el('q').value=s.example;el('q').focus();};
    el('tourProgress').replaceChildren(...steps.map((_,i)=>{const b=document.createElement('button');b.setAttribute('aria-label',`Step ${i+1}: ${steps[i].title}`);b.setAttribute('aria-current',String(i===index));b.className=i===index?'current':'';b.onclick=()=>go(i);return b;}));
    const target=document.querySelector(s.target); target.scrollIntoView({block:'center',behavior:'instant'});
    position(); requestAnimationFrame(position); read(); el('tourNext').focus();
  }
  function go(i){if(i>=steps.length){close();return}index=Math.max(0,i);render()}
  function start(){
    if(open)return; lastFocus=document.activeElement;
    saved={eng:document.body.classList.contains('eng'),panel:el('engPanel').classList.contains('open'),lab:el('director').classList.contains('open'),controls:document.querySelector('.vehicle-controls').open,scrollX:scrollX,scrollY:scrollY};
    el('stopSpeech').click(); open=true;index=0;narrator=false;
    document.body.classList.add('touring');
    [...document.body.children].filter(n=>n!==shade&&n!==spot&&n!==card&&n.tagName!=='SCRIPT'&&!n.inert).forEach(n=>{n.inert=true;n.setAttribute('data-tour-inert','')});
    el('tourRead').textContent='Read aloud';
    shade.hidden=spot.hidden=card.hidden=false;render();
  }
  el('tourSkip').onclick=close;el('tourBack').onclick=()=>go(index-1);el('tourNext').onclick=()=>go(index+1);
  el('tourRead').onclick=()=>{narrator=!narrator;el('tourRead').textContent=narrator?'Stop narration':'Read aloud';read()};
  document.addEventListener('keydown',e=>{
    if(!open)return;
    if(e.key==='Escape'){e.preventDefault();close();}
    else if(e.key==='ArrowRight'){e.preventDefault();go(index+1)}
    else if(e.key==='ArrowLeft'){e.preventDefault();go(index-1)}
    else if(e.key==='Tab'){
      const buttons=[...card.querySelectorAll('button')].filter(b=>!b.disabled&&!b.hidden);
      const first=buttons[0],last=buttons.at(-1);
      if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus()}
      else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus()}
    }
  });
  window.addEventListener('resize',position);window.addEventListener('scroll',position,true);
  el('tourStart').onclick=start;
  window.addEventListener('load',()=>{let seen=false;try{seen=!!localStorage.getItem(SEEN)}catch{}if(!seen)setTimeout(start,900)});
})();
