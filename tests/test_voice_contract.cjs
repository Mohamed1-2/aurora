/* Mocked API contract tests. These do not establish audible output or ASR quality. */
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict'),path=require('path');
const root=process.env.AURORA_TEST_PROJECT || path.resolve(__dirname,'..');
const source=fs.readFileSync(path.join(root,'services/assistant-api/app/static/voice.js'),'utf8');
const results=[];
function harness({permissionError=false,noSpeech=false}={}) {
 const events=[];let utterance,recognizer;
 class Recognition {constructor(){recognizer=this;}start(){events.push('recognition.start');if(permissionError)this.onerror?.({error:'not-allowed'});else this.onstart?.();}abort(){events.push('recognition.abort');}}
 const synth={getVoices:()=>[],cancel:()=>events.push('speech.cancel'),speak:u=>{utterance=u;events.push('speech.start');u.onstart?.();}};
 const window={SpeechRecognition:Recognition};if(!noSpeech)window.speechSynthesis=synth;
 const context={window,speechSynthesis:synth,SpeechSynthesisUtterance:class{constructor(text){this.text=text;}},setTimeout,clearTimeout,console,performance,fetch:async()=>({ok:false}),navigator:{}};
 vm.createContext(context);vm.runInContext(source,context);
 return {voice:window.AuroraVoice,events,current:()=>utterance,recognizer:()=>recognizer};
}
async function check(name,fn){try{await fn();results.push({name,status:'passed'});}catch(e){results.push({name,status:'failed',error:String(e)});}}
(async()=>{
 await check('speech-start-and-completion',async()=>{const h=harness();let started=false;const p=h.voice.speak('Below <10%.',{onStart:()=>started=true,gap:0});assert.equal(started,true);assert.match(h.current().text,/less than 10 percent/);h.current().onend();assert.equal(await p,true);});
 await check('microphone-interrupts-speech-before-recognition',()=>{const h=harness();h.voice.speak('Long answer.');h.events.length=0;h.voice.recognize();assert.deepEqual(h.events.slice(0,2),['speech.cancel','recognition.start']);});
 await check('permission-denial-is-reported',()=>{const h=harness({permissionError:true});let error;h.voice.recognize({onError:e=>error=e});assert.equal(error,'not-allowed');});
 await check('missing-synthesis-is-unavailable',async()=>{const h=harness({noSpeech:true});let ended;assert.equal(await h.voice.speak('Hi',{onEnd:ok=>ended=ok}),false);assert.equal(ended,false);});
 await check('recognition-result-forwarding',()=>{const h=harness();let final;h.voice.recognize({onFinal:text=>final=text});h.recognizer().onresult({resultIndex:0,results:[Object.assign([{transcript:'what is my range'}],{isFinal:true})]});assert.equal(final,'what is my range');h.voice.stopRecognition();assert.ok(h.events.includes('recognition.abort'));});
 await check('studio-service-unavailable-is-honest',async()=>{const h=harness();assert.equal(await h.voice.initStudio(),false);});
 await check('playback-error-propagates',async()=>{const h=harness();let end;const p=h.voice.speak('Hello',{onEnd:ok=>end=ok});h.current().onerror({error:'audio-busy'});assert.equal(await p,false);assert.equal(end,false);});
 const report={scope:'Production voice module with mocked browser APIs. Not a real-audio or cabin-noise test.',results};
 const dest=process.argv[2];if(dest)fs.writeFileSync(dest,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));process.exitCode=results.some(r=>r.status==='failed')?1:0;
})();
