// webui.h — the built-in test page served at GET /chat (single file, no external requests: it must work on an offline tailnet).
#pragma once
namespace ferro {
static const char* kChatHtml = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ferro</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1d212b;--line:#2a2f3b;--fg:#e6e8ee;--mut:#8b93a7;--acc:#e8883a;--acc2:#4aa3ff;--ok:#4cc38a;--bad:#ef5b5b}
*{box-sizing:border-box}html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header b{font-size:16px;letter-spacing:.5px}header b span{color:var(--acc)}
.chip{padding:2px 8px;border:1px solid var(--line);border-radius:999px;color:var(--mut);font-size:12px;white-space:nowrap}.chip.ok{color:var(--ok)}.chip.bad{color:var(--bad)}
nav{display:flex;gap:4px;margin-left:8px}nav button{background:none;border:1px solid transparent;color:var(--mut);padding:4px 10px;border-radius:6px;cursor:pointer}nav button.on{color:var(--fg);border-color:var(--line);background:var(--panel2)}
.sp{flex:1}button,select,input,textarea{font:inherit;color:var(--fg)}
.btn{background:var(--panel2);border:1px solid var(--line);padding:5px 12px;border-radius:6px;cursor:pointer}.btn:hover{border-color:var(--mut)}.btn.pri{background:var(--acc);border-color:var(--acc);color:#111;font-weight:600}.btn.stop{background:var(--bad);border-color:var(--bad);color:#111;font-weight:600}
main{display:flex;height:calc(100% - 49px)}
#chatcol{flex:1;display:flex;flex-direction:column;min-width:0}
#log{flex:1;overflow:auto;padding:14px 14px 4px}
.msg{max-width:900px;margin:0 auto 14px}.role{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--mut);margin-bottom:3px}
.user .body{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 12px;white-space:pre-wrap;word-break:break-word}
.asst .body{white-space:pre-wrap;word-break:break-word}
.asst code,.asst pre{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}.asst pre{background:#0b0d11;border:1px solid var(--line);border-radius:8px;padding:8px 10px;overflow:auto;white-space:pre}
.asst :not(pre)>code{background:#0b0d11;border:1px solid var(--line);border-radius:4px;padding:0 4px}
details.think{margin:0 0 6px;border:1px dashed var(--line);border-radius:8px;padding:4px 10px;color:var(--mut)}details.think summary{cursor:pointer;font-size:12px}details.think div{white-space:pre-wrap;margin-top:4px;font-size:13px}
.stats{margin-top:5px;font-size:11.5px;color:var(--mut);font-family:ui-monospace,Menlo,Consolas,monospace}.warn{color:var(--acc)}.err{color:var(--bad)}
#composer{border-top:1px solid var(--line);padding:10px 14px;background:var(--panel)}#composer .row{max-width:900px;margin:0 auto;display:flex;gap:8px}
#inp{flex:1;resize:none;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;max-height:200px;min-height:42px}#inp:focus{outline:1px solid var(--acc)}
#side{width:300px;border-left:1px solid var(--line);background:var(--panel);padding:12px;overflow:auto}#side h4{margin:14px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--mut)}#side h4:first-child{margin-top:0}
#side label{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:5px 0;color:var(--mut)}#side input[type=number],#side input[type=text],#side input[type=password],#side select,#side textarea{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 6px;width:130px}#side textarea{width:100%;min-height:80px}
#side input[type=range]{width:130px}
kv{display:block;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:var(--mut)}kv b{color:var(--fg);font-weight:500}
#dec{display:none;flex:1;overflow:auto;padding:16px}#dec .wrap{max-width:900px;margin:0 auto}#dec textarea{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
#dec h3{margin:14px 0 6px;font-size:13px;color:var(--mut)}#out{background:#0b0d11;border:1px solid var(--line);border-radius:8px;padding:10px;white-space:pre-wrap;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;min-height:60px}
@media(max-width:820px){#side{display:none}#side.show{display:block;position:fixed;right:0;top:49px;bottom:0;z-index:5;box-shadow:-8px 0 24px #0008}}
</style></head><body>
<header><b>fer<span>ro</span></b><span class="chip" id="model">…</span><span class="chip" id="st">connecting</span><span class="chip" id="rate"></span><span class="chip" id="mem"></span>
<nav><button id="tchat" class="on">Chat</button><button id="tdec">Decide (d1)</button></nav><span class="sp"></span>
<button class="btn" id="clear">New chat</button><button class="btn" id="gear">Settings</button></header>
<main>
 <div id="chatcol"><div id="log"></div>
  <div id="composer"><div class="row"><textarea id="inp" rows="1" placeholder="Message ferro…  (Enter to send, Shift+Enter for a new line)"></textarea><button class="btn pri" id="send">Send</button></div></div></div>
 <div id="dec"><div class="wrap">
  <h3>State (text the decision is about)</h3><textarea id="dstate" rows="3">I was charged twice for the same order, please refund one of them.</textarea>
  <h3>Image or audio (optional)</h3><input type="file" id="dfile" accept="image/*,audio/*"> <span id="dfn" class="chip" style="display:none"></span>
  <h3>Questions (JSON)</h3><textarea id="dq" rows="9">{
  "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
  "team":    {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels", "technical": "App or site faults"}},
  "urgency": {"type": "score",  "instructions": "How urgent is this?", "criteria": ["Can wait", "Today", "Blocking the customer now"]}
}</textarea>
  <p><button class="btn pri" id="drun">Decide</button> <span id="dinfo" class="chip" style="display:none"></span></p><div id="out"></div></div></div>
 <aside id="side">
  <h4>Generation</h4>
  <label>temperature <input type="number" id="temp" step="0.1" min="0" max="2" value="0.7"></label>
  <label>top_p <input type="number" id="topp" step="0.05" min="0.05" max="1" value="0.95"></label>
  <label>max tokens <input type="number" id="maxt" step="64" min="1" value="1024"></label>
  <label>seed <input type="text" id="seed" placeholder="random"></label>
  <label>thinking budget <input type="text" id="tb" placeholder="off"></label>
  <h4>Server features</h4>
  <label>speculative drafts <input type="checkbox" id="draft" checked></label>
  <label>prefix cache <input type="checkbox" id="cache" checked></label>
  <label>show reasoning <input type="checkbox" id="showthink" checked></label>
  <h4>System prompt</h4><textarea id="sys" placeholder="optional (cached after the first request)"></textarea>
  <h4>API key</h4><input type="password" id="key" placeholder="only if the server needs one" style="width:100%">
  <h4>Server</h4><kv id="info">…</kv>
 </aside>
</main>
<script>
const $=id=>document.getElementById(id);const S={msgs:[],busy:false,ctl:null,model:''};
const ls=(k,v)=>v===undefined?localStorage.getItem('ferro.'+k):localStorage.setItem('ferro.'+k,v);
for(const id of ['temp','topp','maxt','seed','tb','sys','key']){const v=ls(id);if(v!==null)$(id).value=v;$(id).addEventListener('change',()=>ls(id,$(id).value))}
for(const id of ['draft','cache','showthink']){const v=ls(id);if(v!==null)$(id).checked=v==='1';$(id).addEventListener('change',()=>ls(id,$(id).checked?'1':'0'))}
const hdr=()=>{const h={'Content-Type':'application/json'};const k=$('key').value.trim();if(k)h['Authorization']='Bearer '+k;return h};
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function md(t){let o='',i=0;const re=/```(\w*)\n?([\s\S]*?)(```|$)/g;let m;while((m=re.exec(t))){o+=inl(t.slice(i,m.index));o+='<pre><code>'+esc(m[2])+'</code></pre>';i=re.lastIndex}return o+inl(t.slice(i))}
function inl(s){return esc(s).replace(/`([^`\n]+)`/g,'<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g,'<b>$1</b>')}
function addMsg(role){const d=document.createElement('div');d.className='msg '+(role==='user'?'user':'asst');d.innerHTML='<div class="role">'+(role==='user'?'you':'ferro')+'</div><div class="think-slot"></div><div class="body"></div><div class="stats"></div>';$('log').appendChild(d);$('log').scrollTop=1e9;return d}
function scroll(){const l=$('log');if(l.scrollHeight-l.scrollTop-l.clientHeight<160)l.scrollTop=1e9}
async function refresh(){try{const r=await fetch('/health');const h=await r.json();$('st').textContent='online';$('st').className='chip ok';
 const lm=h.lm||{};let t=[];if(lm.model)t.push('<b>'+lm.model+'</b> '+(lm.device||''));
 $('info').innerHTML=[lm.model&&'chat: <b>'+lm.model+'</b>',lm.slots&&'slots <b>'+lm.active+'/'+lm.slots+'</b> active, <b>'+(lm.prefilling||0)+'</b> prefilling',lm.ctx&&'context <b>'+lm.ctx+'</b>',lm.prefix_cache&&'prefix cache <b>'+lm.prefix_cache.entries+'</b> entries, '+lm.prefix_cache.hits+' hits',h.prefix_snapshots&&'disk snapshots <b>'+h.prefix_snapshots.entries+'</b>',h.speculative&&'drafts <b>'+h.speculative.accepted+'/'+h.speculative.drafted+'</b> accepted',h.d1&&'d1: <b>'+h.d1.model+'</b>',h.model&&'embeddings: <b>'+h.model+'</b>'].filter(Boolean).join('</kv><kv>');
 const l=h.live||{};$('rate').textContent=(l.decode_tokens_per_second?l.decode_tokens_per_second.toFixed(1)+' tok/s':'idle');
 if(h.memory&&h.memory.limit_mb)$('mem').textContent='mem '+Math.round(h.memory.used_mb)+'/'+Math.round(h.memory.limit_mb)+' MB'}catch(e){$('st').textContent='offline';$('st').className='chip bad'}}
async function init(){try{const r=await fetch('/v1/models',{headers:hdr()});const j=await r.json();const m=(j.data||[]).find(x=>(x.capabilities||[]).includes('chat'));if(m){S.model=m.id;$('model').textContent=m.id+' · ctx '+(m.context_length||'?')}else $('model').textContent='no chat model loaded'}catch(e){$('model').textContent='?'}
 refresh();setInterval(refresh,2000)}
async function send(){if(S.busy)return;const text=$('inp').value.trim();if(!text)return;$('inp').value='';$('inp').style.height='auto';
 S.msgs.push({role:'user',content:text});addMsg('user').querySelector('.body').textContent=text;
 const a=addMsg('asst'),body=a.querySelector('.body'),slot=a.querySelector('.think-slot'),stats=a.querySelector('.stats');
 let content='',think='',tdet=null,tdiv=null;const t0=performance.now();let tFirst=0,tLast=0,nChunks=0,usage=null,timings=null,fin='',warns=[];
 const msgs=($('sys').value.trim()?[{role:'system',content:$('sys').value.trim()}]:[]).concat(S.msgs);
 const req={model:S.model,messages:msgs,stream:true,stream_options:{include_usage:true},temperature:+$('temp').value,top_p:+$('topp').value,max_tokens:+$('maxt').value||1024,draft:$('draft').checked,cache_prompt:$('cache').checked};
 if($('seed').value.trim()!=='')req.seed=+$('seed').value;if($('tb').value.trim()!=='')req.thinking_budget=+$('tb').value;
 S.busy=true;$('send').textContent='Stop';$('send').className='btn stop';S.ctl=new AbortController();
 try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:hdr(),body:JSON.stringify(req),signal:S.ctl.signal});
  if(!r.ok){let e='';try{e=(await r.json()).error.message}catch(_){}throw new Error('HTTP '+r.status+(e?': '+e:''))}
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
  while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let k;
   while((k=buf.indexOf('\n\n'))>=0){const ev=buf.slice(0,k);buf=buf.slice(k+2);if(!ev.startsWith('data: '))continue;const d=ev.slice(6);if(d==='[DONE]')continue;let j;try{j=JSON.parse(d)}catch(_){continue}
    if(j.usage)usage=j.usage;if(j.timings)timings=j.timings;if(j.warnings)warns=warns.concat(j.warnings);
    const c=(j.choices||[])[0];if(!c)continue;if(c.finish_reason)fin=c.finish_reason;const dl=c.delta||{};
    if(dl.reasoning_content){think+=dl.reasoning_content;if(!tdet&&$('showthink').checked){tdet=document.createElement('details');tdet.className='think';tdet.open=true;tdet.innerHTML='<summary>reasoning</summary><div></div>';tdiv=tdet.querySelector('div');slot.appendChild(tdet)}if(tdiv)tdiv.textContent=think}
    if(dl.content){if(!tFirst)tFirst=performance.now();if(tdet)tdet.open=false;content+=dl.content;body.innerHTML=md(content)}
    if(dl.reasoning_content||dl.content){if(!tFirst&&dl.reasoning_content)tFirst=performance.now();tLast=performance.now();nChunks++}
    scroll()}}
 }catch(e){if(e.name==='AbortError')fin='stopped';else{stats.innerHTML='<span class="err">'+esc(e.message)+'</span>';S.msgs.pop();S.busy=false;$('send').textContent='Send';$('send').className='btn pri';return}}
 S.msgs.push({role:'assistant',content:content});
 const p=[];if(usage){p.push('prompt '+usage.prompt_tokens+(usage.prompt_tokens_details&&usage.prompt_tokens_details.cached_tokens?' ('+usage.prompt_tokens_details.cached_tokens+' cached)':''));p.push('gen '+usage.completion_tokens)}
 if(tFirst)p.push('ttft '+Math.round(tFirst-t0)+' ms');if(usage&&tLast>tFirst&&usage.completion_tokens>1)p.push((((usage.completion_tokens-1)/((tLast-tFirst)/1000))).toFixed(1)+' tok/s');
 if(timings&&timings.prompt_ms)p.push('prefill '+Math.round(timings.prompt_ms)+' ms');if(fin)p.push(fin);
 stats.innerHTML=esc(p.join(' · '))+(warns.length?' <span class="warn">⚠ '+esc(warns.join('; '))+'</span>':'');
 S.busy=false;$('send').textContent='Send';$('send').className='btn pri';scroll();$('inp').focus()}
$('send').onclick=()=>{if(S.busy&&S.ctl)S.ctl.abort();else send()};
$('inp').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();if(!S.busy)send()}});
$('inp').addEventListener('input',e=>{e.target.style.height='auto';e.target.style.height=Math.min(200,e.target.scrollHeight)+'px'});
$('clear').onclick=()=>{if(S.ctl&&S.busy)S.ctl.abort();S.msgs=[];$('log').innerHTML='';$('inp').focus()};
$('gear').onclick=()=>$('side').classList.toggle('show');
function tab(w){const c=w==='chat';$('chatcol').style.display=c?'flex':'none';$('dec').style.display=c?'none':'block';$('tchat').className=c?'on':'';$('tdec').className=c?'':'on'}
$('tchat').onclick=()=>tab('chat');$('tdec').onclick=()=>tab('dec');
let dfile=null;$('dfile').onchange=e=>{const f=e.target.files[0];if(!f){dfile=null;$('dfn').style.display='none';return}const r=new FileReader();r.onload=()=>{dfile=r.result;$('dfn').textContent=f.name+' · '+Math.round(f.size/1024)+' KB';$('dfn').style.display=''};r.readAsDataURL(f)};
$('drun').onclick=async()=>{let q;try{q=JSON.parse($('dq').value)}catch(e){$('out').textContent='questions: invalid JSON — '+e.message;return}
 const body={state:$('dstate').value,questions:q};if(dfile)body.files=[dfile];$('out').textContent='…';$('dinfo').style.display='none';const t0=performance.now();
 try{const r=await fetch('/v1/systemone',{method:'POST',headers:hdr(),body:JSON.stringify(body)});const j=await r.json();const sm=j.answers?Object.entries(j.answers).map(([k,a])=>k+': '+(a.type==='noul'?(a.noul*100).toFixed(1)+'% yes':a.type==='choice'?a.choice+' ('+(a.confidence*100).toFixed(0)+'%)':a.type==='score'?a.score.toFixed(2):'')).join('\n')+'\n\n':'';$('out').textContent=sm+JSON.stringify(j,null,2);$('dinfo').textContent=Math.round(performance.now()-t0)+' ms · HTTP '+r.status;$('dinfo').style.display=''}catch(e){$('out').textContent=String(e)}};
init();$('inp').focus();
</script></body></html>
)HTML";
}  // namespace ferro
