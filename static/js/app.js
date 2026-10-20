/* PhishGuard AI front-end. Sections: helpers · sfx · init(data+charts) · map · threshold · result renderer · manual · AI scan · batch · arena · UI chrome */
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],api=u=>fetch(u).then(r=>r.json());
const post=(u,b)=>fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)}).then(async r=>{const d=await r.json();if(!r.ok)throw Error(d.error||"Request failed");return d});
const NAMES={decision_tree:"Decision Tree",knn:"KNN",naive_bayes:"Naïve Bayes",svm:"SVM",boosting:"Boosting"};
const COL=["#d4ff3a","#ff4d2e","#ffb020","#6aa8ff","#cfd8dc"],pretty=f=>f.replace(/_/g," "),esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let META,MET,DS,AIM,charts={};
const pc=v=>(v*100).toFixed(2)+"%",wait=ms=>new Promise(r=>setTimeout(r,ms));
const mk=(id,cfg)=>{const el=$("#"+id);el._cfg=cfg;
  const draw=()=>{charts[id]?.destroy();Chart.defaults.color="#71818b";Chart.defaults.font.family="JetBrains Mono, monospace";Chart.defaults.borderColor="rgba(255,255,255,.07)";charts[id]=new Chart(el,el._cfg)};
  if(charts[id]||el._seen){draw();return}
  new IntersectionObserver((es,o)=>{if(es[0].isIntersecting){el._seen=1;o.disconnect();draw()}},{threshold:.05}).observe(el)};
const counter=(el,to,dec=0)=>{let t=0;const s=()=>{t=Math.min(1,t+.04);el.textContent=(to*t).toFixed(dec);if(t<1)requestAnimationFrame(s)};s()};
const card=(l,v,dec=0)=>{const d=document.createElement("div");d.className="card stat reveal";d.innerHTML=`<b>0</b><span>${l}</span>`;typeof v==="number"?counter(d.firstChild,v,dec):d.firstChild.textContent=v;return d};
const table=(el,head,rows)=>{el.innerHTML="<tr>"+head.map(h=>`<th>${h}</th>`).join("")+"</tr>"+rows.map(r=>"<tr>"+r.map(c=>`<td>${c}</td>`).join("")+"</tr>").join("")};
const io=new IntersectionObserver(es=>es.forEach(e=>e.isIntersecting&&e.target.classList.add("in")),{threshold:.02,rootMargin:"0px 0px 12% 0px"});
const watch=()=>$$(".reveal").forEach(e=>io.observe(e));
const vclass=l=>l=="PHISHING"?"bad":l=="LEGITIMATE"?"good":"warn";

/* ---------- sound effects (off by default) ---------- */
const SFX={on:false,ctx:null,tone(f,d=.12,type="square",v=.05,at=0){if(!this.on)return;try{this.ctx??=new (window.AudioContext||window.webkitAudioContext)();const c=this.ctx,o=c.createOscillator(),g=c.createGain();
  o.type=type;o.frequency.value=f;g.gain.setValueAtTime(v,c.currentTime+at);g.gain.exponentialRampToValueAtTime(.0001,c.currentTime+at+d);o.connect(g).connect(c.destination);o.start(c.currentTime+at);o.stop(c.currentTime+at+d)}catch(e){this.on=false}},
  click(){this.tone(900,.05)},verdict(l){l=="PHISHING"?[220,160,110].forEach((f,i)=>this.tone(f,.2,"sawtooth",.06,i*.12)):l=="LEGITIMATE"?[520,660,880].forEach((f,i)=>this.tone(f,.14,"sine",.06,i*.09)):[440,440].forEach((f,i)=>this.tone(f,.15,"triangle",.06,i*.2))}};
document.addEventListener("click",e=>e.target.closest("button,a")&&SFX.click());
const toggleSfx=()=>{SFX.on=!SFX.on;$("#sfxBtn").textContent="SFX "+(SFX.on?"on":"off");SFX.click()};

/* ---------- init: load data, draw charts ---------- */
async function init(){
  [META,MET,DS,AIM]=await Promise.all([api("/api/meta"),api("/api/metrics"),api("/api/dataset"),api("/api/ai_metrics")]);
  const c=$("#dsCards");[["Records in file",DS.raw_rows],["Unique records used",DS.rows],["Features",DS.features],["Phishing",DS.phishing],["Legitimate",DS.legitimate]].forEach(x=>c.append(card(...x)));
  mk("distChart",{type:"doughnut",data:{labels:["Phishing","Legitimate"],datasets:[{data:[DS.phishing,DS.legitimate],backgroundColor:["#ff4d2e","#d4ff3a"]}]},options:{plugins:{title:{display:true,text:"Class distribution"}}}});
  $("#featSel").innerHTML=META.features.map(f=>`<option value="${f}">${META.info[f].title}</option>`).join("");
  const fc=()=>{const v=DS.value_counts[$("#featSel").value];mk("featChart",{type:"bar",data:{labels:["-1","0","1"],datasets:[{data:["-1","0","1"].map(k=>v[k]||0),backgroundColor:COL[3]}]},options:{plugins:{legend:{display:false}}}})};
  $("#featSel").onchange=fc;fc();
  table($("#statTbl"),["Feature","Mean","Std","Min","Max"],DS.stats.map(s=>[pretty(s.feature),s.mean,s.std,s.min,s.max]));
  table($("#prevTbl"),Object.keys(DS.preview[0]).map(pretty),DS.preview.map(r=>Object.values(r)));
  const ks=Object.keys(MET.models),mets=["accuracy","precision","recall","f1","roc_auc"];
  $("#bestNote").textContent=`Trained on ${DS.rows} unique records (duplicates removed from the ${DS.raw_rows} in the file). Best model by F1 on the held-out ${DS.test}-record test set: ${NAMES[MET.best]}. `+(DS.legacy_rows?`LEGACY-SET ACC = accuracy on the ${DS.legacy_rows}-record older dataset (_old.arff), never used in training.`:"(Legacy validation file not found: copy _old.arff into data/ and re-run train.py to enable the legacy-set column.)");
  mk("cmpChart",{type:"bar",data:{labels:mets.map(m=>m.toUpperCase()),datasets:ks.map((k,i)=>({label:NAMES[k],data:mets.map(m=>MET.models[k][m]),backgroundColor:COL[i]}))},options:{scales:{y:{min:.8,max:1}},animation:{duration:1200}}});
  mk("rocChart",{type:"line",data:{labels:MET.models[ks[0]].roc.fpr.map(x=>x.toFixed(2)),datasets:ks.map((k,i)=>({label:NAMES[k],data:MET.models[k].roc.tpr,borderColor:COL[i],pointRadius:0,tension:.3}))},options:{plugins:{title:{display:true,text:"ROC curves (x: FPR, y: TPR)"}}}});
  table($("#mTbl"),["Model",...mets.map(m=>m.toUpperCase()),"LEGACY-SET ACC"],ks.map(k=>[NAMES[k]+(k==MET.best?" ★":""),...mets.map(m=>pc(MET.models[k][m])),MET.external?pc(MET.external[k].accuracy):"–"]));
  ks.forEach(k=>{const cm=MET.models[k].confusion,d=document.createElement("div");d.className="card reveal";
    d.innerHTML=`<h3>${NAMES[k]}</h3><table><tr><th></th><th>Pred Legit</th><th>Pred Phish</th></tr><tr><th>Actual Legit</th><td class="good">${cm[0][0]}</td><td>${cm[0][1]}</td></tr><tr><th>Actual Phish</th><td>${cm[1][0]}</td><td class="good">${cm[1][1]}</td></tr></table>`;$("#cms").append(d)});
  const top=MET.importance.slice(0,12);
  mk("impChart",{type:"bar",data:{labels:top.map(t=>META.info[t.feature].title),datasets:[{data:top.map(t=>t.value),backgroundColor:COL[0]}]},options:{indexAxis:"y",plugins:{legend:{display:false}}}});
  if(AIM.models){const rows=[];for(const s of ["url","page"])for(const k of Object.keys(AIM.models[s])){const m=AIM.models[s][k];
    rows.push([s=="url"?`Domain only (${AIM.sets.url.length})`:`Domain + page (${AIM.sets.page.length})`,NAMES[k],pc(m.accuracy),pc(m.precision),pc(m.recall),pc(m.f1),pc(m.roc_auc),m.train_rows.toLocaleString()])}
    table($("#aiTbl"),["Feature set","Model","Accuracy","Precision","Recall","F1","ROC-AUC","Train rows"],rows);
    $("#aiNote").textContent=`${AIM.source}: ${AIM.rows.toLocaleString()} unique URLs (${AIM.phishing.toLocaleString()} phishing). Train/test split is grouped by domain (no site appears on both sides). KNN and SVM use a subsample. ${AIM.caveat}`}
  // threshold explorer
  $("#thModel").innerHTML=ks.map(k=>`<option value="${k}" ${k==MET.best?"selected":""}>${NAMES[k]}</option>`).join("");
  const th=()=>{const m=$("#thModel").value,t=+$("#thRange").value,cur=MET.threshold[m],r=cur.find(x=>Math.abs(x.t-t)<.001);
    $("#thVal").textContent=t.toFixed(2);
    $("#thStats").innerHTML=[["Precision",pc(r.precision)],["Recall",pc(r.recall)],["F1",pc(r.f1)],["Accuracy",pc(r.accuracy)],["Caught phishing",`${r.tp} / ${r.tp+r.fn}`],["False alarms",`${r.fp} legit flagged`]].map(x=>`<div class="card stat"><b style="font-size:1.4rem">${x[1]}</b><span>${x[0]}</span></div>`).join("");
    const el=$("#thChart");if(charts.thChart){const ch=charts.thChart;ch.data.datasets=thSets(m,t);ch.update("none")}else mk("thChart",{type:"line",data:{datasets:thSets(m,t)},options:{parsing:false,scales:{x:{type:"linear",min:0,max:1,title:{display:true,text:"decision threshold"}},y:{min:0,max:1}},animation:false,plugins:{legend:{labels:{boxWidth:10}}}}})};
  const thSets=(m,t)=>{const cur=MET.threshold[m],r=cur.find(x=>Math.abs(x.t-t)<.001);return [["precision",COL[3]],["recall",COL[1]],["f1",COL[0]]].map(([k,c])=>({label:k,data:cur.map(x=>({x:x.t,y:x[k]})),borderColor:c,pointRadius:0,borderWidth:2})).concat([{type:"scatter",label:"selected",data:[{x:t,y:r.f1}],backgroundColor:"#fff",pointRadius:6}])};
  $("#thRange").oninput=th;$("#thModel").onchange=th;th();
  // predict panel
  const opts=b=>`<option value="best">Best Model (${b})</option>`+ks.map(k=>`<option value="${k}">${NAMES[k]}</option>`).join("");
  $("#modelSel").innerHTML=opts(NAMES[MET.best]);$("#aiModel").innerHTML=opts(META.ai_best?NAMES[META.ai_best]:"–");
  $("#form").innerHTML=META.features.map(f=>{const i=META.info[f];return `<label>${i.title}<select data-f="${f}">${META.allowed[f].slice().reverse().map(v=>`<option value="${v}">${v} · ${i.options[v]}</option>`).join("")}</select></label>`}).join("");
  $("#hud").innerHTML=[[DS.raw_rows.toLocaleString(),"records in file"],[DS.features,"features"],[5,"classifiers"],[(MET.models[MET.best].accuracy*100).toFixed(1)+"%","best accuracy · "+NAMES[MET.best]],[(MET.models[MET.best].roc_auc*100).toFixed(1),"best ROC-AUC"]].map(x=>`<div><b>${x[0]}</b><span>${x[1]}</span></div>`).join("");
  drawDatasetMap("uci");watch();dash();
}

/* ---------- feature-space map ---------- */
const mapPts={};let lastPlace=null;
function drawMap(cv,pts,place){
  cancelAnimationFrame(cv._raf);const dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;const x=cv.getContext("2d");x.scale(dpr,dpr);
  const base=document.createElement("canvas");base.width=W*dpr;base.height=H*dpr;const b=base.getContext("2d");b.scale(dpr,dpr);
  pts.forEach(([px,py,l])=>{b.fillStyle=l?"rgba(255,77,46,.55)":"rgba(212,255,58,.45)";b.fillRect(8+px*(W-16),8+py*(H-16),2,2)});
  let t=0;const frame=()=>{x.clearRect(0,0,W,H);x.drawImage(base,0,0,W,H);
    if(place){const mx=8+place.x*(W-16),my=8+place.y*(H-16),a=getComputedStyle(document.documentElement).getPropertyValue("--a").trim()||"#d4ff3a";t=(t+.02)%1;
      x.strokeStyle="#fff";x.lineWidth=1;x.globalAlpha=.35;x.beginPath();x.moveTo(mx,0);x.lineTo(mx,H);x.moveTo(0,my);x.lineTo(W,my);x.stroke();x.globalAlpha=1-t;x.strokeStyle=a;x.lineWidth=2;x.beginPath();x.arc(mx,my,6+t*34,0,7);x.stroke();
      x.globalAlpha=1;x.fillStyle="#fff";x.beginPath();x.arc(mx,my,5,0,7);x.fill();x.fillStyle=a;x.font="11px JetBrains Mono, monospace";x.fillText("YOUR SCAN",mx+10,my-10);cv._raf=requestAnimationFrame(frame)}};frame()}
async function drawDatasetMap(name){
  try{mapPts[name]??=await api("/api/map/"+name);const cv=$("#mapCv");mapPts[name].error?.length;
    drawMap(cv,mapPts[name],lastPlace?.map==name?lastPlace:null);$("#mapNote").textContent=name=="uci"?"Each dot is one of the unique UCI-327 records (t-SNE of the 30 features). Manual scans land at the average position of their 10 nearest records.":"Each dot is one of 6,000 sampled PhiUSIIL URLs (t-SNE of the domain features). AI scans land at the average position of their 8 nearest URLs."}
  catch(e){$("#mapNote").textContent="Map not available - run the training scripts to build it."}}
$$("#mapTog button").forEach(b=>b.onclick=()=>{$$("#mapTog button").forEach(x=>x.classList.toggle("on",x==b));drawDatasetMap(b.dataset.m)});
addEventListener("resize",()=>{const on=$("#mapTog .on");on&&mapPts[on.dataset.m]&&drawDatasetMap(on.dataset.m)});

/* ---------- result renderer (manual + AI) ---------- */
function stripHtml(d){const keys=Object.keys(d.impacts);return `<h3>Feature strip <small class="muted">· each cell is one feature · red pushes toward phishing, lime toward legitimate</small></h3><div class="strip">`+
  keys.map((k,i)=>{const v=d.impacts[k],a=Math.min(1,Math.abs(v)*2.2+.08);return `<i style="--c:${v>0?"255,77,46":"212,255,58"};--a:${a};animation-delay:${i*25}ms" data-k="${esc(k)}" title="${esc(d.titles[k])}: ${d.values[k]}"></i>`}).join("")+`</div><p class="muted" id="stripInfo">Click a cell to inspect a feature.</p>`}
function anatomyHtml(a){return `<h3>URL anatomy</h3><div class="anat"><span class="p-s">${esc(a.scheme)}</span>${a.sub?`<span class="p-sub">${esc(a.sub)}.</span>`:""}<span class="p-d">${esc(a.domain)}</span><span class="p-s">${esc(a.port)}</span><span class="p-path">${esc(a.path)}</span></div>`+
  (a.flags.length?a.flags.map(f=>`<span class="chip bad-chip">${esc(f)}</span>`).join(""):'<span class="chip">no structural red flags</span>')}
async function renderResult(d){
  const r=$("#result"),warn=d.label=="SUSPICIOUS"||d.label=="INCONCLUSIVE",ph=d.label=="PHISHING",col=warn?"var(--warn)":ph?"var(--bad)":"var(--ok)";
  window.LAST={features:d.values,mode:d.mode};lastPlace=d.place||null;
  r.className="card";r.classList.remove("hidden");
  r.innerHTML=`<div class="res"><div class="gauge" style="--gc:${col}"><svg viewBox="0 0 110 62"><path class="track" pathLength="100" d="M10 55A45 45 0 0 1 100 55"/><path id="garc" class="arc" pathLength="100" stroke-dasharray="0 100" d="M10 55A45 45 0 0 1 100 55"/></svg><b>${warn?(d.label=="SUSPICIOUS"?"!":"?"):(d.probability_phishing*100).toFixed(0)}<small>${warn?(d.label=="SUSPICIOUS"?"RED FLAGS":"UNVERIFIED"):"% PHISHING RISK"}</small></b></div>
   <div><div class="verdict ${vclass(d.label)}">${d.label}</div><p>MODEL <b>${NAMES[d.model]}</b> · ${warn?"MODEL P(PHISHING)":"CONFIDENCE"} <b>${(warn?d.probability_phishing*100:d.confidence*100).toFixed(1)}%</b> · INPUT <b>${d.mode=="full"?"30 UCI features":d.mode=="page"?"domain + page":"domain only"}</b></p></div></div>
   ${d.note?`<p class="sum warnbox">${esc(d.note)}</p>`:""}<p class="sum">${esc(d.summary)}</p>${d.anatomy?anatomyHtml(d.anatomy):""}${stripHtml(d)}
   <h3>Top contributing features</h3><p class="muted">Positive = pushes toward phishing.</p>`+
   d.explanation.map(e=>{const w=Math.min(100,Math.abs(e.impact)*200);return `<div class="imp"><span>${esc(e.title)} (${esc(e.shown)})</span><div class="bar"><i style="width:${w}%;background:${e.impact>0?"var(--bad)":"var(--ok)"}"></i></div><span>${e.impact>0?"+":""}${e.impact.toFixed(2)}</span></div>`}).join("")+
   (d.place?`<h3>Where this sits in the dataset</h3><canvas id="resMap" class="mapcv"></canvas>`:"")+(d.caveat?`<p class="muted small">${esc(d.caveat)}</p>`:"")+
   `<div id="cmp" style="margin-top:18px"><button class="btn sm ghost" id="cmpBtn">Race all 5 models</button></div>`;
  r.className="card "+(warn?"r-warn":ph?"r-bad":"r-good");
  $("#cmpBtn").onclick=cmpAll;$$(".strip i").forEach(i=>i.onclick=()=>{$("#stripInfo").textContent=`${d.titles[i.dataset.k]} = ${d.values[i.dataset.k]} · impact ${d.impacts[i.dataset.k]>0?"+":""}${d.impacts[i.dataset.k].toFixed(3)}`});
  setTimeout(()=>$("#garc").style.strokeDasharray=warn?"3 2":(d.probability_phishing*100)+" 100",60);
  if(d.place){const pts=mapPts[d.place.map]??=await api("/api/map/"+d.place.map);drawMap($("#resMap"),pts,d.place)}
  SFX.verdict(d.label);dash();
}

/* ---------- manual mode ---------- */
function setForm(v){$$("#form select").forEach(s=>{s.value=v[s.dataset.f]??1;if(s.value==="")s.value="1"})}
$("#resetBtn").onclick=()=>{setForm({});$("#result").classList.add("hidden");$("#sampleBtn").textContent="Load real test sample"};
$("#sampleBtn").onclick=async()=>{const s=await api("/api/sample");setForm(s.features);$("#sampleBtn").textContent="Loaded (actual label: "+s.actual+")"};
$("#goBtn").onclick=async()=>{
  const f={};$$("#form select").forEach(s=>f[s.dataset.f]=+s.value);
  const r=$("#result");r.classList.remove("hidden");r.className="card scan";r.innerHTML='<div class="spin"></div>';r.scrollIntoView({behavior:"smooth",block:"center"});
  try{await renderResult(await post("/api/predict",{model:$("#modelSel").value,features:f}))}catch(e){r.className="card";r.innerHTML=`<p class="bad">Error: ${esc(e.message)}</p>`}};

/* ---------- AI scan: request -> replay real scan log -> render ---------- */
async function aiScan(url){
  const st=$("#aiStatus"),ap=$("#aiPanel");ap.classList.add("scan");st.innerHTML='<div class="skel"></div><div class="skel" style="width:70%"></div>';
  try{
    const d=await post("/api/analyze",{url,model:$("#aiModel").value});ap.classList.remove("scan");
    st.innerHTML='<pre class="log" id="scanLog"></pre>';const lg=$("#scanLog");
    for(const l of d.log){lg.textContent+=`[${String(l.t).padStart(5)} ms] ${l.msg}\n`;SFX.tone(1200,.03,"square",.02);await wait(170)}
    lg.textContent+=`[ done ] scored by ${NAMES[d.model]} on ${d.mode=="page"?"domain + page":"domain-only"} features\n`;await wait(250);
    const r=$("#result");r.classList.remove("hidden");r.scrollIntoView({behavior:"smooth",block:"center"});await renderResult(d);
  }catch(e){ap.classList.remove("scan");st.innerHTML=`<p class="bad">Error: ${esc(e.message)}</p>`}}
$("#aiBtn").onclick=()=>aiScan($("#urlIn").value);$("#urlIn").addEventListener("keydown",e=>e.key=="Enter"&&aiScan($("#urlIn").value));
["https://www.wikipedia.org","https://github.com/anthropics","http://125.98.3.123/fake.html","http://www.Confirme-paypal.com/","https://paypal-secure-login.verify-account.tk/signin","bit.ly/19DXSk4","http://www.hud.ac.uk/students/"].forEach(u=>{
  const b=document.createElement("button");b.className="chip";b.textContent=u.length>34?u.slice(0,34)+"…":u;b.onclick=()=>{$("#urlIn").value=u;aiScan(u)};$("#examples").append(b)});
$$("#modeTog button").forEach(b=>b.onclick=()=>setMode(b.dataset.m));
function setMode(m){const ai=m=="ai";$("#modeTog").classList.toggle("manual",!ai);$$("#modeTog button").forEach(x=>x.classList.toggle("on",x.dataset.m==m));$("#aiPanel").classList.toggle("hidden",!ai);$("#manualPanel").classList.toggle("hidden",ai)}

/* ---------- model arena ---------- */
async function cmpAll(){
  const box=$("#cmp");box.innerHTML='<h3>Racing the models…</h3>'+Object.keys(NAMES).map(k=>`<div class="vote"><span>${NAMES[k]}</span><div class="bar"><i class="race"></i></div><span>…</span></div>`).join("");
  try{const d=await post("/api/compare",window.LAST);await wait(500);
    box.innerHTML=`<h3>Arena result — ${d.phishing_votes}/5 models say PHISHING</h3>`+d.results.map(x=>`<div class="vote"><span>${NAMES[x.model]}</span><div class="bar"><i style="background:${x.label=="PHISHING"?"var(--bad)":"var(--ok)"}"></i></div><span class="${vclass(x.label)}">${x.label} ${(Math.max(x.probability_phishing,1-x.probability_phishing)*100).toFixed(0)}%</span></div>`).join("");
    const bars=$$("#cmp .bar i");d.results.forEach((x,i)=>setTimeout(()=>{bars[i].style.width=x.probability_phishing*100+"%";SFX.tone(300+x.probability_phishing*600,.08,"triangle",.04)},i*260))
  }catch(e){box.innerHTML=`<p class="bad">Error: ${esc(e.message)}</p>`}}

/* ---------- batch ---------- */
let batchRows=[];
$("#batchBtn").onclick=async()=>{const t=$("#batchTbl");t.innerHTML='<tr><td><div class="skel" style="width:260px"></div></td></tr>';$("#batchPanel").classList.add("scan");
  try{const d=await post("/api/batch",{urls:$("#batchIn").value.split("\n"),model:$("#aiModel").value||"best"});batchRows=d.results;
    table(t,["URL","Result","Confidence","Input set"],d.results.map(x=>[esc(x.url.slice(0,60)),x.error?`<span class="bad">${esc(x.error)}</span>`:`<span class="${vclass(x.label)}">${x.label}</span>`,x.error?"–":(x.confidence*100).toFixed(1)+"%",x.mode||"–"]));
    $("#batchDl").classList.remove("hidden");dash()}
  catch(e){t.innerHTML=`<tr><td class="bad">Error: ${esc(e.message)}</td></tr>`}finally{$("#batchPanel").classList.remove("scan")}};
$("#batchDl").onclick=()=>{const csv="url,result,confidence,input_set\n"+batchRows.map(x=>[`"${x.url.replace(/"/g,'""')}"`,x.label||"error",x.confidence?x.confidence.toFixed(4):"",x.mode||""].join(",")).join("\n");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="batch_scan.csv";a.click()};

/* ---------- dashboard ---------- */
async function dash(){
  const h=await api("/api/history"),c=$("#dashCards");c.innerHTML="";
  [["Dataset records",DS.rows],["Models trained",5],[`Best F1 – ${NAMES[MET.best]} (%)`,MET.models[MET.best].f1*100,2],["Predictions made",h.total]].forEach(x=>{const e=card(...x);e.classList.add("in");c.append(e)});
  const s=card("System status",AIM?.models?"● Online · manual + AI models loaded":"● Online · AI models missing");s.classList.add("in");c.append(s);
  mk("predChart",{type:"doughnut",data:{labels:["Phishing","Legitimate","Suspicious / inconclusive"],datasets:[{data:[h.phishing,h.legitimate,h.other],backgroundColor:["#ff4d2e","#d4ff3a","#ffb020"]}]}});
  table($("#histTbl"),["Time","Model","Result","Conf."],h.history.map(x=>[x.time,NAMES[x.model],`<span class="${vclass(x.label)}">${x.label}</span>`,(x.confidence*100).toFixed(1)+"%"]));
}
init().catch(e=>document.body.insertAdjacentHTML("afterbegin",`<p class="bad" style="padding:20px">Failed to load: ${esc(e.message)}. Did you run train.py?</p>`));

/* ---------- chrome: boot, reticle, clock, rail, command palette ---------- */
(()=>{const L=["> PHISHGUARD//KERNEL v2.1","> mounting datasets .......... OK","> loading classifiers ........ OK","> calibrating threat radar ... OK","> SYSTEM ONLINE"],b=$("#boot");let i=0;
 const t=setInterval(()=>{b.textContent+=L[i++]+"\n";if(i>=L.length){clearInterval(t);setTimeout(()=>$("#loader").classList.add("off"),400)}},160)})();
const ret=$("#ret");addEventListener("pointermove",e=>{ret.style.transform=`translate(${e.clientX}px,${e.clientY}px)`;ret.classList.toggle("hot",!!e.target.closest("a,button,select,input,.strip i"))});
setInterval(()=>$("#clk").textContent=new Date().toLocaleTimeString([],{hour12:false}),1000);
addEventListener("scroll",()=>{const h=document.documentElement;$("#prog").style.width=100*h.scrollTop/(h.scrollHeight-h.clientHeight)+"%"},{passive:true});
$$("section h2").forEach((h,i)=>h.dataset.idx="// 0"+(i+1));$$(".reveal").forEach((e,i)=>e.style.setProperty("--i",i%4));
(()=>{const w=["phishing URLs","fake login pages","malicious links","cloned websites"];let i=0,k=0,del=false;const el=$("#tw");
 (function t(){const s=w[i];el.textContent=s.slice(0,k);k+=del?-1:1;if(!del&&k>s.length+8)del=true;if(del&&k<0){del=false;i=(i+1)%w.length}setTimeout(t,del?35:80)})()})();
const go=id=>$("#"+id).scrollIntoView({behavior:"smooth"});
$$("#links a").forEach(a=>a.onclick=e=>{e.preventDefault();go(a.getAttribute("href").slice(1))});
{let n=0;const A=["#d4ff3a","#35e0ff","#ffb020"];window.cycleAccent=()=>document.documentElement.style.setProperty("--a",A[n=(n+1)%3]);$("#theme").onclick=cycleAccent}
$("#sfxBtn").onclick=toggleSfx;
// side rail: one dot per section, highlights the one in view
const secs=$$("section"),rail=$("#rail");secs.forEach(s=>{const a=document.createElement("a");a.href="#"+s.id;a.dataset.t=s.id=="home"?"Home":s.querySelector("h2")?.textContent||s.id;a.onclick=e=>{e.preventDefault();go(s.id)};rail.append(a)});
secs.forEach(s=>new IntersectionObserver(es=>es[0].isIntersecting&&$$("#rail a").forEach(a=>a.classList.toggle("on",a.getAttribute("href")=="#"+s.id)),{threshold:.35}).observe(s));
// command palette (Ctrl/Cmd+K)
const CMDS=[...[["home","Home"],["about","About"],["dataset","Dataset analysis"],["models","Model comparison"],["predict","Scan a website"],["dashboard","Dashboard"]].map(([id,l])=>["Go to · "+l,()=>go(id)]),
 ["Switch to AI Auto-Analyze mode",()=>{setMode("ai");go("predict")}],["Switch to Manual mode",()=>{setMode("manual");go("predict")}],
 ["Run a real manual test sample",async()=>{setMode("manual");go("predict");await $("#sampleBtn").onclick();$("#goBtn").click()}],
 ["Scan example URL: wikipedia.org",()=>{setMode("ai");go("predict");$("#urlIn").value="https://www.wikipedia.org";aiScan("https://www.wikipedia.org")}],
 ["Cycle accent colour",()=>cycleAccent()],["Toggle sound effects",toggleSfx],["Download prediction report (CSV)",()=>location.href="/api/report"]];
const cmdk=$("#cmdk"),cin=$("#cmdIn");let sel=0,shown=[];
const renderCmd=()=>{const q=cin.value.toLowerCase();shown=CMDS.filter(c=>c[0].toLowerCase().includes(q));sel=Math.min(sel,Math.max(0,shown.length-1));
  $("#cmdList").innerHTML=shown.map((c,i)=>`<div class="${i==sel?"on":""}" data-i="${i}">${esc(c[0])}</div>`).join("")||'<div class="muted">No match</div>'};
const openCmd=()=>{cmdk.classList.remove("hidden");cin.value="";sel=0;renderCmd();cin.focus()},closeCmd=()=>cmdk.classList.add("hidden"),runCmd=i=>{closeCmd();shown[i]?.[1]()};
addEventListener("keydown",e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()=="k"){e.preventDefault();cmdk.classList.contains("hidden")?openCmd():closeCmd()}
  else if(!cmdk.classList.contains("hidden")){if(e.key=="Escape")closeCmd();else if(e.key=="ArrowDown"){sel=(sel+1)%shown.length;renderCmd();e.preventDefault()}else if(e.key=="ArrowUp"){sel=(sel-1+shown.length)%shown.length;renderCmd();e.preventDefault()}else if(e.key=="Enter")runCmd(sel)}});
cin.oninput=()=>{sel=0;renderCmd()};$("#cmdList").onclick=e=>e.target.dataset.i&&runCmd(+e.target.dataset.i);cmdk.onclick=e=>e.target==cmdk&&closeCmd();$("#cmdBtn").onclick=openCmd;
