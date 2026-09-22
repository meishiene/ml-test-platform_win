const state = { protocols: [], protocol: 'openai', testKind: 'load', cases: [], selected: [], temporaryCase: '', runs: [], filter: 'all', models: [], downloadSelected: new Set() };
const $ = (s) => document.querySelector(s);
const escapeHtml = (s='') => String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const FORM_STORAGE_KEY = 'ml-test-platform.form.v1';

function readPersistedForm() {
  try {
    const value = JSON.parse(localStorage.getItem(FORM_STORAGE_KEY) || 'null');
    return value && typeof value === 'object' ? value : {};
  } catch (e) {
    return {};
  }
}

function persistForm() {
  try {
    const values = {
      protocol: state.protocol,
      testKind: state.testKind,
      selected: state.selected,
      temporaryCase: state.temporaryCase,
      baseUrl: $('#baseUrl').value,
      apiKey: $('#apiKey').value,
      model: $('#model').value,
      quick: $('#quick').checked,
      customEnabled: $('#customEnabled').checked,
      customRequests: $('#customRequests').value,
      customConcurrency: $('#customConcurrency').value,
      moderationConcurrency: $('#moderationConcurrency').value,
      imageQuality: selectedValues('#imageQuality'),
      imageModels: selectedValues('#imageModels'),
      imageSizes: selectedValues('#imageSizes')
    };
    localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(values));
  } catch (e) {
    // Private browsing or a disabled storage API should not break the form.
  }
}


function inlineMarkdown(source='') {
  let text=escapeHtml(source);
  text=text.replace(/`([^`]+)`/g,'<code>$1</code>');
  text=text.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  text=text.replace(/\*([^*]+)\*/g,'<em>$1</em>');
  text=text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return text;
}
function renderMarkdown(source='') {
  const lines=String(source).replace(/\r/g,'').split('\n'); let html='', paragraph=[], list=null, quote=[];
  const flushParagraph=()=>{if(paragraph.length){html+=`<p>${paragraph.map(inlineMarkdown).join('<br>')}</p>`;paragraph=[]}};
  const flushList=()=>{if(list){html+=`<${list.type}>${list.items.map(x=>`<li>${inlineMarkdown(x)}</li>`).join('')}</${list.type}>`;list=null}};
  const flushQuote=()=>{if(quote.length){html+=`<blockquote>${quote.map(inlineMarkdown).join('<br>')}</blockquote>`;quote=[]}};
  for(let i=0;i<lines.length;i++){
    const line=lines[i];
    if(line.startsWith('```')){flushParagraph();flushList();flushQuote();const lang=line.slice(3).trim();const code=[];while(++i<lines.length&&!lines[i].startsWith('```'))code.push(lines[i]);html+=`<pre data-lang="${escapeHtml(lang)}"><code>${escapeHtml(code.join('\n'))}</code></pre>`;continue}
    if(/^\s*\|/.test(line)&&i+1<lines.length&&/^\s*\|?\s*:?-+/.test(lines[i+1])){flushParagraph();flushList();flushQuote();const table=[];table.push(line);i++;while(++i<lines.length&&/^\s*\|/.test(lines[i]))table.push(lines[i]);i--;const cells=row=>row.trim().replace(/^\||\|$/g,'').split('|').map(x=>x.trim());const head=cells(table[0]);const body=table.slice(1).map(cells);html+='<div class="md-table-wrap"><table><thead><tr>'+head.map(x=>`<th>${inlineMarkdown(x)}</th>`).join('')+'</tr></thead><tbody>'+body.map(row=>'<tr>'+row.map(x=>`<td>${inlineMarkdown(x)}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';continue}
    const heading=line.match(/^(#{1,4})\s+(.+)$/);if(heading){flushParagraph();flushList();flushQuote();const n=heading[1].length;html+=`<h${n}>${inlineMarkdown(heading[2])}</h${n}>`;continue}
    if(/^\s*---+\s*$/.test(line)){flushParagraph();flushList();flushQuote();html+='<hr>';continue}
    const bullet=line.match(/^\s*[-*]\s+(.+)$/),numbered=line.match(/^\s*\d+[.)]\s+(.+)$/);
    if(bullet||numbered){flushParagraph();flushQuote();const type=bullet?'ul':'ol';if(list&&list.type!==type)flushList();list=list||{type,items:[]};list.items.push((bullet||numbered)[1]);continue}else flushList();
    const q=line.match(/^>\s?(.*)$/);if(q){flushParagraph();quote.push(q[1]);continue}else flushQuote();
    if(!line.trim()){flushParagraph();continue}paragraph.push(line.trim());
  }
  flushParagraph();flushList();flushQuote();return html;
}

async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function toast(message) { const el=$('#toast'); el.textContent=message; el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),2600); }
function caseLabel(id) { return state.cases.find(item=>item.id===id)?.title || id; }
function kindLabel(kind){return kind==='open-source'?'开源测试集':kind==='kvv'?'KVV':kind==='deepseek'?'DeepSeek V4':kind==='glm'?'GLM 5.3':kind==='gpt-image'?'GPT 图片压测':kind==='glm_stream_retest'?'GLM 流式复测':kind==='awsb'?'AWSB / Opus 5':kind==='moderation'?'内容审核':kind==='authenticity'?'满血真实性校验':kind==='arxivmath'?'ArXivMath':kind==='deepswe'?'DeepSWE':kind==='swe-atlas-tw'?'SWE-Atlas TW':kind==='swe-atlas-rf'?'SWE-Atlas RF':kind==='legal-agent-benchmark'?'Legal Agent Benchmark':kind==='temporary'?'临时测试案例集':kind==='strict'?'多轮压测（严格）':'多轮压测'}


function renderProtocols() {
  const select=$('#protocolSelect'), providers=[...new Set(state.protocols.map(item=>item.provider))];
  select.innerHTML=providers.map(provider=>`<optgroup label="${escapeHtml(provider)}">${state.protocols.filter(item=>item.provider===provider).map(item=>`<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)} · ${escapeHtml(item.path_template)}</option>`).join('')}</optgroup>`).join('');
  select.value=state.protocol;
  select.onchange=()=>selectProtocol(select.value);
}
function protocolSupports(item, protocolId=state.protocol) {
  const protocol=state.protocols.find(x=>x.id===protocolId);
  return Boolean(protocol && item.protocols.includes(protocol.family) && (!protocol.test_kinds.length || protocol.test_kinds.includes(item.test_kind)));
}
function updateEndpointPreview() {
  const item=state.protocols.find(x=>x.id===state.protocol), preview=$('#endpointPreview');
  if(!item)return;
  const base=$('#baseUrl').value.trim().replace(/\/+$/,'');
  const model=$('#model').value.trim();
  const path=item.path_template.replace('{model}',model?encodeURIComponent(model):'{model}');
  preview.textContent=base?`请求地址：${base}${path}`:`路径将自动拼装：${path}`;
}
function selectProtocol(id) {
  const previous=state.protocols.find(x=>x.id===state.protocol); const item=state.protocols.find(x=>x.id===id); if(!item)return; state.protocol=id; renderProtocols();
  const base=$('#baseUrl'); if(!base.value || (previous && base.value===previous.default_base_url)) base.value=item.default_base_url;
  $('#protocolDescription').textContent=`${item.provider} · ${item.description}`; updateEndpointPreview();
  setModelPlaceholder(item); renderCases(); state.selected=state.selected.filter(caseId=>{const selected=state.cases.find(x=>x.id===caseId);return selected&&protocolSupports(selected,id)}); if(!state.selected.length){const first=state.cases.find(x=>(state.testKind==='open-source'?['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(x.test_kind):x.test_kind===state.testKind)&&protocolSupports(x,id));if(first)state.selected=[first.id];} renderCases();updateSelectionUI(); state.models=[];hideModelSuggestions(); $('#modelHint').textContent='可直接输入；查询后支持按名称或 ID 模糊匹配'; $('#modelHint').className='field-hint'; persistForm();
}

function setModelPlaceholder(protocol){
  const item=protocol || state.protocols.find(x=>x.id===state.protocol) || {};
  $('#model').placeholder=state.testKind==='deepseek'?'输入任意 DeepSeek 模型名或网关模型 ID':
    state.testKind==='glm'?'输入 GLM 5.3 模型名或网关模型 ID':
    state.testKind==='gpt'?'输入 GPT 模型名或网关模型 ID':
    state.testKind==='gpt-image'?'图片模型/接入点 ID（请求模型按下方选项随机）':
    state.testKind==='authenticity'?'输入 DeepSeek 或 Kimi K3 模型名/网关模型 ID':
    state.testKind==='moderation'?'输入审核模型或网关模型 ID':
    state.testKind==='open-source'?'输入用于运行开源测试集的模型或网关模型 ID':
    state.testKind==='awsb'?'输入 claude-opus-5 或网关模型别名':item.model_hint||'';
}

function caseFacts(item){
  const facts=[['测试案例数',item.case_count_text || (item.case_count != null ? `${item.case_count} 条` : null)],['并发 / 速率',item.concurrency_text],['输入 Token',item.input_tokens],['输出 Token 上限',item.output_tokens],['总请求数',item.request_text]];
  if(item.warmup)facts.push(['预热时间',item.warmup]);
  if(item.steady)facts.push(['稳定运行',item.steady]);
  if(item.workload_profile)facts.push(['负载画像',item.workload_profile]);
  return facts;
}
function renderCases() {
  const cases=state.cases.filter(item=>state.testKind==='open-source'?['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(item.test_kind):item.test_kind===state.testKind);
  $('#caseGrid').innerHTML = cases.map(item => {const supported=protocolSupports(item);const tempOptions=item.test_kind==='temporary'?(item.case_options||[]).map(option=>`<option value="${escapeHtml(option.id)}" ${state.temporaryCase===option.id?'selected':''}>${escapeHtml(option.id)} · ${escapeHtml(option.name)}${option.merged?`（${option.repetitions} 次）`:''}</option>`).join(''):'';return `<div class="case-item"><button type="button" class="case-card ${state.selected.includes(item.id)?'selected':''} ${supported?'':'unsupported'}" data-case="${item.id}" ${supported?'':'disabled'}>
    <div class="case-head"><b>${item.id}</b><span class="cost">${supported?'成本 '+escapeHtml(item.cost):'当前协议不支持'}</span></div><div class="case-source">${escapeHtml(item.source||'专项自研代码')} · ${escapeHtml(item.executor||'')}</div><h3>${escapeHtml(item.title)}</h3>
    <dl class="case-facts">${caseFacts(item).map(([label,value])=>`<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value||'—')}</dd></div>`).join('')}</dl>
    <div class="admission-standard"><strong>准入标准</strong><p>${escapeHtml(item.admission||'以报告中的全部检查项通过为准。')}</p></div>
  </button>
  ${item.test_kind==='temporary'?`<label class="temporary-case-picker"><span>执行范围（同类项已合并）</span><select id="temporaryCaseSelect"><option value="">全部 ${item.case_count||''} 个合并案例</option>${tempOptions}</select></label>`:''}</div>`}).join('');
  document.querySelectorAll('.case-card:not(:disabled)').forEach(el => el.onclick = () => selectCase(el.dataset.case));
  const picker=$('#temporaryCaseSelect'); if(picker) picker.onchange=event=>{state.temporaryCase=event.target.value;persistForm()};
}
function selectedValues(selector){
  return [...document.querySelector(selector).selectedOptions].map(option=>option.value);
}
function renderImageConfig(){
  const box=$('#imageConfig'), item=state.cases.find(x=>state.selected.includes(x.id));
  if(!box) return;
  const visible=state.testKind==='gpt-image' && item && !item.fixed_parameters;
  box.style.display=visible?'flex':'none';
  if(!visible) return;
  const renderSelect=(id, values, selected)=>{const el=$(id);el.innerHTML=(values||[]).map(value=>`<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('');[...el.options].forEach(option=>option.selected=selected.includes(option.value));};
  const quality=$('#imageQuality'), models=$('#imageModels'), sizes=$('#imageSizes');
  const previousQuality=selectedValues('#imageQuality'), previousModels=selectedValues('#imageModels'), previousSizes=selectedValues('#imageSizes');
  renderSelect('#imageQuality',item.image_qualities,previousQuality.filter(value=>(item.image_qualities||[]).includes(value)).length?previousQuality:item.image_qualities||[]);
  renderSelect('#imageModels',item.image_models,previousModels.filter(value=>(item.image_models||[]).includes(value)).length?previousModels:item.image_models||[]);
  renderSelect('#imageSizes',item.image_sizes,previousSizes.filter(value=>(item.image_sizes||[]).includes(value)).length?previousSizes:item.image_sizes||[]);
  quality.title='按住 Command/Ctrl 可多选';models.title=quality.title;sizes.title=quality.title;
}
function updateCustomDefaults(){
  const item=state.cases.find(x=>x.id===state.selected[0]); const box=$('#customLoad');
  if(!box) return; const fixedLoad=Boolean(item?.fixed_parameters); const manualLoad=Boolean(item?.manual_parameters); const enabled=item && ((state.testKind==='load' && !fixedLoad) || (state.testKind==='gpt-image' && !fixedLoad)); box.style.display=enabled?'flex':'none'; const moderationBox=$('#moderationConfig'); if(moderationBox) moderationBox.style.display=state.testKind==='moderation'?'flex':'none';
  const isImage=state.testKind==='gpt-image';
  const quickToggle=$('#quick').closest('.mode-toggle'); quickToggle.style.display=state.testKind==='load'&&!fixedLoad&&!manualLoad?'flex':'none'; if(fixedLoad||manualLoad)$('#quick').checked=false;
  $('#customEnabled').disabled=isImage||manualLoad;
  if(isImage||manualLoad){ $('#customEnabled').checked=true; $('#customEnabled').closest('.custom-toggle').querySelector('span').textContent=isImage?'图片压测参数':'随机压测参数'; box.querySelector('small').textContent=isImage?'图片压测严格使用下方手动输入的总请求数和并发数。':'随机输入压测严格使用下方手动输入的总请求数和并发数。'; }
  else { $('#customEnabled').checked=state.testKind==='load'?$('#customEnabled').checked:false; $('#customEnabled').closest('.custom-toggle').querySelector('span').textContent='启用自定义性能参数'; box.querySelector('small').textContent='关闭时使用测试项原有默认配置；开启后应用下方参数'; }
  if(enabled && !$('#customRequests').dataset.edited){ $('#customRequests').value=item.requests||1; $('#customConcurrency').value=item.concurrency||1; }
  if(enabled){ $('#customRequests').disabled=!$('#customEnabled').checked; $('#customConcurrency').disabled=!$('#customEnabled').checked; }
  renderImageConfig();
}
function updateSelectionUI(){
  updateCustomDefaults();
  const selected=state.cases.filter(x=>state.selected.includes(x.id));
  $('#selectedName').textContent=selected.length?selected.map(x=>x.id).join(' + '):'请选择测试';
  $('#selectedMeta').textContent=selected.length===1?`${caseFacts(selected[0]).map(x=>x.join(' ')).join(' · ')} · 成本 ${selected[0].cost}${selected[0].display_only?' · 仅展示':''}`:selected.length?`已选择 ${selected.length} 项 · 标称请求共 ${selected.reduce((n,x)=>n+Number(x.requests||0),0)} 个；将串行执行并生成聚合报告`:'—';
}
function selectCase(id) {
  const item=state.cases.find(x=>x.id===id);
  const hasFixedLoad=state.selected.some(selectedId=>state.cases.find(x=>x.id===selectedId)?.fixed_parameters);
  const hasManualLoad=state.selected.some(selectedId=>state.cases.find(x=>x.id===selectedId)?.manual_parameters);
  if(state.testKind==='open-source' || state.testKind==='gpt-image' || item?.fixed_parameters || item?.manual_parameters || hasFixedLoad || hasManualLoad) state.selected=[id];
  else state.selected=state.selected.includes(id)?state.selected.filter(x=>x!==id):[...state.selected,id];
  if(state.selected.length!==1){$('#customRequests').dataset.edited='';$('#customConcurrency').dataset.edited='';}
  renderCases();updateSelectionUI(); if(state.testKind==='gpt-image') renderImageConfig();
}
function formatDuration(run) {
  if (!run.started_at) return '—'; const end=run.finished_at ? new Date(run.finished_at) : new Date();
  const sec=Math.max(0,Math.floor((end-new Date(run.started_at))/1000));
  return sec>=60 ? `${Math.floor(sec/60)}m ${sec%60}s` : `${sec}s`;
}
function statusLabel(status){return ({queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',interrupted:'已中断'})[status]||status}
function renderRuns() {
  const visible=state.runs.filter(run=> state.filter==='all' || (state.filter==='running' ? ['queued','running'].includes(run.status) : state.filter==='failed' ? ['failed','interrupted'].includes(run.status) : run.status===state.filter));
  $('#runRows').innerHTML = visible.length ? visible.map(run => `<tr>
    <td class="select-col"><input type="checkbox" class="download-check" data-run-id="${run.id}" ${state.downloadSelected.has(run.id)?'checked':''} ${run.download_url?'':'disabled'} aria-label="选择下载 ${escapeHtml(run.case_name)}"></td><td class="run-title"><b><span class="kind-label">${kindLabel(run.test_kind)}</span> ${escapeHtml(run.is_batch?'批量聚合':run.case_name)} · ${escapeHtml(run.is_batch?run.case_ids.join(' + '):caseLabel(run.case_name))}${run.quick?' [快速]':''}</b><small>${escapeHtml(run.created_at)}</small><button class="record-id" title="点击复制记录 ID" onclick="copyRecordId('${run.id}')">ID ${run.id}</button></td>
    <td class="endpoint"><b title="${escapeHtml(run.model)}">${escapeHtml(run.model)}</b><small>${escapeHtml((state.protocols.find(p=>p.id===run.protocol)?`${state.protocols.find(p=>p.id===run.protocol).provider} ${state.protocols.find(p=>p.id===run.protocol).name}`:run.protocol)+' · '+run.base_url)}</small></td>
    <td><span class="badge ${run.status}">${statusLabel(run.status)}</span></td>
    <td>${run.passed===true?'<b class="pass">通过</b>':run.passed===false?'<b class="fail">未通过</b>':'<span class="pending">待判定</span>'}</td>
    <td data-start="${run.id}">${formatDuration(run)}</td>
    <td><div class="actions">${run.report_url?`<button class="link-btn" onclick="openArtifact('${run.id}','report')">报告</button>`:''}${run.summary_url?`<button class="link-btn" onclick="openArtifact('${run.id}','summary')">指标</button>`:''}${run.manifest_url?`<button class="link-btn" onclick="openArtifact('${run.id}','manifest')">聚合数据</button>`:''}${run.requests_url?`<button class="link-btn" onclick="openArtifact('${run.id}','requests')">请求明细</button>`:''}${run.download_url?`<a class="link-btn download-btn" href="${run.download_url}" download>下载全部</a>`:''}${run.log_url?`<button class="link-btn" onclick="openArtifact('${run.id}','log')">日志</button>`:''}${['queued','running'].includes(run.status)?`<button class="link-btn danger" onclick="cancelRun('${run.id}')">${run.status==='queued'?'取消':'停止'}</button>`:''}${!run.report_url&&!run.log_url&&!['queued','running'].includes(run.status)?'—':''}</div></td>
  </tr>`).join('') : '<tr><td colspan="7" class="empty">暂无匹配的测试记录</td></tr>';
  document.querySelectorAll('.download-check').forEach(el=>el.onchange=()=>{if(el.checked)state.downloadSelected.add(el.dataset.runId);else state.downloadSelected.delete(el.dataset.runId);updateDownloadUI()});
  updateDownloadUI();
}
function updateDownloadUI(){
  state.downloadSelected = new Set([...state.downloadSelected].filter(id=>state.runs.some(r=>r.id===id && r.download_url)));
  const button=$('#downloadSelected'), count=state.downloadSelected.size;
  button.disabled=!count; button.textContent=`下载选中（${count}）`;
}
$('#selectAllDownloads').onclick=()=>{
  const downloadable=state.runs.filter(r=>r.download_url).map(r=>r.id);
  const all=downloadable.length && downloadable.every(id=>state.downloadSelected.has(id));
  downloadable.forEach(id=>all?state.downloadSelected.delete(id):state.downloadSelected.add(id));
  renderRuns();
};
$('#downloadSelected').onclick=async()=>{
  const ids=[...state.downloadSelected]; if(!ids.length)return;
  const button=$('#downloadSelected'); button.disabled=true; button.textContent='打包中…';
  try{
    const response=await fetch('/api/runs/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_ids:ids})});
    if(!response.ok){let data={};try{data=await response.json()}catch{};throw new Error(data.error||'打包下载失败')}
    const blob=await response.blob(), url=URL.createObjectURL(blob), link=document.createElement('a');
    link.href=url; link.download=`test-data-selected-${new Date().toISOString().slice(0,10)}.zip`; link.click(); URL.revokeObjectURL(url); toast(`已打包下载 ${ids.length} 条测试记录`);
  }catch(e){toast(e.message)}finally{updateDownloadUI()}
};

async function refreshRuns(silent=true) {
  try { state.runs=await api('/api/runs'); renderRuns(); }
  catch(e) { if(!silent) toast(e.message); }
}


window.cancelRun = async id => {
  const run=state.runs.find(x=>x.id===id); if(!run) return;
  const action=run.status==='queued'?'取消排队':'停止测试';
  if(!confirm(`确认${action}？已执行的请求会保留，并生成中断报告。`)) return;
  try { await api(`/api/runs/${id}/cancel`,{method:'POST',body:'{}'}); toast(`${action}请求已提交`); await refreshRuns(false); }
  catch(e) { toast(e.message); }
};

window.copyRecordId = async id => {
  try { await navigator.clipboard.writeText(id); toast(`记录 ID 已复制：${id}`); }
  catch(e) { toast(`记录 ID：${id}`); }
};

window.openArtifact = async (id, kind) => {
  try {
    const response=await fetch(`/api/runs/${id}/${kind}`); if(!response.ok) throw new Error('文件尚未生成');
    const text=await response.text(); const run=state.runs.find(r=>r.id===id);
    $('#dialogEyebrow').textContent=kind.toUpperCase(); $('#dialogTitle').textContent=`${run.is_batch?'批量聚合':run.case_name} · ${kind==='report'?'测试报告':kind==='summary'?'指标明细':kind==='requests'?'请求明细':'执行日志'}`;
    if(kind==='report'){$('#reportBody').className='report-body markdown-body';$('#reportBody').innerHTML=renderMarkdown(text)}else{$('#reportBody').className='report-body code-body';$('#reportBody').textContent=text} $('#reportDialog').showModal();
  } catch(e){toast(e.message)}
};

$('#runForm').addEventListener('submit', async event => {
  event.preventDefault(); const message=$('#formMessage'); message.textContent='';
  if(!state.selected.length){message.textContent='请先选择一个测试场景';return}
  const quick=$('#quick').checked;
  const selectedItem=state.cases.find(x=>x.id===state.selected[0]);
  const effectiveTestKind=state.testKind==='open-source'?(selectedItem?.test_kind||''):state.testKind;
  const isImage=effectiveTestKind==='gpt-image';
  if(effectiveTestKind==='load' && !quick && state.selected.some(id=>['C1','SIMPLE_LOAD','FULL_LOAD'].includes(id)) && !confirm(`${state.selected.join('、')} 包含高成本测试，会产生较高 Token 成本，确认立即排队执行？`)) return;
  if((isImage && !selectedItem?.fixed_parameters || selectedItem?.manual_parameters) && (!Number($('#customRequests').value) || !Number($('#customConcurrency').value))){message.textContent='请填写总请求数和并发数';return}
  const button=$('#submitBtn'); button.disabled=true;
  try {
    const run=await api('/api/runs',{method:'POST',body:JSON.stringify({protocol:state.protocol,test_kind:effectiveTestKind,base_url:$('#baseUrl').value.trim(),api_key:$('#apiKey').value.trim(),model:$('#model').value.trim(),cases:state.selected,temporary_case:effectiveTestKind==='temporary'?state.temporaryCase:null,quick,requests:(effectiveTestKind==='load'||isImage)&&$('#customEnabled').checked?Number($('#customRequests').value):null,concurrency:effectiveTestKind==='moderation'?Number($('#moderationConcurrency').value):((state.testKind==='load'||isImage)&&$('#customEnabled').checked?Number($('#customConcurrency').value):null),image_quality:isImage?selectedValues('#imageQuality'):null,image_models:isImage?selectedValues('#imageModels'):null,image_sizes:isImage?selectedValues('#imageSizes'):null})});
    state.runs.unshift(run); renderRuns(); toast(`${state.selected.join('、')} 已加入执行队列`);
    document.querySelector('.history').scrollIntoView({behavior:'smooth'});
  } catch(e){message.textContent=e.message} finally {button.disabled=false}
});
$('#connectivityBtn').onclick=async()=>{
  const button=$('#connectivityBtn'), result=$('#connectivityResult');
  const base_url=$('#baseUrl').value.trim(), api_key=$('#apiKey').value.trim(), model=$('#model').value.trim();
  result.className='connectivity-result show';
  if(!base_url || !api_key || !model){result.textContent='请先填写 Base URL、API Key 和 Model';result.classList.add('error');return}
  button.disabled=true;button.querySelector('span').textContent='测试中…';result.textContent='正在发送一个小请求检查连接…';
  try{
    const response=await fetch('/api/connectivity/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({protocol:state.protocol,base_url,api_key,model})});
    const data=await response.json();
    if(!response.ok || !data.passed){
      const details=[data.error||'连通性测试失败',data.diagnosis,data.endpoint&&`请求地址：${data.endpoint}`,data.provider_request_id&&`请求 ID：${data.provider_request_id}`].filter(Boolean);
      throw new Error(details.join('；'));
    }
    result.textContent=`连接成功 · HTTP ${data.http_status} · ${Number(data.duration_ms).toFixed(0)} ms`;
    result.classList.add('success');toast('连通性测试通过');
  }catch(e){result.textContent=`连接失败：${e.message}`;result.classList.add('error')}
  finally{button.disabled=false;button.querySelector('span').textContent='测试连通性'}
};

$('#queryModels').onclick=async()=>{
  const button=$('#queryModels'), hint=$('#modelHint'); hint.className='field-hint';
  const base_url=$('#baseUrl').value.trim(), api_key=$('#apiKey').value.trim();
  if(!base_url || !api_key){hint.textContent='请先填写 Base URL 和 API Key';hint.classList.add('error');return}
  button.disabled=true; button.textContent='查询中…'; hint.textContent='正在请求模型列表…';
  try{
    const result=await api('/api/models/query',{method:'POST',body:JSON.stringify({protocol:state.protocol,base_url,api_key})});
    state.models=result.models||[];
    hint.textContent=result.count?`已查询到 ${result.count} 个模型；输入名称或 ID 可模糊筛选`:'接口返回成功，但未发现可用于生成内容的模型';
    hint.classList.add(result.count?'success':'error');
    if(result.count===1)$('#model').value=result.models[0].id;
    renderModelSuggestions($('#model').value);
  }catch(e){hint.textContent=e.message;hint.classList.add('error')}finally{button.disabled=false;button.textContent='查询模型'}
};
function fuzzyModelMatch(model, query){
  const haystack=`${model.id} ${model.name}`.toLowerCase(), needle=query.trim().toLowerCase();
  if(!needle)return true;if(haystack.includes(needle))return true;
  let index=0;for(const char of haystack){if(char===needle[index])index++;if(index===needle.length)return true}return false;
}
function hideModelSuggestions(){const box=$('#modelSuggestions');box.classList.remove('show');box.innerHTML='';$('#model').setAttribute('aria-expanded','false')}
function selectSuggestedModel(id){$('#model').value=id;hideModelSuggestions();$('#model').focus();persistForm()}
function renderModelSuggestions(query=''){
  const box=$('#modelSuggestions');if(!state.models.length){hideModelSuggestions();return}
  const matches=state.models.filter(model=>fuzzyModelMatch(model,query)).slice(0,50);
  box.innerHTML=matches.length?matches.map(model=>`<button type="button" role="option" data-model-id="${escapeHtml(model.id)}"><b>${escapeHtml(model.name)}</b><small>${escapeHtml(model.id)}</small></button>`).join(''):'<div class="no-model-match">没有匹配模型，可继续使用手工输入值</div>';
  box.classList.add('show');$('#model').setAttribute('aria-expanded','true');
  box.querySelectorAll('[data-model-id]').forEach(button=>button.onmousedown=event=>{event.preventDefault();selectSuggestedModel(button.dataset.modelId)});
}
$('#model').oninput=()=>{renderModelSuggestions($('#model').value);updateEndpointPreview();persistForm()};
$('#apiKey').oninput=()=>persistForm();
$('#baseUrl').oninput=()=>{updateEndpointPreview();persistForm()};
$('#baseUrl').onblur=()=>{const input=$('#baseUrl');input.value=input.value.trim().replace(/\/+$/,'');updateEndpointPreview();persistForm()};
$('#model').onfocus=()=>{if(state.models.length)renderModelSuggestions($('#model').value)};
$('#model').onkeydown=event=>{if(event.key==='Escape')hideModelSuggestions()};
document.addEventListener('mousedown',event=>{if(!event.target.closest('.model-combobox'))hideModelSuggestions()});
$('#toggleKey').onclick=()=>{const input=$('#apiKey');const masked=input.classList.toggle('secret-masked');$('#toggleKey').textContent=masked?'显示':'隐藏'};
$('#customRequests').oninput=()=>{$('#customRequests').dataset.edited='1';persistForm()}; $('#customConcurrency').oninput=()=>{$('#customConcurrency').dataset.edited='1';persistForm()};
$('#customEnabled').onchange=()=>{ $('#customRequests').disabled=!$('#customEnabled').checked; $('#customConcurrency').disabled=!$('#customEnabled').checked; persistForm(); };
$('#resetCustom').onclick=()=>{delete $('#customRequests').dataset.edited; delete $('#customConcurrency').dataset.edited; updateCustomDefaults();persistForm()};
$('#refreshBtn').onclick=()=>refreshRuns(false);
$('#closeDialog').onclick=()=>$('#reportDialog').close();
$('#reportDialog').addEventListener('click',e=>{if(e.target===$('#reportDialog')) $('#reportDialog').close()});
document.querySelectorAll('#runForm input, #runForm select').forEach(input=>input.addEventListener('change',persistForm));
document.querySelectorAll('.kind-tab').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.kind-tab').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.testKind=button.dataset.kind;state.selected=[];if(state.testKind!=='temporary')state.temporaryCase=''; delete $('#customRequests').dataset.edited; delete $('#customConcurrency').dataset.edited; $('#customEnabled').checked=false;
  $('#quick').closest('.mode-toggle').style.display=state.testKind==='load'?'flex':'none';
  setModelPlaceholder();
  renderCases();
  let first=state.cases.find(x=>(state.testKind==='open-source'?['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(x.test_kind):x.test_kind===state.testKind)&&protocolSupports(x));
  if(!first){const matchingProtocol=state.protocols.find(protocol=>state.cases.some(x=>(state.testKind==='open-source'?['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(x.test_kind):x.test_kind===state.testKind)&&protocolSupports(x,protocol.id)));if(matchingProtocol){selectProtocol(matchingProtocol.id);first=state.cases.find(x=>(state.testKind==='open-source'?['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(x.test_kind):x.test_kind===state.testKind)&&protocolSupports(x));}}
  if(first)state.selected=[first.id];renderCases();updateSelectionUI();updateEndpointPreview();
  if(state.testKind==='gpt-image' && first && !$('#model').value) $('#model').value=first.image_models?.[0]||first.title;
  persistForm();
});
document.querySelectorAll('.filter').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.filter').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.filter=button.dataset.filter;renderRuns();
});

(async function init(){
  try {
    const saved=readPersistedForm();
    state.testKind=['load','kvv','deepseek','glm','gpt-image','awsb','open-source','temporary','moderation'].includes(saved.testKind)?saved.testKind:'load';
    document.querySelectorAll('.kind-tab').forEach(button=>button.classList.toggle('active',button.dataset.kind===state.testKind));
    $('#quick').closest('.mode-toggle').style.display=state.testKind==='load'?'flex':'none';
    state.protocols=await api('/api/protocols');
    state.protocol=state.protocols.some(item=>item.id===saved.protocol)?saved.protocol:'openai';
    renderProtocols(); selectProtocol(state.protocol);
    state.cases=await api('/api/cases');
    if(saved.baseUrl!=null)$('#baseUrl').value=String(saved.baseUrl);
    if(saved.apiKey!=null)$('#apiKey').value=String(saved.apiKey);
    if(saved.model!=null)$('#model').value=String(saved.model);
    if(saved.temporaryCase!=null)state.temporaryCase=String(saved.temporaryCase);
    if(typeof saved.quick==='boolean')$('#quick').checked=saved.quick;
    if(typeof saved.customEnabled==='boolean')$('#customEnabled').checked=saved.customEnabled;
    if(saved.customRequests!=null){$('#customRequests').value=String(saved.customRequests);$('#customRequests').dataset.edited='1'}
    if(saved.customConcurrency!=null){$('#customConcurrency').value=String(saved.customConcurrency);$('#customConcurrency').dataset.edited='1'}
    if(saved.moderationConcurrency!=null)$('#moderationConcurrency').value=String(saved.moderationConcurrency);
    state.selected=Array.isArray(saved.selected)?saved.selected.filter(id=>state.cases.some(item=>item.id===id&&protocolSupports(item))):[];
    if(!state.selected.length){const first=state.cases.find(item=>(state.testKind==='open-source'?['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(item.test_kind):item.test_kind===state.testKind)&&protocolSupports(item));if(first)state.selected=[first.id];}
    renderCases(); updateSelectionUI();
    [['#imageQuality',saved.imageQuality],['#imageModels',saved.imageModels],['#imageSizes',saved.imageSizes]].forEach(([selector,values])=>{
      if(!Array.isArray(values))return;
      const element=$(selector); if(!element)return;
      [...element.options].forEach(option=>option.selected=values.includes(option.value));
    });
    updateEndpointPreview(); persistForm(); await refreshRuns();
  }
  catch(e){$('#serviceState').innerHTML='<i style="background:#fb7185"></i>服务异常';toast(e.message)}
  setInterval(()=>refreshRuns(),3000); setInterval(()=>{state.runs.filter(r=>['queued','running'].includes(r.status)).forEach(r=>{const el=document.querySelector(`[data-start="${r.id}"]`);if(el)el.textContent=formatDuration(r)})},1000);
})();
