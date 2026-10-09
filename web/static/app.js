const state = { protocols: [], protocol: 'openai', testKind: 'load', cases: [], selected: [], temporaryCase: '', billingProbe: '', claudeVariant: 'fable5', runs: [], runsTotal: 0, runsPage: 1, runsPageSize: 20, filter: 'all', conclusion: 'all', runsSearch: '', models: [], downloadSelected: new Set() };
const CLAUDE_VARIANT_MODELS = { fable5: 'claude-fable-5', opus5: 'claude-opus-5', sonnet5: 'claude-sonnet-5', haiku45: 'claude-haiku-4-5', sonnet46: 'claude-sonnet-4-6' };
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
      billingProbe: state.billingProbe,
      claudeVariant: state.claudeVariant,
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
      imageSizes: selectedValues('#imageSizes'),
      geminiQuality: selectedValues('#geminiQuality'),
      geminiModels: selectedValues('#geminiModels'),
      geminiAspects: selectedValues('#geminiAspects'),
      geminiCounts: selectedValues('#geminiCounts')
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
function kindLabel(kind){return kind==='open-source'?'开源测试集':kind==='kvv'?'KVV':kind==='deepseek'?'DeepSeek V4':kind==='glm'?'GLM 5.3':kind==='gpt-image'?'GPT 图片压测':kind==='fable'?'Claude 测试':kind==='gemini-image'?'Gemini 图片测试':kind==='glm_stream_retest'?'GLM 流式复测':kind==='awsb'?'AWSB / Opus 5':kind==='moderation'?'内容审核':kind==='authenticity'?'满血真实性校验':kind==='arxivmath'?'ArXivMath':kind==='deepswe'?'DeepSWE':kind==='swe-atlas-tw'?'SWE-Atlas TW':kind==='swe-atlas-rf'?'SWE-Atlas RF':kind==='legal-agent-benchmark'?'Legal Agent Benchmark':kind==='temporary'?'临时测试案例集':kind==='strict'?'多轮压测（严格）':'多轮压测'}
function kindMatches(item){
  if(state.testKind==='open-source')return ['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(item.test_kind);
  // always_visible：Claude 模块下常驻的案例（如中转计费探针），不随模型变体切换。
  if(state.testKind==='fable')return item.test_kind==='fable'&&(item.always_visible||(item.claude_variant||'fable5')===state.claudeVariant);
  return item.test_kind===state.testKind;
}


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
  setModelPlaceholder(item); renderCases(); state.selected=state.selected.filter(caseId=>{const selected=state.cases.find(x=>x.id===caseId);return selected&&protocolSupports(selected,id)}); if(!state.selected.length){const first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x,id));if(first)state.selected=[first.id];} renderCases();updateSelectionUI(); state.models=[];hideModelSuggestions(); $('#modelHint').textContent='可直接输入；查询后支持按名称或 ID 模糊匹配'; $('#modelHint').className='field-hint'; persistForm();
}

function setModelPlaceholder(protocol){
  const item=protocol || state.protocols.find(x=>x.id===state.protocol) || {};
  $('#model').placeholder=state.testKind==='deepseek'?'输入任意 DeepSeek 模型名或网关模型 ID':
    state.testKind==='glm'?'输入 GLM 5.3 模型名或网关模型 ID':
    state.testKind==='gpt'?'输入 GPT 模型名或网关模型 ID':
    state.testKind==='fable'?`输入 ${CLAUDE_VARIANT_MODELS[state.claudeVariant]} 或网关模型别名`:
    state.testKind==='gpt-image'?'图片模型/接入点 ID（请求模型按下方选项随机）':
    state.testKind==='gemini-image'?'Gemini 图像模型/接入点 ID（请求模型按下方选项随机）':
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
function caseScopeSelection(){
  // 子项选择的存放位置按模块区分：K3 临时案例集 vs Claude 模块下的计费探针。
  return state.testKind==='temporary' ? state.temporaryCase : state.billingProbe;
}
function caseScopeOptions(item){
  const options=item.case_options||[];
  const current=caseScopeSelection();
  const lead=item.test_kind==='temporary'?'<option value="">全部 K3 合并案例</option>':'';
  const buckets=[];
  options.forEach(option=>{
    const label=option.group||'可选案例';
    let bucket=buckets.find(x=>x.label===label);
    if(!bucket){bucket={label,items:[]};buckets.push(bucket);}
    bucket.items.push(option);
  });
  // 非临时案例集（计费探针）默认选中整组入口，避免「选了卡片却没选范围」的歧义。
  const isSelected=option=>current?current===option.id:(item.test_kind!=='temporary'&&option.id===(options[0]||{}).id);
  return lead+buckets.map(bucket=>`<optgroup label="${escapeHtml(bucket.label)}">${bucket.items.map(option=>`<option value="${escapeHtml(option.id)}" ${isSelected(option)?'selected':''}>${escapeHtml(option.id)} · ${escapeHtml(option.name)}${option.merged?`（${option.repetitions} 次）`:''}</option>`).join('')}</optgroup>`).join('');
}
function renderCases() {
  const cases=state.cases.filter(item=>kindMatches(item));
  $('#caseGrid').innerHTML = cases.map(item => {const supported=protocolSupports(item);const scopeOptions=item.case_options?caseScopeOptions(item):'';return `<div class="case-item"><button type="button" class="case-card ${state.selected.includes(item.id)?'selected':''} ${supported?'':'unsupported'}" data-case="${item.id}" ${supported?'':'disabled'}>
    <div class="case-head"><b>${item.id}</b><span class="cost">${supported?'成本 '+escapeHtml(item.cost):'当前协议不支持'}</span></div><div class="case-source">${escapeHtml(item.source||'专项自研代码')} · ${escapeHtml(item.executor||'')}</div><h3>${escapeHtml(item.title)}</h3>
    <dl class="case-facts">${caseFacts(item).map(([label,value])=>`<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value||'—')}</dd></div>`).join('')}</dl>
    <div class="admission-standard"><strong>准入标准</strong><p>${escapeHtml(item.admission||'以报告中的全部检查项通过为准。')}</p></div>
  </button>
  ${item.case_options?`<label class="temporary-case-picker"><span>${escapeHtml(item.options_label||'执行范围')}</span><select class="case-scope-select">${scopeOptions}</select></label>`:''}</div>`}).join('');
  document.querySelectorAll('.case-card:not(:disabled)').forEach(el => el.onclick = () => selectCase(el.dataset.case));
  const picker=$('.case-scope-select');
  if(picker) picker.onchange=event=>{
    if(state.testKind==='temporary')state.temporaryCase=event.target.value;
    else state.billingProbe=event.target.value;
    persistForm();
  };
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
function renderGeminiImageConfig(){
  const box=$('#geminiImageConfig'), item=state.cases.find(x=>state.selected.includes(x.id));
  if(!box) return;
  const visible=state.testKind==='gemini-image' && item;
  box.style.display=visible?'flex':'none';
  if(!visible) return;
  const allowed=(values)=> (values||[]).map(value=>String(value));
  const previous=selector=>selectedValues(selector);
  const keep=(picked, pool)=> picked.filter(value=>pool.includes(value)).length ? picked.filter(value=>pool.includes(value)) : pool;
  const renderSelect=(selector, pool, selected)=>{
    const el=$(selector);
    el.innerHTML=pool.map(value=>`<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('');
    [...el.options].forEach(option=>option.selected=selected.includes(option.value));
    el.title='按住 Command/Ctrl 可多选';
  };
  const qualities=allowed(item.image_qualities), models=allowed(item.image_models), aspects=allowed(item.image_aspect_ratios), counts=allowed(item.image_counts);
  renderSelect('#geminiQuality',qualities,keep(previous('#geminiQuality'),qualities));
  renderSelect('#geminiModels',models,keep(previous('#geminiModels'),models));
  renderSelect('#geminiAspects',aspects,keep(previous('#geminiAspects'),aspects));
  renderSelect('#geminiCounts',counts,keep(previous('#geminiCounts'),counts));
}
function renderClaudeVariantConfig(){
  const box=$('#claudeVariantConfig');
  if(box) box.style.display=state.testKind==='fable'?'flex':'none';
  const select=$('#claudeVariantSelect');
  if(select) select.value=state.claudeVariant;
}
$('#claudeVariantSelect').onchange=event=>{
  state.claudeVariant=event.target.value in CLAUDE_VARIANT_MODELS?event.target.value:'fable5';
  state.selected=[];
  const modelInput=$('#model'), current=modelInput.value.trim();
  if(!current || Object.values(CLAUDE_VARIANT_MODELS).includes(current)) modelInput.value=CLAUDE_VARIANT_MODELS[state.claudeVariant];
  setModelPlaceholder(); updateEndpointPreview();
  const first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x));
  if(first)state.selected=[first.id];
  renderCases(); updateSelectionUI(); persistForm();
};
function updateCustomDefaults(){
  const item=state.cases.find(x=>x.id===state.selected[0]); const box=$('#customLoad');
  const isImage=state.testKind==='gpt-image'||state.testKind==='gemini-image';
  if(!box) return; const fixedLoad=Boolean(item?.fixed_parameters); const manualLoad=Boolean(item?.manual_parameters); const enabled=item && ((state.testKind==='load' && !fixedLoad) || (isImage && !fixedLoad)); box.style.display=enabled?'flex':'none'; const moderationBox=$('#moderationConfig'); if(moderationBox) moderationBox.style.display=state.testKind==='moderation'?'flex':'none';
  const quickToggle=$('#quick').closest('.mode-toggle'); quickToggle.style.display=state.testKind==='load'&&!fixedLoad&&!manualLoad?'flex':'none'; if(fixedLoad||manualLoad)$('#quick').checked=false;
  $('#customEnabled').disabled=isImage||manualLoad;
  if(isImage||manualLoad){ $('#customEnabled').checked=true; $('#customEnabled').closest('.custom-toggle').querySelector('span').textContent=isImage?'图片压测参数':'随机压测参数'; box.querySelector('small').textContent=isImage?'图片压测严格使用下方手动输入的总请求数和并发数。':'随机输入压测严格使用下方手动输入的总请求数和并发数。'; }
  else { $('#customEnabled').checked=state.testKind==='load'?$('#customEnabled').checked:false; $('#customEnabled').closest('.custom-toggle').querySelector('span').textContent='启用自定义性能参数'; box.querySelector('small').textContent='关闭时使用测试项原有默认配置；开启后应用下方参数'; }
  if(enabled && !$('#customRequests').dataset.edited){ $('#customRequests').value=item.requests||1; $('#customConcurrency').value=item.concurrency||1; }
  if(enabled){ $('#customRequests').disabled=!$('#customEnabled').checked; $('#customConcurrency').disabled=!$('#customEnabled').checked; }
  renderImageConfig();
  renderGeminiImageConfig();
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
  renderCases();updateSelectionUI(); if(state.testKind==='gpt-image') renderImageConfig(); if(state.testKind==='gemini-image') renderGeminiImageConfig();
}
function formatDuration(run) {
  if (!run.started_at) return '—'; const end=run.finished_at ? new Date(run.finished_at) : new Date();
  const sec=Math.max(0,Math.floor((end-new Date(run.started_at))/1000));
  return sec>=60 ? `${Math.floor(sec/60)}m ${sec%60}s` : `${sec}s`;
}
function statusLabel(status){return ({queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',interrupted:'已中断'})[status]||status}
function renderRuns() {
  const visible=state.runs;
  $('#runRows').innerHTML = visible.length ? visible.map(run => `<tr>
    <td class="select-col"><input type="checkbox" class="download-check" data-run-id="${run.id}" ${state.downloadSelected.has(run.id)?'checked':''} ${run.download_url?'':'disabled'} aria-label="选择下载 ${escapeHtml(run.case_name)}"></td><td class="run-title"><b><span class="kind-label">${kindLabel(run.test_kind)}</span> ${escapeHtml(run.is_batch?'批量聚合':run.case_name)} · ${escapeHtml(run.is_batch?run.case_ids.join(' + '):(run.case_title||caseLabel(run.case_name))) }${run.quick?' [快速]':''}</b><small>${escapeHtml(run.created_at)}</small><button class="record-id" title="点击复制记录 ID" onclick="copyRecordId('${run.id}')">ID ${run.id}</button></td>
    <td class="endpoint"><b title="${escapeHtml(run.model)}">${escapeHtml(run.model)}</b><small>${escapeHtml((state.protocols.find(p=>p.id===run.protocol)?`${state.protocols.find(p=>p.id===run.protocol).provider} ${state.protocols.find(p=>p.id===run.protocol).name}`:run.protocol)+' · '+run.base_url)}</small></td>
    <td><span class="badge ${run.status}">${statusLabel(run.status)}</span></td>
    <td>${run.passed===true?'<b class="pass">通过</b>':run.passed===false?'<b class="fail">未通过</b>':'<span class="pending">待判定</span>'}</td>
    <td data-start="${run.id}">${formatDuration(run)}</td>
    <td><div class="actions">${run.report_url?`<button class="link-btn" onclick="openArtifact('${run.id}','report')">报告</button>`:''}${run.summary_url?`<button class="link-btn" onclick="openArtifact('${run.id}','summary')">指标</button>`:''}${run.manifest_url?`<button class="link-btn" onclick="openArtifact('${run.id}','manifest')">聚合数据</button>`:''}${run.requests_url?`<button class="link-btn" onclick="openArtifact('${run.id}','requests')">请求明细</button>`:''}${run.download_url?`<a class="link-btn download-btn" href="${run.download_url}" download>下载全部</a>`:''}${run.log_url?`<button class="link-btn" onclick="openArtifact('${run.id}','log')">日志</button>`:''}${['queued','running'].includes(run.status)?`<button class="link-btn danger" onclick="cancelRun('${run.id}')">${run.status==='queued'?'取消':'停止'}</button>`:''}${!['queued','running'].includes(run.status)?`<button class="link-btn danger" onclick="deleteRun('${run.id}')">删除</button>`:''}</div></td>
  </tr>`).join('') : '<tr><td colspan="7" class="empty">暂无匹配的测试记录</td></tr>';
  document.querySelectorAll('.download-check').forEach(el=>el.onchange=()=>{if(el.checked)state.downloadSelected.add(el.dataset.runId);else state.downloadSelected.delete(el.dataset.runId);updateDownloadUI()});
  const pages=Math.max(1,Math.ceil(state.runsTotal/state.runsPageSize));
  $('#runsPageInfo').textContent=state.runsTotal?`${(state.runsPage-1)*state.runsPageSize+1}–${Math.min(state.runsPage*state.runsPageSize,state.runsTotal)} / ${state.runsTotal}`:'0 条记录';
  $('#runsPrev').disabled=state.runsPage<=1; $('#runsNext').disabled=state.runsPage>=pages;
  updateDownloadUI();
}
function updateDownloadUI(){
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
  try {
    const query=new URLSearchParams({page:String(state.runsPage),page_size:String(state.runsPageSize),status:state.filter,conclusion:state.conclusion,q:state.runsSearch});
    const data=await api(`/api/runs?${query}`); state.runs=data.items; state.runsTotal=data.total;
    const lastPage=Math.max(1,Math.ceil(state.runsTotal/state.runsPageSize));
    if(state.runsPage>lastPage){state.runsPage=lastPage;return refreshRuns(silent)}
    renderRuns();
  }
  catch(e) { if(!silent) toast(e.message); }
}


window.cancelRun = async id => {
  const run=state.runs.find(x=>x.id===id); if(!run) return;
  const action=run.status==='queued'?'取消排队':'停止测试';
  if(!confirm(`确认${action}？已执行的请求会保留，并生成中断报告。`)) return;
  try { await api(`/api/runs/${id}/cancel`,{method:'POST',body:'{}'}); toast(`${action}请求已提交`); await refreshRuns(false); }
  catch(e) { toast(e.message); }
};

window.deleteRun = async id => {
  const run=state.runs.find(item=>item.id===id); if(!run)return;
  const target=run.is_batch?'整个批次及其子记录和报告文件':'这条记录及其报告文件';
  if(!confirm(`确定删除${target}？此操作不可恢复。`))return;
  try{
    const response=await fetch(`/api/runs/${encodeURIComponent(id)}`,{method:'DELETE'});
    const data=await response.json(); if(!response.ok)throw new Error(data.error||'删除失败');
    state.downloadSelected.delete(id); toast(`已删除 ${data.deleted_records} 条记录`); await refreshRuns(false);
  }catch(e){toast(e.message)}
};

window.copyRecordId = async id => {
  try { await navigator.clipboard.writeText(id); toast(`记录 ID 已复制：${id}`); }
  catch(e) { toast(`记录 ID：${id}`); }
};

const CASE_STATUS_META = { passed:['passed','通过'], failed:['failed','失败'], error:['failed','错误'], warning:['warning','警告'], blocked:['blocked','阻塞'] };
function caseBadge(status) {
  const meta = CASE_STATUS_META[status] || ['unknown', status || '未知'];
  return `<span class="case-badge ${meta[0]}">${escapeHtml(meta[1])}</span>`;
}
function fmtMetricSeconds(v) { return typeof v==='number' && isFinite(v) ? `${v.toFixed(3)}s` : '—'; }
function fmtMetricRate(v) { return typeof v==='number' && isFinite(v) ? `${(v*100).toFixed(1)}%` : '—'; }
function batchSuiteTable(manifest) {
  const items = manifest && Array.isArray(manifest.results) ? manifest.results : [];
  if (!items.length) return '';
  const rows = items.map((item) => {
    const m = item.metrics || {}, s = item.summary || {};
    const cases = Array.isArray(s.results) ? s.results : [];
    const caseTotal = m.case_total != null ? m.case_total : (cases.length || null);
    const casePassed = m.case_passed != null ? m.case_passed : cases.filter(r=>r.status==='passed').length;
    const extras = (m.case_warning || m.case_blocked) ? ` <small class="case-extras">⚠${m.case_warning||0} · ⏸${m.case_blocked||0}</small>` : '';
    const caseText = caseTotal ? `${casePassed}/${caseTotal}${extras}` : '—';
    const detailRows = cases.length ? cases.map(r => `<tr>
        <td><code>${escapeHtml(r.case_id || '—')}</code></td>
        <td>${escapeHtml(r.name || '')}</td>
        <td>${caseBadge(r.status)}</td>
        <td class="num">${typeof r.duration_seconds==='number' ? r.duration_seconds.toFixed(2) : '—'}</td>
        <td class="case-detail-text">${escapeHtml(String(r.details || r.error || '—').slice(0, 240))}</td>
      </tr>`).join('') : `<tr><td colspan="5" class="detail-empty">${escapeHtml(item.error || '无用例级结果（子记录可能未生成 summary）')}</td></tr>`;
    return `<tr class="suite-row" aria-expanded="false" title="子记录 ${escapeHtml(item.record_id || '')}">
      <td><span class="chev">▸</span> <b>${escapeHtml(item.case || '')}</b></td>
      <td><span class="badge ${escapeHtml(item.status || '')}">${escapeHtml(statusLabel(item.status || ''))}</span></td>
      <td>${item.passed===true?'<b class="pass">通过</b>':item.passed===false?'<b class="fail">未通过</b>':'<span class="pending">待判定</span>'}</td>
      <td class="num">${typeof m.requests==='number' ? m.requests : '—'}</td>
      <td class="num">${fmtMetricRate(m.success_rate)}</td>
      <td class="num">${caseText}</td>
      <td class="num">${fmtMetricSeconds(m.ttft_p90_seconds)}</td>
      <td class="num">${fmtMetricSeconds(m.ttfc_p90_seconds)}</td>
      <td class="num">${fmtMetricSeconds(m.e2e_p90_seconds)}</td>
      <td class="num">${typeof m.tpot_p90_ms_per_token==='number' ? m.tpot_p90_ms_per_token.toFixed(1) : '—'}</td>
    </tr>
    <tr class="suite-detail" hidden><td colspan="10"><div class="case-detail"><p class="case-record">子记录 <code>${escapeHtml(item.record_id || '')}</code></p><table><thead><tr><th>用例</th><th>名称</th><th>状态</th><th>耗时(s)</th><th>结论说明</th></tr></thead><tbody>${detailRows}</tbody></table></div></td></tr>`;
  }).join('');
  return `<div class="batch-suites md-table-wrap"><table><thead><tr>
    <th>轮次</th><th>状态</th><th>结论</th><th>请求</th><th>成功率</th><th>用例通过</th><th>TTFT P90</th><th>TTFC P90</th><th>E2E P90</th><th>TPOT(ms/tok)</th>
  </tr></thead><tbody>${rows}</tbody></table>
  <p class="batch-suites-hint">点击套件行展开 / 收起该集合下的用例通过情况。请求数与成功率来自执行证据（requests.jsonl）；TTFT / TTFC P90 仅统计流式请求样本，E2E P90 覆盖全部请求。</p></div>`;
}
function renderBatchReport(text, manifest) {
  // 把报告按 "## " 标题分节；"各轮结果" 一节用可展开/收起的交互表格替代静态 markdown 表格。
  const lines = String(text).replace(/\r/g,'').split('\n');
  const sections = []; let current = { heading: null, lines: [] };
  for (const line of lines) {
    const h = line.match(/^##\s+(.+)$/);
    if (h) { sections.push(current); current = { heading: h[1].trim(), lines: [] }; }
    else current.lines.push(line);
  }
  sections.push(current);
  return sections.map(sec => {
    const body = sec.lines.join('\n');
    if (sec.heading === '各轮结果' && manifest) {
      const table = batchSuiteTable(manifest);
      if (table) return table;
    }
    const content = renderMarkdown(body);
    return sec.heading ? `<h2>${inlineMarkdown(sec.heading)}</h2>${content}` : content;
  }).join('');
}

window.openArtifact = async (id, kind) => {
  try {
    const response=await fetch(`/api/runs/${id}/${kind}`); if(!response.ok) throw new Error('文件尚未生成');
    const run=state.runs.find(r=>r.id===id);
    const titleFor=k=>`${run.is_batch?'批量聚合':run.case_name} · ${k==='report'?'测试报告':k==='summary'?'指标明细':k==='requests'?'请求明细':'执行日志'}`;
    if(kind==='report' && run.is_batch){
      // 批量报告：manifest 提供每轮指标与用例级结论，前端渲染成可展开/收起的套件表格。
      const text=await response.text();
      let manifest=null;
      try{ manifest=await api(`/api/runs/${id}/manifest`); }catch(e){ manifest=null; }
      $('#dialogEyebrow').textContent='REPORT'; $('#dialogTitle').textContent=titleFor('report');
      $('#reportBody').className='report-body markdown-body';
      $('#reportBody').innerHTML=renderBatchReport(text, manifest && manifest.record_type==='batch' ? manifest : null);
      $('#reportDialog').showModal(); return;
    }
    const text=await response.text();
    $('#dialogEyebrow').textContent=kind.toUpperCase(); $('#dialogTitle').textContent=titleFor(kind);
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
  const isGeminiImage=effectiveTestKind==='gemini-image';
  if(effectiveTestKind==='load' && !quick && state.selected.some(id=>['C1','SIMPLE_LOAD','FULL_LOAD'].includes(id)) && !confirm(`${state.selected.join('、')} 包含高成本测试，会产生较高 Token 成本，确认立即排队执行？`)) return;
  if(((isImage||isGeminiImage) && !selectedItem?.fixed_parameters || selectedItem?.manual_parameters) && (!Number($('#customRequests').value) || !Number($('#customConcurrency').value))){message.textContent='请填写总请求数和并发数';return}
  const button=$('#submitBtn'); button.disabled=true;
  try {
    const run=await api('/api/runs',{method:'POST',body:JSON.stringify({protocol:state.protocol,test_kind:effectiveTestKind,base_url:$('#baseUrl').value.trim(),api_key:$('#apiKey').value.trim(),model:$('#model').value.trim(),cases:state.selected,temporary_case:effectiveTestKind==='temporary'?state.temporaryCase:null,billing_probe:(effectiveTestKind==='fable'&&selectedItem?.billing_probe)?(state.billingProbe||'BILLING-ALL'):null,quick,requests:(effectiveTestKind==='load'||isImage||isGeminiImage)&&$('#customEnabled').checked?Number($('#customRequests').value):null,concurrency:effectiveTestKind==='moderation'?Number($('#moderationConcurrency').value):((state.testKind==='load'||isImage||isGeminiImage)&&$('#customEnabled').checked?Number($('#customConcurrency').value):null),image_quality:isImage?selectedValues('#imageQuality'):null,image_models:isImage?selectedValues('#imageModels'):null,image_sizes:isImage?selectedValues('#imageSizes'):null,gemini_quality:isGeminiImage?selectedValues('#geminiQuality'):null,gemini_models:isGeminiImage?selectedValues('#geminiModels'):null,gemini_aspects:isGeminiImage?selectedValues('#geminiAspects'):null,gemini_counts:isGeminiImage?selectedValues('#geminiCounts').map(Number):null})});
    state.runsPage=1; await refreshRuns(); toast(`${state.selected.join('、')} 已加入执行队列`);
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
// 批量报告：点击套件行展开 / 收起该集合下的用例级结果（事件委托，表格由 renderBatchReport 注入）。
$('#reportBody').addEventListener('click',event=>{
  const row=event.target.closest('.suite-row'); if(!row) return;
  const detail=row.nextElementSibling;
  if(!detail || !detail.classList.contains('suite-detail')) return;
  const open=detail.hidden;
  detail.hidden=!open; row.classList.toggle('open',open); row.setAttribute('aria-expanded',String(open));
});
document.querySelectorAll('#runForm input, #runForm select').forEach(input=>input.addEventListener('change',persistForm));
document.querySelectorAll('.kind-tab').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.kind-tab').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.testKind=button.dataset.kind;state.selected=[];if(state.testKind!=='temporary')state.temporaryCase='';if(state.testKind!=='fable')state.billingProbe=''; delete $('#customRequests').dataset.edited; delete $('#customConcurrency').dataset.edited; $('#customEnabled').checked=false;
  $('#quick').closest('.mode-toggle').style.display=state.testKind==='load'?'flex':'none';
  setModelPlaceholder();
  renderClaudeVariantConfig();
  renderCases();
  let first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x));
  if(!first){const matchingProtocol=state.protocols.find(protocol=>state.cases.some(x=>kindMatches(x)&&protocolSupports(x,protocol.id)));if(matchingProtocol){selectProtocol(matchingProtocol.id);first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x));}}
  if(first)state.selected=[first.id];renderCases();updateSelectionUI();updateEndpointPreview();
  if((state.testKind==='gpt-image'||state.testKind==='gemini-image') && first && !$('#model').value) $('#model').value=first.image_models?.[0]||first.title;
  persistForm();
});
document.querySelectorAll('.filter[data-filter]').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.filter[data-filter]').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.filter=button.dataset.filter;state.runsPage=1;refreshRuns();
});
document.querySelectorAll('.cfilter').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.cfilter').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.conclusion=button.dataset.conclusion;state.runsPage=1;refreshRuns();
});
$('#runsPrev').onclick=()=>{if(state.runsPage>1){state.runsPage--;refreshRuns()}};
$('#runsNext').onclick=()=>{state.runsPage++;refreshRuns()};
$('#runsPageSize').onchange=event=>{state.runsPageSize=Number(event.target.value);state.runsPage=1;refreshRuns()};
let runsSearchTimer;
$('#runsSearch').oninput=event=>{
  state.runsSearch=event.target.value.trim();state.runsPage=1;clearTimeout(runsSearchTimer);
  runsSearchTimer=setTimeout(()=>refreshRuns(),250);
};

(async function init(){
  try {
    const saved=readPersistedForm();
    state.testKind=['load','kvv','deepseek','glm','gpt','fable','gpt-image','gemini-image','awsb','open-source','temporary','moderation'].includes(saved.testKind)?saved.testKind:'load';
    if(saved.claudeVariant in CLAUDE_VARIANT_MODELS)state.claudeVariant=saved.claudeVariant;
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
    if(saved.billingProbe!=null)state.billingProbe=String(saved.billingProbe);
    if(typeof saved.quick==='boolean')$('#quick').checked=saved.quick;
    if(typeof saved.customEnabled==='boolean')$('#customEnabled').checked=saved.customEnabled;
    if(saved.customRequests!=null){$('#customRequests').value=String(saved.customRequests);$('#customRequests').dataset.edited='1'}
    if(saved.customConcurrency!=null){$('#customConcurrency').value=String(saved.customConcurrency);$('#customConcurrency').dataset.edited='1'}
    if(saved.moderationConcurrency!=null)$('#moderationConcurrency').value=String(saved.moderationConcurrency);
    state.selected=Array.isArray(saved.selected)?saved.selected.filter(id=>state.cases.some(item=>item.id===id&&protocolSupports(item))):[];
    if(!state.selected.length){const first=state.cases.find(item=>kindMatches(item)&&protocolSupports(item));if(first)state.selected=[first.id];}
    renderCases(); updateSelectionUI(); renderClaudeVariantConfig();
    [['#imageQuality',saved.imageQuality],['#imageModels',saved.imageModels],['#imageSizes',saved.imageSizes],['#geminiQuality',saved.geminiQuality],['#geminiModels',saved.geminiModels],['#geminiAspects',saved.geminiAspects],['#geminiCounts',saved.geminiCounts]].forEach(([selector,values])=>{
      if(!Array.isArray(values))return;
      const element=$(selector); if(!element)return;
      [...element.options].forEach(option=>option.selected=values.includes(option.value));
    });
    updateEndpointPreview(); persistForm(); await refreshRuns();
  }
  catch(e){$('#serviceState').innerHTML='<i style="background:#fb7185"></i>服务异常';toast(e.message)}
  setInterval(()=>refreshRuns(),3000); setInterval(()=>{state.runs.filter(r=>['queued','running'].includes(r.status)).forEach(r=>{const el=document.querySelector(`[data-start="${r.id}"]`);if(el)el.textContent=formatDuration(r)})},1000);
})();
