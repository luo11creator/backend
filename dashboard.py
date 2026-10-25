"""
多维盾卫 — 后端管理面板
FastAPI 子应用，提供Web管理界面
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from datetime import datetime
import os

router = APIRouter()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>多维盾卫 · 管理面板</title>
<style>
:root {
  --bg: #0f172a; --card: #1e293b; --border: #334155;
  --primary: #3b82f6; --success: #10b981; --danger: #ef4444;
  --warning: #f59e0b; --text: #f1f5f9; --text2: #94a3b8;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--text);min-height:100vh}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:240px;
  background:var(--card);border-right:1px solid var(--border);
  padding:24px 20px;overflow-y:auto}
.sidebar h2{font-size:18px;margin-bottom:8px;color:var(--primary)}
.sidebar .ver{font-size:11px;color:var(--text2);margin-bottom:24px}
.sidebar nav a{display:flex;align-items:center;gap:10px;padding:10px 12px;
  border-radius:8px;color:var(--text2);text-decoration:none;
  font-size:14px;margin-bottom:4px;transition:.2s}
.sidebar nav a:hover,.sidebar nav a.active{background:#1e3a5f;color:var(--text)}
.sidebar .stat-mini{padding:12px;background:#0f172a;border-radius:8px;margin-top:20px}
.sidebar .stat-mini .val{font-size:24px;font-weight:700;color:var(--primary)}
.sidebar .stat-mini .lbl{font-size:11px;color:var(--text2);margin-top:4px}
.main{margin-left:240px;padding:32px}
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:32px}
.header h1{font-size:26px}
.header .status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--success)}
.header .status .dot{width:8px;height:8px;background:var(--success);border-radius:50%}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px}
.card{background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:20px}
.card h3{font-size:12px;color:var(--text2);text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:12px}
.card .big{font-size:36px;font-weight:700}
.card .big.blue{color:var(--primary)}
.card .big.green{color:var(--success)}
.card .big.red{color:var(--danger)}
.card .big.yellow{color:var(--warning)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.api-item{display:flex;align-items:center;gap:12px;padding:12px 16px;
  background:var(--card);border:1px solid var(--border);border-radius:8px;margin-bottom:8px}
.api-item .method{font-size:11px;font-weight:700;padding:4px 8px;
  border-radius:4px;color:#fff;min-width:48px;text-align:center}
.api-item .method.post{background:var(--primary)}
.api-item .method.get{background:var(--success)}
.api-item .path{font-family:monospace;font-size:13px;flex:1}
.api-item .arrow{color:var(--text2)}
.log-list{max-height:300px;overflow-y:auto;font-family:monospace;font-size:12px}
.log-item{padding:6px 0;border-bottom:1px solid var(--border);color:var(--text2)}
.log-item .ts{color:var(--text2);margin-right:8px}
.log-item.ok{color:var(--success)}
.log-item.err{color:var(--danger)}
@media(max-width:768px){.sidebar{display:none}.main{margin-left:0}}
</style>
</head>
<body>
<div class="sidebar">
  <h2>多维盾卫</h2>
  <div class="ver">v2.0 · 管理面板</div>
  <nav>
    <a href="#" class="active">仪表盘</a>
    <a href="/docs" target="_blank">API接口文档</a>
    <a href="#" onclick="testRandom()">快速测试</a>
    <a href="/api/health" target="_blank">健康检查</a>
  </nav>
  <div class="stat-mini">
    <div class="val" id="uptime">--</div>
    <div class="lbl">服务运行时间</div>
  </div>
</div>
<div class="main">
  <div class="header">
    <div>
      <h1>管理仪表盘</h1>
      <p style="color:var(--text2);margin-top:4px">实时监控AI检测服务运行状态</p>
    </div>
    <div class="status"><span class="dot"></span> 服务运行中</div>
  </div>

  <div class="stats">
    <div class="card">
      <h3>总请求数</h3>
      <div class="big blue" id="total">0</div>
    </div>
    <div class="card">
      <h3>音频检测</h3>
      <div class="big green" id="audio">0</div>
    </div>
    <div class="card">
      <h3>视频检测</h3>
      <div class="big yellow" id="video">0</div>
    </div>
    <div class="card">
      <h3>识破欺诈</h3>
      <div class="big red" id="fraud">0</div>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h3>API 端点</h3>
      <div class="api-item"><span class="method post">POST</span><span class="path">/api/v1/detect/audio</span><span>音频检测</span></div>
      <div class="api-item"><span class="method post">POST</span><span class="path">/api/v1/detect/video</span><span>视频检测</span></div>
      <div class="api-item"><span class="method post">POST</span><span class="path">/api/v1/detect/text</span><span>文本分析</span></div>
      <div class="api-item"><span class="method post">POST</span><span class="path">/api/v1/faceswap</span><span>换脸演示</span></div>
      <div class="api-item"><span class="method get">GET</span><span class="path">/api/v1/faceswap/status</span><span>换脸状态</span></div>
      <div class="api-item"><span class="method get">GET</span><span class="path">/api/v1/result/{id}</span><span>结果查询</span></div>
    </div>
    <div class="card">
      <h3>模型状态</h3>
      <div style="display:flex;flex-direction:column;gap:8px">
        <div style="display:flex;justify-content:space-between">
          <span>AASIST-L (音频)</span><span style="color:var(--success)" id="as-status">--</span></div>
        <div style="display:flex;justify-content:space-between">
          <span>Whisper (ASR)</span><span style="color:var(--warning)" id="wh-status">--</span></div>
        <div style="display:flex;justify-content:space-between">
          <span>可视化引擎</span><span style="color:var(--success)" id="vz-status">--</span></div>
      </div>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h3>视频上传检测</h3>
      <input type="file" id="videoFile" accept="video/*" style="margin-bottom:8px;color:var(--text)">
      <button onclick="uploadVideo()" style="background:var(--primary);color:#fff;border:none;
        padding:10px 20px;border-radius:8px;cursor:pointer;font-size:13px">上传并检测</button>
      <div id="videoResult" style="margin-top:12px;font-size:12px;color:var(--text2)"></div>
    </div>
    <div class="card">
      <h3>文本快速测试</h3>
      <textarea id="textInput" placeholder="输入测试文本..." style="width:100%;height:60px;
        background:#0f172a;border:1px solid var(--border);border-radius:8px;padding:10px;
        color:var(--text);font-size:13px;resize:none;margin-bottom:8px"></textarea>
      <button onclick="testText()" style="background:var(--primary);color:#fff;border:none;
        padding:10px 20px;border-radius:8px;cursor:pointer;font-size:13px">分析文本</button>
      <div id="textResult" style="margin-top:12px;font-size:12px;color:var(--text2)"></div>
    </div>
  </div>

  <div class="card">
    <h3>最近请求日志</h3>
    <div class="log-list" id="logList">
      <div class="log-item"><span class="ts">--</span>等待请求...</div>
    </div>
  </div>
</div>

<script>
async function refresh(){
  try{
    const r=await fetch('/api/health');
    const d=await r.json();
    document.getElementById('total').textContent=d.stats?.total||0;
    document.getElementById('audio').textContent=d.stats?.audio||0;
    document.getElementById('video').textContent=d.stats?.video||0;
    document.getElementById('fraud').textContent=d.stats?.fraud||0;
    document.getElementById('as-status').textContent=d.models?.audio?'已加载':'未加载';
    document.getElementById('wh-status').textContent=d.models?.whisper||false?'已加载':'未安装';
    document.getElementById('vz-status').textContent=d.models?.viz?'已加载':'未安装';
    const t=new Date(d.time||Date.now());
    document.getElementById('uptime').textContent=t.toLocaleTimeString();
  }catch(e){}
}
function log(msg,cls=''){
  const l=document.getElementById('logList');
  const ts=new Date().toLocaleTimeString();
  l.innerHTML='<div class="log-item '+cls+'"><span class="ts">'+ts+'</span>'+msg+'</div>'+l.innerHTML;
}
async function testText(){
  const text=document.getElementById('textInput').value||'测试文本';
  const r=document.getElementById('textResult');
  r.textContent='检测中...';
  try{
    const resp=await fetch('/api/v1/detect/text',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:text,model:'bert-base-chinese'})
    });
    const d=await resp.json();
    r.innerHTML='风险评分: <b>'+d.risk_score+'</b> | 等级: '+d.risk_level+' | '+d.verdict;
    log('文本分析: score='+d.risk_score+' verdict='+d.verdict,(d.risk_level>=4?'err':'ok'));
  }catch(e){r.textContent='错误: '+e.message;log('文本分析失败','err')}
}
async function uploadVideo(){
  const file=document.getElementById('videoFile').files[0];
  if(!file){alert('请选择视频文件');return}
  const r=document.getElementById('videoResult');
  r.textContent='上传中...';
  const fd=new FormData();fd.append('video',file);
  try{
    const resp=await fetch('/api/v1/detect/video',{method:'POST',body:fd});
    const d=await resp.json();
    r.innerHTML='风险评分: <b>'+d.risk_score+'</b> | 等级: '+d.risk_level+' | '+d.verdict;
    log('视频检测: '+file.name+' score='+d.risk_score,(d.risk_level>=4?'err':'ok'));
  }catch(e){r.textContent='错误: '+e.message;log('视频上传失败','err')}
}
async function testRandom(){
  log('快速测试已触发');
  await testText();
}
setInterval(refresh,5000);
refresh();
</script>
</body>
</html>
"""

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """管理面板首页"""
    return DASHBOARD_HTML
