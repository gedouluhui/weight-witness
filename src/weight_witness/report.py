"""Self-contained, offline, progressively enhanced audit evidence."""
import base64
import hashlib
import html
import json

CSS = """
:root{color-scheme:light;--frost:#eef3f7;--paper:#fff;--navy:#17314d;--ink:#152333;--muted:#536579;--teal:#007d79;--amber:#9a5200;--red:#ad263c;--line:#ccd7e1;--mono:Consolas,'Liberation Mono',monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--frost);color:var(--ink);font:15px/1.65 'Segoe UI',system-ui,sans-serif}a{color:var(--navy)}a:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid var(--teal);outline-offset:4px}.skip{position:absolute;top:-70px;background:white;padding:12px}.skip:focus{top:12px;z-index:5}.wrap{max-width:1220px;margin:auto;padding:32px 42px 64px}.mast{display:flex;align-items:center;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:21px}.brand{font-size:15px;font-weight:750;letter-spacing:.05em}.brandmark{display:inline-flex;align-items:center;gap:3px;vertical-align:middle;margin-right:11px}.brandmark i{display:block;width:5px;background:var(--navy);height:16px}.brandmark i:nth-child(2){height:23px;background:var(--teal)}.utility,.eyebrow{font:11px var(--mono);text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}.hero{display:grid;grid-template-columns:1fr 260px;align-items:center;gap:35px;padding:46px 0 38px}h1{font-size:clamp(36px,5.5vw,67px);line-height:1.03;letter-spacing:-.05em;margin:16px 0 20px;font-weight:750;max-width:800px}.lede{max-width:650px;color:var(--muted);font-size:16px}.verdict{border-left:3px solid var(--amber);padding:6px 0 6px 24px}.verdict strong{display:block;font-size:24px;line-height:1.3;margin:8px 0}.verdict p{font-size:12px;color:var(--muted);margin:8px 0}.verdict[data-state=equal],.verdict[data-state=valid],.verdict[data-state=close]{border-color:var(--teal)}.overview{display:flex;flex-wrap:wrap;gap:28px;padding:22px 26px;background:var(--navy);color:white}.metric{min-width:88px}.metric b{display:block;font:27px/1.1 var(--mono);font-weight:500}.metric span{font-size:11px;letter-spacing:.05em;text-transform:uppercase;opacity:.8}.section{margin-top:34px}.section-head{display:flex;gap:15px;align-items:baseline;justify-content:space-between;margin-bottom:12px}h2{font-size:20px;margin:0;letter-spacing:-.03em}.hint{color:var(--muted);font-size:12px}.strip{display:flex;flex-wrap:wrap;gap:5px;padding:19px 20px;background:var(--paper);border:1px solid var(--line)}.cell{width:23px;height:34px;border-radius:2px;background:#73889c;display:block;text-decoration:none;transition:height .12s}.cell:hover{height:28px}.cell.equal,.cell.valid{background:var(--teal)}.cell.close{background:#428590}.cell.changed,.cell.nonfinite{background:var(--red)}.cell.added{background:#2a5aa0}.cell.removed{background:#9a5200}.cell.incompatible{background:#7554a3}.legend{display:flex;gap:17px;flex-wrap:wrap;margin:12px 0;font-size:11px;color:var(--muted)}.legend span:before{content:'';display:inline-block;width:7px;height:7px;margin-right:6px;background:var(--teal)}.legend .changed:before{background:var(--red)}.legend .added:before{background:#2a5aa0}.legend .removed:before{background:var(--amber)}.legend .incompatible:before{background:#7554a3}.controls{display:none;gap:20px;align-items:center;flex-wrap:wrap;background:var(--paper);padding:17px 20px;border:1px solid var(--line);border-bottom:0}.enhanced .controls{display:flex}.search{flex:1;min-width:200px}.search label{display:block;font-size:11px;font-weight:650;margin-bottom:4px}.search input{width:100%;max-width:500px;font:14px var(--mono);padding:9px 12px;border:1px solid #97a9b9;border-radius:3px;background:#f7f9fb;color:var(--ink)}.toggle{font-size:12px}.toggle input{accent-color:var(--teal)}.status-count{font:11px var(--mono);color:var(--muted)}.table-scroll{overflow:auto;border:1px solid var(--line);background:var(--paper)}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;font:11px var(--mono);font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);padding:13px 18px;border-bottom:1px solid var(--line);background:#f7f9fc}td{padding:17px 18px;border-bottom:1px solid #e4ebf0;vertical-align:top}tbody tr:last-child td{border-bottom:0}tbody tr:target{background:#eef7f6}.tensor-name{font:13px var(--mono);overflow-wrap:anywhere;display:block;min-width:140px;max-width:490px}.tensor-shape{font:11px var(--mono);color:var(--muted);margin-top:4px}.badge{font:10px var(--mono);text-transform:uppercase;white-space:nowrap;padding:4px 7px;background:#e3edf4;color:var(--navy);border-radius:2px}.badge.changed,.badge.nonfinite{background:#f9e8ec;color:#8a152d}.badge.equal,.badge.valid,.badge.close{background:#e3f3ef;color:#00605d}.badge.incompatible{background:#f0e9f7;color:#67468e}.number{font:14px var(--mono)}details{margin-top:9px}summary{cursor:pointer;color:var(--navy);font-size:12px;width:fit-content}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.65 var(--mono);background:#f5f8fb;border-left:2px solid var(--line);padding:13px;max-width:760px}.policy{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:28px}.note{padding:20px 24px;border-top:2px solid var(--navy);background:#e7eef4}.note h3{font-size:13px;margin:0 0 8px}.note p,.note li{font-size:12px;color:var(--muted)}.note ul{padding-left:17px;margin:8px 0}.note code{font-family:var(--mono);font-size:12px}.empty{padding:24px;color:var(--muted);background:white}.footer{display:flex;gap:20px;justify-content:space-between;border-top:1px solid var(--line);margin-top:32px;padding-top:18px;font:10px var(--mono);color:var(--muted)}[hidden]{display:none!important}@media(max-width:720px){.wrap{padding:20px 18px 40px}.hero{grid-template-columns:1fr;gap:18px;padding:28px 0}.verdict{padding-left:15px}.overview{gap:20px;padding:20px}.metric{min-width:65px}.metric b{font-size:22px}.policy{grid-template-columns:1fr}.section-head{display:block}.mast .utility{max-width:120px;text-align:right}.footer{display:block}th,td{padding:12px}.table-scroll{max-width:100%}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.cell{transition:none}}@media print{.controls,.skip{display:none!important}.wrap{max-width:none;padding:0}.hero{padding:20px 0}.table-scroll{overflow:visible}.note,pre,tr{break-inside:avoid}}
"""
SCRIPT = """
(() => {
  document.documentElement.classList.add('enhanced');
  const search = document.getElementById('tensor-search');
  const only = document.getElementById('diff-only');
  const rows = Array.from(document.querySelectorAll('tr[data-tensor]'));
  const counter = document.getElementById('result-count');
  const empty = document.getElementById('no-matches');
  function update() {
    const query = search.value.toLocaleLowerCase();
    let visible = 0;
    rows.forEach(row => {
      const state = row.dataset.state;
      const changed = !['equal', 'valid', 'close'].includes(state);
      row.hidden = !row.dataset.tensor.toLocaleLowerCase().includes(query) || (only.checked && !changed);
      if (!row.hidden) visible++;
    });
    counter.textContent = visible + ' / ' + rows.length + ' tensors shown';
    empty.hidden = visible !== 0;
  }
  search.addEventListener('input', update);
  only.addEventListener('change', update);
  document.querySelectorAll('.cell').forEach(cell => cell.addEventListener('click', () => {
    search.value = ''; only.checked = false; update();
  }));
  update();
})();
"""


def _e(value):
    return html.escape(str(value), quote=True)


def _n(value):
    if value is None:
        return 'not scanned'
    if isinstance(value, float):
        return format(value, '.6g')
    return str(value)


def render_html(report: dict) -> str:
    """Render a completed report. All report-originated content is escaped text."""
    if report.get('status') == 'error':
        error = _e(report.get('error', 'No diagnostic supplied.'))
        return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>Weight Witness — incomplete audit</title><style>{CSS}</style></head><body><main class="wrap"><header class="mast"><div class="brand">WEIGHT WITNESS</div><span class="utility">Incomplete audit</span></header><section class="hero"><div><h1>Audit could not complete</h1><p class="lede">No successful conclusion or zero-result count is available. Resolve the input or I/O issue and run the audit again.</p><pre role="alert">{error}</pre></div></section></main></body></html>'''
    inspect = report.get('kind') == 'inspect'
    rows = report.get('tensors', [])
    valid_states = {'equal', 'close', 'changed', 'added', 'removed', 'incompatible', 'valid', 'nonfinite'}
    overall = report.get('status', 'unknown')
    safe_overall = overall if overall in valid_states | {'different'} else 'unknown'
    title = 'Structural audit' if inspect else 'What changed in<br>your checkpoint?'
    explanation = ('Inspect the declared shards and tensor payloads without loading a model.' if inspect else
                   'Compare the weights, not the filenames. Every result below comes from the input tensor bytes, even when the shard layout changes.')
    headlines = {'different': 'Differences found', 'equal': 'Payloads identical', 'close': 'Within tolerance',
                 'valid': 'Structure validated', 'nonfinite': 'Nonfinite values found'}
    summary = report.get('summary', {})
    keys = ('tensors', 'files', 'bytes', 'nonfinite') if inspect else ('total', 'equal', 'close', 'changed', 'added', 'removed', 'incompatible')
    metrics = ''.join(f'<div class="metric"><b>{_e(summary.get(k, 0))}</b><span>{_e(k)}</span></div>' for k in keys)
    strip, table = [], []
    for i, row in enumerate(rows):
        state = ('nonfinite' if row.get('nonfinite') else 'valid') if inspect else row.get('status', 'unknown')
        state = state if state in valid_states else 'unknown'
        name = str(row.get('name', ''))
        d = row if inspect else (row.get('right') or row.get('left') or {})
        shape = d.get('shape', [])
        dim = ' × '.join(map(str, shape)) if shape else 'scalar'
        strip.append(f'<a class="cell {state}" href="#tensor-{i}" aria-label="{_e(name)}: {state}" title="{_e(name)}: {state}"></a>')
        number = row.get('nonfinite') if inspect else (row.get('metrics') or {}).get('max_abs')
        details = json.dumps(row, ensure_ascii=True, indent=2, allow_nan=False)
        table.append(f'<tr id="tensor-{i}" data-tensor="{_e(name)}" data-state="{state}"><td><span class="tensor-name">{_e(name)}</span><div class="tensor-shape">{_e(d.get("dtype", "—"))} · {_e(dim)} · {_e(d.get("nbytes", 0))} bytes</div><details><summary>View evidence</summary><pre>{_e(details)}</pre></details></td><td><span class="badge {state}">{state}</span></td><td class="number">{_e(_n(number))}</td></tr>')
    policy = report.get('policy', {})
    tolerance = ('Supported float payloads are scanned for NaN and infinity. Other types are fingerprinted only.' if inspect else
        f'<code>abs(right − left) ≤ atol + rtol × abs(left)</code><br>atol = {_e(policy.get("atol", 0))} · rtol = {_e(policy.get("rtol", 0))}<br>Equal means identical payload hashes; close means numerically within tolerance. Nonfinite floating values never pass. Other supported dtypes use bytewise equality.')
    limitations = ''.join(f'<li>{_e(item)}</li>' for item in report.get('limitations', []))
    source = _e(json.dumps(report.get('sources', {}), ensure_ascii=True, indent=2, allow_nan=False))
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    csp = f"default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="{_e(csp)}"><title>Weight Witness — checkpoint evidence</title><style>{CSS}</style></head>
<body><a class="skip" href="#evidence">Skip to tensor evidence</a><main class="wrap">
<header class="mast"><div class="brand"><span class="brandmark" aria-hidden="true"><i></i><i></i><i></i></span>WEIGHT WITNESS</div><div class="utility">Offline evidence · schema {_e(report.get('schema_version', 1))}</div></header>
{('<p class="note" role="note">' + _e(report['example']) + '</p>') if report.get('example') else ''}
<section class="hero"><div><div class="eyebrow">Safetensors / {'inspect' if inspect else 'compare'}</div><h1>{title}</h1><p class="lede">{explanation}</p></div><aside class="verdict" data-state="{safe_overall}"><span class="utility">Audit result</span><strong>{_e(headlines.get(overall, 'Unknown result'))}</strong><p>This is a weight-artifact check.<br>It is not a model-quality verdict.</p></aside></section>
<section class="overview" aria-label="Audit counts">{metrics}</section>
<section class="section"><div class="section-head"><h2>Tensor fingerprint</h2><span class="hint">One cell per tensor · select a cell to inspect its evidence</span></div><nav class="strip" aria-label="Tensor evidence shortcuts">{''.join(strip) or '<span>No tensors were scanned.</span>'}</nav><div class="legend"><span>Equal / valid</span><span class="changed">Changed / nonfinite</span><span class="added">Added</span><span class="removed">Removed</span><span class="incompatible">Incompatible</span></div></section>
<section class="section" id="evidence"><div class="section-head"><h2>Follow the evidence</h2><span class="hint">Full payload fingerprints · flat C-order difference positions</span></div>
<div class="controls"><div class="search"><label for="tensor-search">Search tensor names</label><input id="tensor-search" type="search" placeholder="e.g. attention.q_proj" autocomplete="off"></div><label class="toggle"><input id="diff-only" type="checkbox"> Differences / alerts only</label><span class="status-count" id="result-count" role="status" aria-live="polite"></span></div>
<div class="table-scroll"><table><thead><tr><th scope="col">Tensor / evidence</th><th scope="col">State</th><th scope="col">{'Nonfinite values' if inspect else 'Max abs delta'}</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p class="empty" id="no-matches" {'hidden' if rows else ''}>No tensors match this view. Clear the search or disable the filter.</p></section>
<div class="policy"><section class="note"><h3>{'Scan coverage' if inspect else 'The comparison rule'}</h3><p>{tolerance}</p><details><summary>Source header fingerprints</summary><pre>{source}</pre></details></section><section class="note"><h3>What this report does not prove</h3><ul>{limitations}</ul></section></div>
<footer class="footer"><span>WEIGHT WITNESS · LOCAL, READ-ONLY ANALYSIS</span><span>No telemetry · no external assets · no model execution</span></footer>
</main><script>{SCRIPT}</script></body></html>'''
