"""Optional real-browser checks: pip install playwright; use an installed Edge.

A fresh headless context is used, never the user's browsing profile. Run after
examples/render_demo.py. Outputs are local test evidence, not usage analytics.
"""
import argparse
import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright
from weight_witness.report import render_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='msedge', help='installed Playwright browser channel')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifacts = root/'artifacts'; artifacts.mkdir(exist_ok=True)
    source = root/'docs/demo/index.html'
    report = json.loads((root/'docs/demo/report.json').read_text(encoding='utf8'))
    checks = []
    def check(name, truth):
        if not truth: raise AssertionError(name)
        checks.append(name)
    with sync_playwright() as p, tempfile.TemporaryDirectory() as temp:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        version = browser.version
        context = browser.new_context(viewport={'width': 1360, 'height': 1100}, reduced_motion='reduce')
        page = context.new_page()
        errors, external = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda req: external.append(req.url) if req.url.startswith(('http://', 'https://')) else None)
        page.goto(source.as_uri())
        check('script enabled under CSP', page.locator('#result-count').inner_text().startswith('13 / 13'))
        check('synthetic label', 'Synthetic weights' in page.locator('body').inner_text())
        check('all evidence rows', page.locator('tr[data-tensor]:visible').count() == 13)
        page.locator('#tensor-search').fill('q_proj')
        check('search changes actual rows', page.locator('tr[data-tensor]:visible').count() == 3)
        check('search names match', all('q_proj' in n for n in page.locator('tr[data-tensor]:visible').evaluate_all('(rows) => rows.map(r => r.dataset.tensor)')))
        page.locator('#tensor-search').fill('not-a-real-tensor')
        check('empty search state', page.locator('#no-matches').is_visible())
        page.locator('#tensor-search').fill('')
        page.locator('#diff-only').check()
        check('difference toggle changes actual rows', page.locator('tr[data-tensor]:visible').count() == 7)
        page.locator('.cell.equal').first.click()
        check('strip unhides target before navigation', not page.locator('#diff-only').is_checked() and page.locator('tr[data-tensor]:visible').count() == 13)
        summary = page.locator('tr[data-state=changed] summary').first
        summary.focus(); page.keyboard.press('Enter')
        check('keyboard expands details', page.locator('tr[data-state=changed] details').first.get_attribute('open') is not None)
        summary.press('Enter')
        for width in (1360, 768, 390):
            page.set_viewport_size({'width':width,'height':1000})
            check(f'no document horizontal overflow at {width}', page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
        page.set_viewport_size({'width':1360,'height':1100}); page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(root/'docs/demo/preview.png'))
        page.set_viewport_size({'width':390,'height':950}); page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(artifacts/'mobile.png'), full_page=True)
        check('no external network', not external)
        check('no page exceptions', not errors)
        nojs = browser.new_context(java_script_enabled=False)
        plain = nojs.new_page(); plain.goto(source.as_uri())
        check('all rows readable without JS', plain.locator('tr[data-tensor]:visible').count() == 13)
        check('no-JS controls hidden', not plain.locator('.controls').is_visible())
        attack = '</script><img src=x onerror="window.__injected=1">'
        report['tensors'][0]['name'] = attack
        hostile = Path(temp)/'hostile.html'; hostile.write_text(render_html(report), encoding='utf8')
        page.goto(hostile.as_uri())
        check('hostile name is text', attack in page.locator('tbody').inner_text())
        check('hostile name creates no element or effect', page.locator('img').count() == 0 and page.evaluate('window.__injected === undefined'))
        browser.close()
    result = {'browser': version, 'checks': checks, 'passed': len(checks), 'failed': 0,
              'scope': 'Local headless Chromium/Edge. Not Safari/Firefox, physical phone, or screen-reader validation.'}
    (artifacts/'browser-verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
