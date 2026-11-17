import importlib.util
import unittest
from html.parser import HTMLParser


class Tags(HTMLParser):
    def __init__(self):
        super().__init__(); self.tags = []
    def handle_starttag(self, tag, attrs): self.tags.append((tag, dict(attrs)))


def report(kind='diff', name='encoder.layer.0.weight'):
    descriptor = {'dtype': 'F32', 'shape': [2, 2], 'nbytes': 16, 'sha256': 'a' * 64}
    row = {'name': name, 'status': 'changed', 'left': descriptor, 'right': descriptor,
           'metrics': {'count': 4, 'max_abs': .5, 'rmse': .25, 'worst_index': 2, 'outside_tolerance': 1, 'finite_pairs': 4, 'nonfinite_left': 0, 'nonfinite_right': 0}}
    if kind == 'inspect': row = {'name': name, **descriptor, 'nonfinite': 0}
    return {'schema_version': 1, 'kind': kind, 'status': 'different' if kind == 'diff' else 'valid',
        'summary': {'total': 1, 'changed': 1} if kind == 'diff' else {'tensors': 1, 'bytes': 16, 'nonfinite': 0},
        'tensors': [row], 'policy': {'atol': 0, 'rtol': 0},
        'sources': {'left': {'files': ['base.safetensors'], 'header_sha256': {'base.safetensors': 'b' * 64}}},
        'limitations': ['Not model quality. Only declared shards.']}


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert importlib.util.find_spec('weight_witness.report'), 'report implementation not present'
        from weight_witness.report import render_html
        cls.render = staticmethod(render_html)

    def test_real_evidence_and_policy_present(self):
        html = self.render(report())
        for text in ('Weight Witness', 'encoder.layer.0.weight', '0.5', 'base.safetensors', 'Not model quality.', 'atol', 'rtol'):
            self.assertIn(text, html)
        self.assertIn('id="tensor-search"', html)
        self.assertIn('id="diff-only"', html)

    def test_untrusted_strings_cannot_become_markup(self):
        attack = '</script><img src=x onerror=alert(1)>"<&'
        data = report(name=attack)
        data['limitations'] = [attack]
        data['sources'] = {'input': {'files': [attack]}}
        html = self.render(data)
        parser = Tags(); parser.feed(html)
        self.assertFalse(any(tag == 'img' for tag, attrs in parser.tags))
        self.assertIn('&lt;/script&gt;', html)
        self.assertNotIn('onerror="', html)

    def test_no_external_assets_and_csp(self):
        parser = Tags(); parser.feed(self.render(report()))
        for tag, attrs in parser.tags:
            if tag in ('script', 'img', 'iframe', 'link'):
                self.assertNotIn('src', attrs)
                self.assertNotIn('href', attrs)
        self.assertTrue(any(tag == 'meta' and attrs.get('http-equiv') == 'Content-Security-Policy' for tag, attrs in parser.tags))

    def test_inspect_and_empty_without_javascript(self):
        html = self.render(report('inspect'))
        self.assertIn('Structural audit', html)
        self.assertIn('encoder.layer.0.weight', html)
        data = report(); data['tensors'] = []; data['summary'] = {'total': 0}
        self.assertIn('No tensors', self.render(data))

    def test_example_label_is_visible_and_escaped(self):
        data = report(); data['example'] = 'Synthetic weights <not a trained model>'
        self.assertIn('Synthetic weights &lt;not a trained model&gt;', self.render(data))

    def test_error_report_is_not_zero_result_or_unknown(self):
        result = self.render({'schema_version': 1, 'kind': 'inspect', 'status': 'error',
                              'error': 'unique-error-reason <img src=x>'})
        self.assertIn('Audit could not complete', result)
        self.assertIn('unique-error-reason &lt;img src=x&gt;', result)
        self.assertNotIn('Unknown result', result)
        self.assertNotIn('class="metric"', result)

    def test_status_attribute_not_injectable(self):
        data = report(); data['tensors'][0]['status'] = 'x" onclick="alert(1)'
        parser = Tags(); parser.feed(self.render(data))
        self.assertFalse(any('onclick' in attrs for _, attrs in parser.tags))


if __name__ == '__main__': unittest.main()
