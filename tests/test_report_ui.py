import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

class ReportUiTests(unittest.TestCase):
    def test_markdown_renderer_and_styles_exist(self):
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        self.assertIn('function renderMarkdown',js)
        self.assertIn("kind==='report'",js)
        self.assertIn('.markdown-body h1',css)
        self.assertIn('.md-table-wrap',css)

    def test_batch_report_renders_expandable_suites_with_metrics(self):
        """批量聚合报告：指标列（正式请求/成功率/TTFT/TTFC/E2E）+ 套件行可展开用例明细。"""
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        server=(ROOT/'web/server.py').read_text(encoding='utf-8')
        self.assertIn('function renderBatchReport',js)
        self.assertIn('function batchSuiteTable',js)
        self.assertIn('caseBadge',js)
        for column in ('请求','成功率','用例通过','TTFT P90','TTFC P90','E2E P90','TPOT(ms/tok)'):
            self.assertIn(column,js)
        # 交互：点击套件行切换展开态（事件委托 + hidden 切换）
        self.assertIn(".closest('.suite-row')",js)
        self.assertIn('suite-detail',js)
        # 表格不横向滚动：无吸底条，子记录 ID 收进展开明细，表头紧凑
        self.assertNotIn('batch-hscroll',js)
        self.assertNotIn('wireBatchHScroll',js)
        self.assertNotIn('batch-hscroll',css)
        self.assertIn('case-record',js)
        self.assertIn('<th>TPOT(ms/tok)</th>',js)
        self.assertIn('.batch-suites th,.batch-suites td{padding:7px 10px}',css)
        # 样式与徽标
        self.assertIn('.suite-row.open',css)
        self.assertIn('.case-badge.passed',css)
        self.assertIn('.case-badge.failed',css)
        # 旧批次：manifest 读取时即时补齐 metrics
        self.assertIn('ensure_batch_metrics',server)

    def test_case_cards_show_complete_execution_parameters(self):
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        css=(ROOT/'web/static/app.css').read_text(encoding='utf-8')
        self.assertIn("caseFacts(item)", js)
        for label in ("并发 / 速率", "输入 Token", "总请求数", "预热时间", "稳定运行"):
            self.assertIn(label, js)
        self.assertIn("case-facts", css)
        self.assertIn("准入标准", js)
        self.assertIn("admission-standard", css)

    def test_generated_reports_include_ttft(self):
        reporting=(ROOT/'volcperf/reporting.py').read_text(encoding='utf-8')
        aggregation=(ROOT/'web/aggregation.py').read_text(encoding='utf-8')
        self.assertIn('TTFT P50/P90/P99',reporting)
        self.assertIn('ttft_p90_seconds',reporting)
        self.assertIn('TTFT P90(s)',aggregation)

    def test_non_report_artifacts_stay_plain_text(self):
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        self.assertIn("className='report-body code-body'",js)
        self.assertIn("textContent=text",js)

    def test_model_uses_searchable_combobox(self):
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        self.assertIn('id="modelSuggestions"',html)
        self.assertNotIn('id="modelSelect"',html)
        self.assertIn('function fuzzyModelMatch',js)
        self.assertIn("$('#model').oninput",js)
        self.assertIn('role="combobox"',html)

    def test_key_is_retained_and_password_save_is_discouraged(self):
        html=(ROOT/'web/static/index.html').read_text(encoding='utf-8')
        js=(ROOT/'web/static/app.js').read_text(encoding='utf-8')
        self.assertIn('id="apiKey"',html)
        self.assertIn('type="text" class="secret-masked"',html)
        self.assertIn('autocomplete="off"',html)
        self.assertIn('data-lpignore="true"',html)
        self.assertIn('data-1p-ignore="true"',html)
        self.assertNotIn("$('#apiKey').value=''",js)

if __name__=='__main__':unittest.main()
