// Markdown review-report viewer.
//
// The report format is defined by docs/report_format_spec.md. This file is a
// presentation layer only: it renders the Markdown and reads the YAML finding
// blocks for summary/filter UI. It is deliberately not the only thing that can
// parse a report -- nothing here is required to consume one.

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const dashboardSection = document.getElementById('dashboard');
    const dashboardContent = document.getElementById('dashboard-content');
    const filterBar = document.getElementById('filter-bar');
    const reportContent = document.getElementById('report-content');
    const markdownBody = document.getElementById('markdown-body');
    const statusBanner = document.getElementById('status-banner');
    const exportBtn = document.getElementById('export-btn');
    const themeToggle = document.getElementById('theme-toggle');
    const sidebar = document.getElementById('sidebar');
    const navMenu = document.getElementById('nav-menu');
    const body = document.body;

    const STATUS_LABELS = {
        pass: '통과',
        warning: '주의',
        needs_revision: '수정 필요',
        blocked: '판정 없음',
    };

    let findings = [];
    let frontMatter = {};
    let activeFilter = 'all';

    // --- Theming (unchanged behavior) ---
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        body.setAttribute('data-theme', 'dark');
        themeToggle.innerHTML = '<i class="ri-sun-line"></i>';
    }
    themeToggle.addEventListener('click', () => {
        if (body.getAttribute('data-theme') === 'dark') {
            body.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
            themeToggle.innerHTML = '<i class="ri-moon-line"></i>';
        } else {
            body.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            themeToggle.innerHTML = '<i class="ri-sun-line"></i>';
        }
    });

    // --- Input: drag & drop, file picker, ?file= auto-load ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    dropZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') fileInput.click();
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function showBanner(message, kind) {
        statusBanner.textContent = message;
        statusBanner.className = `status-banner ${kind || ''}`;
        statusBanner.style.display = 'block';
    }
    function hideBanner() {
        statusBanner.style.display = 'none';
    }

    function handleFile(file) {
        if (!/\.(md|markdown)$/i.test(file.name)) {
            showBanner('Markdown(.md) 리포트만 지원합니다. TXT 리포트는 더 이상 사용하지 않습니다.', 'error');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => renderReport(e.target.result);
        reader.onerror = () => showBanner('파일을 읽지 못했습니다.', 'error');
        reader.readAsText(file, 'utf-8');
    }

    async function loadFromUrl(url) {
        showBanner('리포트를 불러오는 중...', '');
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            renderReport(await res.text());
        } catch (err) {
            showBanner(`리포트를 불러오지 못했습니다: ${err.message}`, 'error');
        }
    }

    // --- Parsing: front matter + per-finding YAML blocks ---

    function splitFrontMatter(text) {
        const normalized = text.replace(/^﻿/, '').replace(/\r\n?/g, '\n');
        const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(normalized);
        if (!match) return { meta: {}, body: normalized };
        let meta = {};
        try {
            meta = jsyaml.load(match[1]) || {};
        } catch (err) {
            console.warn('front matter parse failed', err);
        }
        return { meta, body: match[2] };
    }

    function collectFindings(bodyText) {
        // Each finding is a `### heading` followed by a ```yaml block.
        const found = [];
        const re = /^###\s+(.+?)\s*(?:\{#([^}]+)\})?\s*$\n+```yaml\n([\s\S]*?)\n```/gm;
        let m;
        while ((m = re.exec(bodyText)) !== null) {
            let meta = {};
            try {
                meta = jsyaml.load(m[3]) || {};
            } catch (err) {
                console.warn('finding yaml parse failed', err);
            }
            found.push({
                heading: m[1].trim(),
                anchor: m[2] || slugify(meta.finding_id || m[1]),
                status: meta.status || 'blocked',
                applyStatus: meta.apply_status || '',
                findingId: meta.finding_id || m[1].trim(),
                sheet: meta.sheet || '',
            });
        }
        return found;
    }

    function slugify(value) {
        return String(value).toLowerCase().replace(/[^0-9a-z]+/g, '-').replace(/^-|-$/g, '');
    }

    // --- Rendering ---

    function renderReport(text) {
        const { meta, body: bodyText } = splitFrontMatter(text);
        frontMatter = meta;
        findings = collectFindings(bodyText);

        const md = window.markdownit({ html: true, linkify: true, breaks: false });
        // Renders `> [!type] title` (Obsidian callouts / GitHub alerts). Absent
        // if the CDN module failed to load -- callouts then fall back to plain
        // blockquotes, which is degraded but never broken.
        if (window.mditObsidianCallouts) md.use(window.mditObsidianCallouts);
        // DOMPurify is the security boundary: report bodies can contain raw HTML
        // written by an agent or a human, and html:true above lets it through.
        // svg:true is required too -- callout title icons are inline <svg>.
        const clean = DOMPurify.sanitize(md.render(bodyText), { USE_PROFILES: { html: true, svg: true } });
        markdownBody.innerHTML = clean;

        decorateFindings();
        buildDashboard();
        buildNav();
        applyFilter('all');

        uploadSection.style.display = 'none';
        hideBanner();
        dashboardSection.style.display = 'block';
        reportContent.style.display = 'block';
        sidebar.style.display = 'block';
        exportBtn.style.display = 'inline-flex';
        window.scrollTo({ top: 0 });
    }

    function decorateFindings() {
        // Anchor each finding section and tag it so filters can hide it.
        const headings = markdownBody.querySelectorAll('h3');
        headings.forEach((h3) => {
            const text = h3.textContent.trim();
            const finding = findings.find((f) => text.startsWith(f.heading)) ||
                findings.find((f) => f.heading.startsWith(text));
            if (!finding) return;
            const section = document.createElement('section');
            section.className = 'finding';
            section.id = finding.anchor;
            section.dataset.status = finding.status;
            h3.parentNode.insertBefore(section, h3);

            // Move the heading and everything up to the next h2/h3 into the section.
            let node = h3;
            const collected = [];
            while (node && !(node !== h3 && /^H[23]$/.test(node.tagName))) {
                collected.push(node);
                node = node.nextSibling;
            }
            collected.forEach((n) => section.appendChild(n));

            const chip = document.createElement('span');
            chip.className = `status-chip status-${finding.status}`;
            chip.textContent = STATUS_LABELS[finding.status] || finding.status;
            h3.appendChild(chip);

            collapseStructuredBlocks(section);
            pairTranslationBlocks(section);
            addCopyButtons(section);
        });
    }

    function collapseStructuredBlocks(section) {
        // Only the finding's own YAML meta block needs client-side folding --
        // report_builder.py emits it as a bare fenced block. The Payload JSON
        // blocks are already wrapped in a native <details> in the .md itself
        // (report_builder.py's _details()), so they collapse the same way
        // here and in Obsidian; re-wrapping them here would double-nest.
        section.querySelectorAll('pre > code').forEach((code) => {
            const cls = code.className || '';
            if (!/language-yaml/.test(cls)) return;
            const pre = code.parentElement;
            const details = document.createElement('details');
            const summary = document.createElement('summary');
            summary.textContent = '메타데이터 (YAML)';
            details.appendChild(summary);
            pre.parentNode.insertBefore(details, pre);
            details.appendChild(pre);
        });
    }

    function pairTranslationBlocks(section) {
        // Show 현재 번역문 / 제안 번역문 side by side, directly under 원문 --
        // NOT at the end of the card. 원문/현재 번역문/제안 번역문 render as
        // `.callout` divs (title baked into `.callout-title-inner`), not
        // h4 headings, so match on that instead. Insert the grid at 현재
        // 번역문's original slot before moving it in; appendChild alone would
        // relocate the pair to the bottom, after 검수 상세 / Payload sections.
        const callouts = Array.from(section.querySelectorAll('.callout'));
        const byTitle = (text) => callouts.find((c) => {
            const title = c.querySelector('.callout-title-inner');
            return title && title.textContent.trim() === text;
        });
        const current = byTitle('현재 번역문');
        const proposed = byTitle('제안 번역문');
        if (!current || !proposed) return;

        const grid = document.createElement('div');
        grid.className = 'translation-pair';
        section.insertBefore(grid, current);
        grid.appendChild(current);
        grid.appendChild(proposed);
    }

    // The old card UI let reviewers copy a cell's text with one click (to
    // paste back into Excel); the callout redesign dropped that. Restored
    // here as a per-callout button -- 원문/현재 번역문/제안 번역문 only, the
    // three things someone would actually paste elsewhere.
    const COPYABLE_CALLOUTS = ['원문', '현재 번역문', '제안 번역문'];

    function addCopyButtons(section) {
        section.querySelectorAll('.callout').forEach((callout) => {
            const titleEl = callout.querySelector('.callout-title-inner');
            const title = titleEl ? titleEl.textContent.trim() : '';
            if (!COPYABLE_CALLOUTS.includes(title)) return;
            const code = callout.querySelector('.callout-content pre code');
            if (!code) return;

            const btn = document.createElement('button');
            btn.className = 'copy-btn';
            btn.type = 'button';
            btn.title = `${title} 복사`;
            btn.innerHTML = '<i class="ri-file-copy-line"></i>';
            btn.addEventListener('click', () => copyToClipboard(code.textContent, btn));
            callout.querySelector('.callout-title').appendChild(btn);
        });
    }

    async function copyToClipboard(text, btn) {
        try {
            await navigator.clipboard.writeText(text);
            btn.innerHTML = '<i class="ri-check-line"></i>';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.innerHTML = '<i class="ri-file-copy-line"></i>';
                btn.classList.remove('copied');
            }, 1500);
        } catch (err) {
            console.error('Failed to copy:', err);
            alert('클립보드 복사에 실패했습니다.');
        }
    }

    function buildDashboard() {
        const counts = { pass: 0, warning: 0, needs_revision: 0, blocked: 0 };
        let pendingApproval = 0;
        findings.forEach((f) => {
            if (counts[f.status] === undefined) counts[f.status] = 0;
            counts[f.status] += 1;
            if (f.applyStatus === 'pending_approval') pendingApproval += 1;
        });

        const cards = [
            { label: '총 검수 항목', value: findings.length, key: 'total' },
            { label: STATUS_LABELS.needs_revision, value: counts.needs_revision || 0, key: 'needs_revision' },
            { label: STATUS_LABELS.warning, value: counts.warning || 0, key: 'warning' },
            { label: STATUS_LABELS.pass, value: counts.pass || 0, key: 'pass' },
            { label: '승인 대기 제안', value: pendingApproval, key: 'pending' },
        ];
        if (counts.blocked) {
            cards.push({ label: STATUS_LABELS.blocked, value: counts.blocked, key: 'blocked' });
        }

        dashboardContent.innerHTML = '';
        cards.forEach((card) => {
            const el = document.createElement('div');
            el.className = `stat-card stat-${card.key}`;
            el.innerHTML = `<div class="stat-value"></div><div class="stat-label"></div>`;
            el.querySelector('.stat-value').textContent = card.value;
            el.querySelector('.stat-label').textContent = card.label;
            dashboardContent.appendChild(el);
        });

        const metaBits = [];
        if (frontMatter.report_id) metaBits.push(`report_id: ${frontMatter.report_id}`);
        if (frontMatter.source_file_id) metaBits.push(`source: ${frontMatter.source_file_id}`);
        if (frontMatter.translation_model) metaBits.push(`translation: ${frontMatter.translation_model}`);
        if (frontMatter.audit_model) metaBits.push(`audit: ${frontMatter.audit_model}`);
        if (frontMatter.generated_at) metaBits.push(frontMatter.generated_at);
        if (metaBits.length) {
            const meta = document.createElement('div');
            meta.className = 'report-meta';
            meta.textContent = metaBits.join(' · ');
            dashboardContent.appendChild(meta);
        }

        filterBar.innerHTML = '';
        const filters = [['all', '전체']].concat(
            Object.keys(STATUS_LABELS)
                .filter((k) => counts[k])
                .map((k) => [k, `${STATUS_LABELS[k]} (${counts[k]})`])
        );
        filters.forEach(([key, label]) => {
            const btn = document.createElement('button');
            btn.className = 'filter-btn';
            btn.dataset.filter = key;
            btn.textContent = label;
            btn.addEventListener('click', () => applyFilter(key));
            filterBar.appendChild(btn);
        });
    }

    function applyFilter(key) {
        activeFilter = key;
        filterBar.querySelectorAll('.filter-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.filter === key);
        });
        markdownBody.querySelectorAll('.finding').forEach((section) => {
            section.style.display = key === 'all' || section.dataset.status === key ? '' : 'none';
        });
        navMenu.querySelectorAll('.nav-link').forEach((link) => {
            link.style.display = key === 'all' || link.dataset.status === key ? '' : 'none';
        });
    }

    function buildNav() {
        navMenu.innerHTML = '';
        findings.forEach((f) => {
            const link = document.createElement('a');
            link.className = 'nav-link';
            link.href = `#${f.anchor}`;
            link.dataset.status = f.status;
            const dot = document.createElement('span');
            dot.className = `nav-dot status-${f.status}`;
            const label = document.createElement('span');
            label.textContent = f.heading;
            link.appendChild(dot);
            link.appendChild(label);
            navMenu.appendChild(link);
        });
    }

    // --- PDF export (unchanged behavior) ---
    exportBtn.addEventListener('click', () => {
        const opts = {
            margin: 10,
            filename: `${frontMatter.report_id || 'review-report'}.pdf`,
            image: { type: 'jpeg', quality: 0.95 },
            html2canvas: { scale: 2, useCORS: true },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
        };
        html2pdf().set(opts).from(markdownBody).save();
    });

    // Auto-load ?file=/api/report/{task_id}
    const fileParam = new URLSearchParams(window.location.search).get('file');
    if (fileParam) loadFromUrl(fileParam);
});
