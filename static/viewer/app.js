document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const dashboardSection = document.getElementById('dashboard');
    const dashboardContent = document.getElementById('dashboard-content');
    const reportContent = document.getElementById('report-content');
    const itemsList = document.getElementById('items-list');
    const exportBtn = document.getElementById('export-btn');
    const themeToggle = document.getElementById('theme-toggle');
    const sidebar = document.getElementById('sidebar');
    const navMenu = document.getElementById('nav-menu');
    const body = document.body;

    // --- State ---
    let parsedData = {
        meta: {},
        items: [],
        sections: [], // Stores unique sheet names like "UK(영국)"
        stats: {
            total: 0,
            pass: 0,
            warn: 0,
            fail: 0,
            casingIssues: 0,
            glossaryIssues: 0
        }
    };

    // --- Theming ---
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

    // --- Drag and Drop ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.name.endsWith('.txt')) {
            alert('TXT 파일만 업로드 가능합니다.');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            parseTxtContent(e.target.result);
            renderDashboard();
            renderItems();
            renderSidebar();
            initScrollObserver();

            uploadSection.style.display = 'none';
            dashboardSection.style.display = 'block';
            reportContent.style.display = 'block';
            sidebar.style.display = 'block';
            exportBtn.style.display = 'flex';
        };
        reader.readAsText(file);
    }

    // --- Parser Engine ---
    function parseTxtContent(text) {
        parsedData.items = [];
        parsedData.sections = [];
        parsedData.stats = { total: 0, pass: 0, warn: 0, fail: 0, casingIssues: 0, glossaryIssues: 0 };

        const delimiter = "==========================================================================================";
        let parts = text.split(delimiter).map(p => p.trim()).filter(p => p.length > 0);
        if (parts.length === 0) return;

        for (let i = 1; i < parts.length; i++) {
            let itemText = parts[i];
            if (itemText.split('\n').length < 3) continue;

            const getBlock = (pattern) => {
                const regex = new RegExp(`\\[${pattern}\\]([\\s\\S]*?)(?=\\n\\[상세 -|$)`);
                const match = itemText.match(regex);
                return match ? match[1].trim() : "";
            };

            const fullHeader = (itemText.match(/^(.*?)\n-+/) || ["", ""])[1].trim();
            // Extact sheet name like '[시트] UK(영국) | [셀] C15' -> 'UK(영국)'
            let sheetName = "";
            const sheetMatch = fullHeader.match(/\[시트\]\s*([^|]+)/);
            if (sheetMatch) {
                sheetName = sheetMatch[1].trim();
                if (!parsedData.sections.includes(sheetName)) {
                    parsedData.sections.push(sheetName);
                }
            }

            let item = {
                header: fullHeader,
                sheetName: sheetName,
                sourceText: getBlock('상세 - 원문'),
                targetText: getBlock('상세 - 번역문'),
                casingCheck: getBlock('상세 - 대소문자 점검'),
                casingSuggestion: getBlock('단순 규칙 기반 문장형 변환안') || (itemText.match(/\[단순 규칙 기반 문장형 변환안\]:?\n([\s\S]*?)(?=\n\[|$)/) || ["", ""])[1].trim(),
                glossaryCheck: getBlock('상세 - 용어집 점검'),
                ragCheck: getBlock('상세 - RAG Payload') || getBlock('상세 - RAG 일관성 참고'),
                backTranslation: getBlock('상세 - 역번역'),
                geminiQa: getBlock('상세 - AI Payload') || getBlock('상세 - AI 검수 결과') || getBlock('상세 - Gemini Payload') || getBlock('상세 - Gemini 검수 결과'),
                status: 'pass',
                tags: []
            };

            // Analyze Status and Tags
            if (item.casingCheck && !item.casingCheck.includes('추가 대문자 수: 0개') && item.casingCheck.includes('문장형 아님')) {
                item.tags.push('대소문자'); 
                parsedData.stats.casingIssues++;
            }
            if (item.glossaryCheck && !item.glossaryCheck.includes('별도 지적 사항 없음')) {
                item.tags.push('용어집'); 
                parsedData.stats.glossaryIssues++;
            }

            if (item.geminiQa && (item.geminiQa.includes('[AI QA 오류]') || item.geminiQa.includes('[Gemini QA 오류]'))) {
                item.status = 'fail'; 
                item.tags.push('AI 오류');
            } else if (item.geminiQa) {
                try {
                    const aiData = JSON.parse(item.geminiQa);
                    const isExcellent = aiData.is_excellent === true || aiData.grade === 'Excellent';
                    const needsRevision = aiData.grade === 'Needs Revision' || aiData.is_excellent === false;
                    const hasFix = (aiData.suggested_fix && aiData.suggested_fix.trim() !== '') || aiData.grade === 'Good';
                    
                    if (needsRevision) {
                        item.status = 'fail';
                    } else if (hasFix || !isExcellent) {
                        item.status = 'warn';
                        if (!item.tags.includes('AI 제안')) item.tags.push('AI 추천');
                    }
                } catch (e) {
                    // Fallback to string matching for non-JSON or malformed data
                    if (item.geminiQa.includes('Needs Revision') || item.geminiQa.includes('수정 필요') || item.geminiQa.includes('오류')) {
                        item.status = 'fail';
                    } else if (item.geminiQa.includes('수정 제안') || item.geminiQa.includes('Good') || item.geminiQa.includes('양호')) {
                        item.status = 'warn';
                        if (!item.tags.includes('AI 제안')) item.tags.push('AI 추천');
                    }
                }
            }

            if (item.status === 'pass') parsedData.stats.pass++;
            if (item.status === 'warn') parsedData.stats.warn++;
            if (item.status === 'fail') parsedData.stats.fail++;

            parsedData.stats.total++;
            parsedData.items.push(item);
        }
    }

    // --- Render Dashboard ---
    function renderDashboard() {
        const score = Math.round((parsedData.stats.pass / (parsedData.stats.total || 1)) * 100);
        let scoreColor = 'stat-success';
        if (score < 80) scoreColor = 'stat-warning';
        if (score < 60) scoreColor = 'stat-danger';

        dashboardContent.innerHTML = `
            <div class="stat-card ${scoreColor}">
                <div class="stat-icon"><i class="ri-percent-line"></i></div>
                <div class="stat-info">
                    <div class="stat-label">Overall Health Score</div>
                    <div class="stat-value">${score}점</div>
                    <div class="stat-subtext">총 ${parsedData.stats.total}개 항목 중 ${parsedData.stats.pass}개 통과</div>
                </div>
            </div>
            <div class="stat-card stat-primary">
                <div class="stat-icon"><i class="ri-checkbox-circle-line"></i></div>
                <div class="stat-info">
                    <div class="stat-label">검토 완료 (Pass)</div>
                    <div class="stat-value">${parsedData.stats.pass}건</div>
                    <div class="stat-subtext">수정 불필요</div>
                </div>
            </div>
            <div class="stat-card stat-warning">
                <div class="stat-icon"><i class="ri-error-warning-line"></i></div>
                <div class="stat-info">
                    <div class="stat-label">수정 권장 (Warning)</div>
                    <div class="stat-value">${parsedData.stats.warn}건</div>
                    <div class="stat-subtext">AI가 개선을 제안한 항목</div>
                </div>
            </div>
            <div class="stat-card stat-danger">
                <div class="stat-icon"><i class="ri-close-circle-line"></i></div>
                <div class="stat-info">
                    <div class="stat-label">수정 필수 (Fail)</div>
                    <div class="stat-value">${parsedData.stats.fail}건</div>
                    <div class="stat-subtext">AI가 수정을 필수로 판정한 항목</div>
                </div>
            </div>
            <div class="stat-card stat-info" style="background: var(--bg-secondary); border: 1px solid var(--border-color);">
                <div class="stat-icon" style="color: var(--text-secondary);"><i class="ri-microscope-line"></i></div>
                <div class="stat-info">
                    <div class="stat-label" style="color: var(--text-secondary);">참고: 기계검수 (Heuristic)</div>
                    <div class="stat-value" style="color: var(--text-primary); font-size: 1.4rem;">${parsedData.stats.casingIssues + parsedData.stats.glossaryIssues}건</div>
                    <div class="stat-subtext">용어집 ${parsedData.stats.glossaryIssues}건, 대소문자 ${parsedData.stats.casingIssues}건</div>
                </div>
            </div>
        `;
    }

    // --- Render Sidebar ---
    function renderSidebar() {
        navMenu.innerHTML = `
            <a href="#dashboard" class="nav-link" data-target="dashboard">
                <i class="ri-dashboard-line" style="margin-right: 8px;"></i> 요약 (Dashboard)
            </a>
        `;
        parsedData.sections.forEach((section) => {
            const safeId = section.replace(/[\W_]+/g, "-").toLowerCase();
            navMenu.insertAdjacentHTML('beforeend', `
                <a href="#section-${safeId}" class="nav-link" data-target="section-${safeId}">
                    <i class="ri-global-line" style="margin-right: 8px;"></i> ${escapeHTML(section)}
                </a>
            `);
        });
    }

    // --- Simple Diff Logic ---
    function generateDiffHtml(original, suggested) {
        if (!suggested || original === suggested) return escapeHTML(original);
        const origWords = original.split(/\s+/);
        const suggWords = suggested.split(/\s+/);
        let html = '';
        suggWords.forEach((word) => {
            const exactMatch = origWords.find(w => w === word);
            const lowerMatch = origWords.find(w => w.toLowerCase() === word.toLowerCase());
            if (exactMatch) {
                html += escapeHTML(word) + ' ';
            } else if (lowerMatch) {
                html += `<span class="diff-highlight-warn" title="${escapeHTML(lowerMatch)} -> ${escapeHTML(word)}">${escapeHTML(word)}</span> `;
            } else {
                html += `<span class="diff-highlight-add">${escapeHTML(word)}</span> `;
            }
        });
        return html;
    }

    // --- Render Items ---
    function renderItems() {
        itemsList.innerHTML = '';

        let currentSection = "";

        parsedData.items.forEach((item, index) => {
            // Anchor ID generation
            let wrapperId = "";
            if (item.sheetName && item.sheetName !== currentSection) {
                currentSection = item.sheetName;
                const safeId = currentSection.replace(/[\W_]+/g, "-").toLowerCase();
                wrapperId = `id="section-${safeId}"`;
            }

            let statusTagHtml = '';
            if (item.status === 'pass') statusTagHtml = `<span class="status-tag tag-pass"><i class="ri-check-line"></i> Pass</span>`;
            if (item.status === 'warn') statusTagHtml = `<span class="status-tag tag-warn"><i class="ri-alert-line"></i> Warning</span>`;
            if (item.status === 'fail') statusTagHtml = `<span class="status-tag tag-fail"><i class="ri-close-line"></i> Fail</span>`;

            let tagsHtml = item.tags.map(t => `<span class="cat-tag">${t}</span>`).join('');

            let htmlString = `
                <div class="card-wrapper scroll-target" ${wrapperId}>
                <div class="review-card status-${item.status}">
                    <div class="card-header">
                        <div class="card-title-group">
                            <span class="card-badge"><i class="ri-map-pin-line"></i> ${escapeHTML(item.header || `항목 ${index + 1}`)}</span>
                            ${statusTagHtml}
                        </div>
                        <div class="category-tags">${tagsHtml}</div>
                    </div>
                    <div class="card-body">
                        <div class="diff-viewer">
                            <div class="diff-panel">
                                <span class="diff-header"><i class="ri-text"></i> 원문 (Source)</span>
                                <div class="diff-content">${escapeHTML(item.sourceText)}</div>
                            </div>
                            <div class="diff-panel">
                                <span class="diff-header"><i class="ri-translate"></i> 번역문 (Target)</span>
                                <div class="diff-content">${escapeHTML(item.targetText)}</div>
                            </div>
                        </div>
                        ${(item.casingSuggestion && item.status !== 'pass' && item.casingSuggestion.trim() !== '') ? `
                        <div class="diff-viewer diff-suggestion" style="margin-top: 10px; border-top: 1px dashed rgba(0,0,0,0.1); padding-top: 10px;">
                            <div class="diff-panel" style="width: 100%;">
                                <span class="diff-header" style="color:var(--warning-color);"><i class="ri-edit-line"></i> 교정 제안 (Diff Viewer)</span>
                                <div class="diff-content">${generateDiffHtml(item.targetText, item.casingSuggestion)}</div>
                            </div>
                        </div>` : ''}
                        
                        <div class="analysis-section">
            `;

            // 역번역 카드
            const backTransText = (item.backTranslation && !item.backTranslation.includes('False') && item.backTranslation.trim() !== '') 
                ? escapeHTML(item.backTranslation) 
                : '<span style="color:#94a3b8;">생략됨 (지원하지 않거나 수행되지 않음)</span>';
            htmlString += `
                <div class="action-card info">
                    <span class="action-title"><i class="ri-arrow-left-right-line"></i> 역번역 결과 (Back Translation)</span>
                    <div class="action-content" style="color: ${backTransText.includes('생략됨') ? '#94a3b8' : '#2b6cb0'};">${backTransText}</div>
                </div>
            `;

            // RAG 카드
            let ragText = '';
            const ragRaw = item.ragCheck || '';
            if (ragRaw.trim() !== '' && ragRaw !== '[]' && !ragRaw.includes('별도 지적 사항 없음')) {
                try {
                    const ragData = JSON.parse(ragRaw);
                    if (ragData && Array.isArray(ragData) && ragData.length > 0) {
                        if (ragData[0].error) {
                            ragText = `<div style="color:var(--danger-color);"><i class="ri-error-warning-line"></i> RAG 조회 오류: ${escapeHTML(ragData[0].error)}</div>`;
                        } else {
                            let ragHtml = '';
                            ragData.forEach(r => {
                                let color = r.score > 90 ? '#10b981' : (r.score > 70 ? '#f59e0b' : '#ef4444');
                                ragHtml += `
                                    <div style="background: rgba(0,0,0,0.02); border: 1px solid rgba(0,0,0,0.05); border-radius: 6px; padding: 12px; margin-bottom: 10px;">
                                        <div style="display:flex; justify-content:space-between; margin-bottom: 8px; font-size: 0.85rem; color:#64748b; font-weight:500;">
                                            <span><i class="ri-article-line"></i> ${escapeHTML(r.story_id)} / ${escapeHTML(r.section)}</span>
                                            <span style="color:${color};"><i class="ri-percent-line"></i> ${r.score.toFixed(1)}% Match</span>
                                        </div>
                                        <div style="width: 100%; background: #e2e8f0; height: 6px; border-radius: 3px; margin-bottom: 10px; overflow:hidden;">
                                            <div style="width: ${r.score}%; height: 100%; background: ${color}; border-radius: 3px; transition: width 1s ease-in-out;"></div>
                                        </div>
                                        <div style="font-size: 0.95rem; color: #334155; font-weight:500;"><i class="ri-text"></i> ${escapeHTML(r.source)}</div>
                                        <div style="font-size: 0.95rem; color: #1e293b; font-weight:500; margin-top:4px;"><i class="ri-translate"></i> ${escapeHTML(r.target)}</div>
                                    </div>
                                `;
                            });
                            ragText = ragHtml;
                        }
                    } else {
                        ragText = '<span style="color:#94a3b8;">특이사항 없음 (사례 발견되지 않음)</span>';
                    }
                } catch (e) {
                     ragText = (ragRaw && ragRaw !== '[]' && !ragRaw.includes('별도 지적 사항 없음')) ? `<div style="margin-bottom:8px;">${parseMarkdown(escapeHTML(ragRaw))}</div>` : '<span style="color:#94a3b8;">특이사항 없음 (사례 발견되지 않음)</span>';
                }
            } else {
                ragText = '<span style="color:#94a3b8;">특이사항 없음 (사례 발견되지 않음)</span>';
            }
            htmlString += `
                <div class="action-card info">
                    <span class="action-title"><i class="ri-database-2-line"></i> RAG 일관성 참고 내역</span>
                    <div class="action-content">${ragText}</div>
                </div>
            `;

            // 대소문자 카드
            const hasCasingIssue = item.tags.includes('대소문자');
            const casingClass = hasCasingIssue ? 'warn' : 'success';
            const casingTitle = hasCasingIssue ? '<i class="ri-font-size"></i> 대소문자 표기 오류 (Warning)' : '<i class="ri-font-size"></i> 대소문자 점검 (Pass)';
            const casingContent = (item.casingCheck && item.casingCheck.trim() !== '') ? escapeHTML(item.casingCheck) : '특이사항 없음';
            htmlString += `
                <div class="action-card ${casingClass}">
                    <span class="action-title">${casingTitle}</span>
                    <div class="action-content">${casingContent}</div>
                    ${(hasCasingIssue && item.casingSuggestion) ? `<div class="suggestion-box"><strong>제안:</strong> ${escapeHTML(item.casingSuggestion)}</div>` : ''}
                </div>
            `;

            // 용어집 카드
            const hasGlossaryIssue = item.tags.includes('용어집');
            const glossaryClass = hasGlossaryIssue ? 'error' : 'success';
            const glossaryTitle = hasGlossaryIssue ? '<i class="ri-book-read-line"></i> 용어집 불일치 (Fail)' : '<i class="ri-book-read-line"></i> 용어집 점검 (Pass)';
            const glossaryContent = (item.glossaryCheck && item.glossaryCheck.trim() !== '') ? escapeHTML(item.glossaryCheck) : '특이사항 없음';
            htmlString += `
                <div class="action-card ${glossaryClass}">
                    <span class="action-title">${glossaryTitle}</span>
                    <div class="action-content">${glossaryContent}</div>
                </div>
            `;

            // AI 평가 카드
            let geminiClass = item.tags.includes('AI 오류') ? 'error' : (item.tags.includes('AI 추천') ? 'warn' : 'info');
            if(item.status === 'pass' && !item.tags.includes('AI 추천')) geminiClass = 'success';
            
            let aiContent = '';
            if (item.geminiQa && item.geminiQa.trim() !== '') {
                try {
                    const aiData = JSON.parse(item.geminiQa);
                    
                    if (aiData.evaluation && Array.isArray(aiData.evaluation)) {
                        aiData.evaluation.forEach(ev => {
                            if (!ev.comment || ev.comment === '' || ev.category === '') return;
                            aiContent += `
                                <div style="background: rgba(255,255,255,0.6); border-left: 3px solid #8b5cf6; padding: 12px 14px; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 0.95rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02); line-height: 1.5;">
                                    <strong>${escapeHTML(ev.category)}:</strong> ${parseMarkdown(escapeHTML(ev.comment))}
                                </div>
                            `;
                        });
                    }
                    if (aiData.grade === "Excellent" || aiData.is_excellent === true) {
                        aiContent += `<div style="margin-top: 10px; color: var(--success-color); font-weight: bold;"><i class="ri-check-double-line"></i> 종합 평가: 우수 <span>(직역 없이 자연스러운 현지화)</span></div>`;
                    } else if (aiData.grade === "Good") {
                        aiContent += `<div style="margin-top: 10px; color: #d97706; font-weight: bold;"><i class="ri-check-line"></i> 종합 평가: 양호 <span>(경미한 개선 여지)</span></div>`;
                    } else if (aiData.grade === "Needs Revision" || aiData.is_excellent === false) {
                        aiContent += `<div style="margin-top: 10px; color: #dc2626; font-weight: bold;"><i class="ri-error-warning-line"></i> 종합 평가: 수정 필요</div>`;
                    }
                    if (aiData.suggested_fix && aiData.suggested_fix.trim() !== '') {
                        let suggClass = "warn";
                        aiContent += `<div style="margin-top: 10px; padding: 10px 14px; background: rgba(245, 158, 11, 0.08); border-radius: 6px; color: #92400e; font-size:0.95rem; font-weight: 500;">
                                        <strong><i class="ri-edit-2-line"></i> 자연스러운 문장 다듬기 제안:</strong><br/>
                                        <div style="margin-top: 6px; color: #78350f;">${escapeHTML(aiData.suggested_fix)}</div>
                                      </div>`;
                    }
                } catch(e) {
                    let rawQa = item.geminiQa.trim();
                    rawQa = rawQa.replace(/\s+-\s/g, '\n- '); 
                    let lines = rawQa.split('\n');
                    let fallbackHtml = '';
                    lines.forEach(line => {
                        line = line.trim();
                        if (!line) return;
                        if (line.startsWith('- ') || line.startsWith('* ')) {
                            let text = line.substring(2);
                            text = parseMarkdown(escapeHTML(text));
                            fallbackHtml += `<div style="background: rgba(255,255,255,0.6); border-left: 3px solid #8b5cf6; padding: 12px 14px; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 0.95rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02); line-height: 1.5;">${text}</div>`;
                        } else {
                            fallbackHtml += `<div style="margin-bottom: 10px; font-size: 0.95rem; padding: 0 4px; color:#475569; font-weight: 500;">${parseMarkdown(escapeHTML(line))}</div>`;
                        }
                    });
                    aiContent = fallbackHtml;
                }
            } else {
                aiContent = '<span style="color:#94a3b8;">AI 리뷰 결과 없음</span>';
            }
            
            htmlString += `
                <div class="action-card ${geminiClass}" style="background: linear-gradient(to right, rgba(139, 92, 246, 0.05), rgba(139, 92, 246, 0.01));">
                    <span class="action-title" style="color: #6d28d9;"><i class="ri-robot-2-line"></i> AI 상세 검수 코멘트</span>
                    <div class="action-content" style="padding-top: 10px;">${aiContent}</div>
                </div>
            `;

            htmlString += `</div></div></div></div>`;
            itemsList.insertAdjacentHTML('beforeend', htmlString);
        });
    }

    // --- Search Params Auto Load ---
    const urlParams = new URLSearchParams(window.location.search);
    const fileUrl = urlParams.get('file');
    if (fileUrl) {
        fetch(fileUrl)
            .then(response => {
                if(!response.ok) throw new Error("HTTP error " + response.status);
                return response.text();
            })
            .then(text => {
                parseTxtContent(text);
                renderDashboard();
                renderItems();
                renderSidebar();
                initScrollObserver();
                uploadSection.style.display = 'none';
                dashboardSection.style.display = 'block';
                reportContent.style.display = 'block';
                sidebar.style.display = 'block';
                exportBtn.style.display = 'flex';
            })
            .catch(err => {
                alert("리포트를 로드하는데 실패했습니다: " + err.message);
            });
    }

    // --- Scroll Observer ---
    function initScrollObserver() {
        const observerOptions = {
            root: null,
            rootMargin: '-20% 0px -60% 0px',
            threshold: 0
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    updateActiveNavLink(id);
                }
            });
        }, observerOptions);

        // Observe Dashboard
        observer.observe(document.getElementById('dashboard'));

        // Observe all section headers
        document.querySelectorAll('.scroll-target[id^="section-"]').forEach((section) => {
            observer.observe(section);
        });
    }

    function updateActiveNavLink(id) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-target') === id) {
                link.classList.add('active');
            }
        });
    }

    // --- Format Helpers ---
    function escapeHTML(str) {
        if (!str) return "";
        return str.replace(/[&<>'"]/g, t => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[t] || t));
    }
    function parseMarkdown(str) {
        if (!str) return "";
        let res = str;
        // Basic bold
        res = res.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        return res;
    }

    // --- Export PDF (Restored html2pdf/Canvas snapshot method) ---
    exportBtn.addEventListener('click', () => {
        const contentArea = document.querySelector('.content-area');
        const header = document.querySelector('.app-header');
        const sidebar = document.getElementById('sidebar');

        const isDark = body.getAttribute('data-theme') === 'dark';
        if (isDark) body.removeAttribute('data-theme');

        const origSidebarDisplay = sidebar.style.display;
        const origHeaderDisplay = header.style.display;
        const origWidth = contentArea.style.width;
        const origMargin = contentArea.style.margin;

        // 화면 요소 감추기 및 모바일 뷰(.pdf-mode) 강제 적용
        sidebar.style.display = 'none';
        header.style.display = 'none';
        
        contentArea.classList.add('pdf-mode'); 
        contentArea.style.width = '100%'; 
        contentArea.style.maxWidth = '900px';
        contentArea.style.margin = '0 auto';
        
        const opt = {
            margin: 0, // 상하 여백 제거
            filename: 'Translation_Review_Report.pdf',
            image: { type: 'jpeg', quality: 1.0 },
            html2canvas: { scale: 2, useCORS: true }, 
            pagebreak: { mode: ['avoid-all', 'css', 'legacy'] },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };

        html2pdf().set(opt).from(contentArea).save().then(() => {
            if (isDark) body.setAttribute('data-theme', 'dark');
            sidebar.style.display = origSidebarDisplay;
            header.style.display = origHeaderDisplay;
            contentArea.classList.remove('pdf-mode');
            contentArea.style.width = origWidth;
            contentArea.style.maxWidth = '';
            contentArea.style.margin = origMargin;
        });
    });
});
