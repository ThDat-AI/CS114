const API_URL = 'http://127.0.0.1:8080';
const HISTORY_KEY = 'news_classifier_history';

document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // UI Elements
    const urlInput = document.getElementById('url-input');
    const predictUrlBtn = document.getElementById('predict-url-btn');

    const titleInput = document.getElementById('title-input');
    const bodyInput = document.getElementById('body-input');
    const predictTextBtn = document.getElementById('predict-text-btn');

    // Result Elements
    const resultContainer = document.getElementById('result-container');
    const loadingEl = document.getElementById('loading');
    const resultContent = document.getElementById('result-content');
    const resultEmpty = document.getElementById('result-empty');
    const predictionBadge = document.getElementById('prediction-badge');
    const confidenceText = document.getElementById('confidence-text');
    const warningMsg = document.getElementById('warning-msg');
    const errorMsg = document.getElementById('error-msg');
    const extractedTitle = document.getElementById('extracted-title');
    const extractedBody = document.getElementById('extracted-body');
    const extractedEmpty = document.getElementById('extracted-empty');
    const extractedContent = document.getElementById('extracted-content');

    // History & Chart Elements
    const historyList = document.getElementById('history-list');
    const chartSection = document.getElementById('chart-section');
    const chartContainer = document.getElementById('chart-container');

    // Fetch available models
    fetch(`${API_URL}/models`)
        .then(res => res.json())
        .then(data => {
            modelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                modelSelect.appendChild(option);
            });
            modelSelect.disabled = false;
        })
        .catch(err => {
            console.error('Failed to load models:', err);
            modelSelect.innerHTML = '<option value="">Error loading models</option>';
        });

    // Left Tab switching
    const leftTabBtns = document.querySelectorAll('.card:not(#result-container) .tab-btn');
    const leftTabContents = document.querySelectorAll('.card:not(#result-container) .tab-content');
    leftTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            leftTabBtns.forEach(b => b.classList.remove('active'));
            leftTabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });

    // Right Tab switching
    const rightTabBtns = document.querySelectorAll('.right-tab-btn');
    const rightTabContents = document.querySelectorAll('.right-tab-content');
    rightTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            rightTabBtns.forEach(b => b.classList.remove('active'));
            rightTabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });

    const CHART_LABELS = {
        'business_finance': 'BUSINESS',
        'culture_lifestyle': 'LIFESTYLE',
        'film': 'FILM',
        'food': 'FOOD',
        'music': 'MUSIC',
        'sport': 'SPORTS',
        'tech_science_games': 'TECH/GAME',
        'world_environment': 'WORLD/ENV'
    };

    // --- Chart function ---
    const renderScoresChart = (scores, isSvm, predictedLabel) => {
        if (!scores) {
            chartSection.style.display = 'none';
            return;
        }
        
        const labels = Object.keys(scores);
        if (labels.length === 0) {
            chartSection.style.display = 'none';
            return;
        }

        chartSection.style.display = 'block';
        chartContainer.innerHTML = '';
        
        let minVal = 0;
        let maxVal = 0;
        let range = 1;
        if (isSvm) {
            const vals = Object.values(scores);
            minVal = Math.min(...vals);
            maxVal = Math.max(...vals);
            range = maxVal - minVal;
        }

        // Sort items descending by score
        const sortedItems = labels
            .map(label => ({ label, score: scores[label] }))
            .sort((a, b) => b.score - a.score);

        // Draw vertical columns
        sortedItems.forEach(item => {
            const colEl = document.createElement('div');
            colEl.className = 'histogram-col';
            if (item.label.toLowerCase() === predictedLabel.toLowerCase()) {
                colEl.classList.add('predicted');
            }

            const chartLabel = CHART_LABELS[item.label.toLowerCase()] || item.label.substring(0, 7).toUpperCase();
            const formattedLabel = item.label.replace(/_/g, ' ').toUpperCase();

            let fillPercent = 0;
            let valText = '';
            if (isSvm) {
                fillPercent = range > 0 ? ((item.score - minVal) / range) * 100 : 0;
                valText = item.score.toFixed(2);
            } else {
                fillPercent = item.score * 100;
                valText = `${Math.round(fillPercent)}%`;
            }

            colEl.innerHTML = `
                <div class="histogram-bar-wrapper" style="height: 0%" title="${formattedLabel}: ${valText}">
                    <div class="histogram-value-top">${valText}</div>
                </div>
                <div class="histogram-label" title="${formattedLabel}">${chartLabel}</div>
            `;

            chartContainer.appendChild(colEl);
            
            // Animate height upwards
            setTimeout(() => {
                const bar = colEl.querySelector('.histogram-bar-wrapper');
                if (bar) bar.style.height = `${Math.max(5, fillPercent)}%`;
            }, 50);
        });
    };

    // --- History functions ---
    const getHistory = () => {
        try {
            const data = localStorage.getItem(HISTORY_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error('Failed to load scan history:', e);
            return [];
        }
    };

    const saveToHistory = (item) => {
        const history = getHistory();
        // Remove duplicates of same title & body to keep list clean
        const duplicateIndex = history.findIndex(h => h.title === item.title && h.body === item.body && h.isSuccess === item.isSuccess);
        if (duplicateIndex !== -1) {
            history.splice(duplicateIndex, 1);
        }
        history.unshift(item);
        if (history.length > 10) {
            history.pop();
        }
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        renderHistory();
    };

    const deleteFromHistory = (index) => {
        const history = getHistory();
        history.splice(index, 1);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        renderHistory();
    };

    const renderHistory = () => {
        const history = getHistory();
        if (history.length === 0) {
            historyList.innerHTML = `<div class="history-empty">No classification history yet. Run a scan to see it here!</div>`;
            return;
        }

        historyList.innerHTML = '';
        history.forEach((item, index) => {
            const itemEl = document.createElement('div');
            itemEl.className = 'history-item';
            
            // Subtle random rotation for Y2K sticker feel
            const rotation = ((index % 3 - 1) * 0.8).toFixed(1);
            itemEl.style.transform = `rotate(${rotation}deg)`;

            const formattedPrediction = item.prediction
                .toString()
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

            itemEl.innerHTML = `
                <div class="history-item-header">
                    <span class="history-item-title">${item.title || 'Untitled Scan'}</span>
                    <button class="history-item-delete" data-index="${index}" title="Delete scan">X</button>
                </div>
                <div class="history-item-meta">
                    <span class="history-item-badge ${item.isSuccess ? '' : 'error'}">${item.isSuccess ? formattedPrediction : 'Failed'}</span>
                    <span class="history-item-model">${item.modelName}</span>
                    ${item.distance !== undefined && item.distance !== null ? `<span class="history-item-confidence">Dist: ${item.distance.toFixed(3)}</span>` : ''}
                    ${item.confidence !== undefined && item.confidence !== null ? `<span class="history-item-confidence">${(item.confidence * 100).toFixed(0)}% Conf</span>` : ''}
                </div>
            `;

            // Delete action
            const deleteBtn = itemEl.querySelector('.history-item-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteFromHistory(index);
            });

            // Restore action
            itemEl.addEventListener('click', () => {
                if (item.url) {
                    const urlTabBtn = document.querySelector('[data-tab="url-tab"]');
                    urlTabBtn.click();
                    urlInput.value = item.url;
                } else {
                    const textTabBtn = document.querySelector('[data-tab="text-tab"]');
                    textTabBtn.click();
                    titleInput.value = item.title || '';
                    bodyInput.value = item.body || '';
                }

                modelSelect.value = item.modelId;

                // Load old scan outputs directly
                loadingEl.style.display = 'none';
                resultEmpty.style.display = 'none';
                resultContent.style.display = 'block';
                warningMsg.style.display = 'none';
                errorMsg.style.display = 'none';

                // Automatically switch to 'Active Scan' tab
                const activeScanTabBtn = document.querySelector('[data-tab="active-tab"]');
                if (activeScanTabBtn) activeScanTabBtn.click();

                if (item.isSuccess) {
                    predictionBadge.textContent = formattedPrediction;
                    predictionBadge.className = 'badge success';
                    const isSvm = item.distance !== undefined && item.distance !== null;
                    if (isSvm) {
                        confidenceText.textContent = `Distance: ${item.distance.toFixed(4)}`;
                    } else if (item.confidence !== undefined && item.confidence !== null) {
                        confidenceText.textContent = `Confidence: ${(item.confidence * 100).toFixed(1)}%`;
                    } else {
                        confidenceText.textContent = '';
                    }

                    if (item.warning) {
                        warningMsg.textContent = item.warning;
                        warningMsg.style.display = 'block';
                    }
                    
                    // Render score distribution chart
                    renderScoresChart(item.scores, isSvm, item.prediction);
                } else {
                    errorMsg.textContent = item.errorMessage || 'Failed to process';
                    errorMsg.style.display = 'block';
                    predictionBadge.textContent = 'Failed';
                    predictionBadge.className = 'badge danger';
                    confidenceText.textContent = '';
                    chartSection.style.display = 'none';
                }

                extractedTitle.textContent = item.title || 'No title extracted';
                extractedBody.textContent = item.body || 'No content extracted';

                if (item.title || item.body) {
                    extractedEmpty.style.display = 'none';
                    extractedContent.style.display = 'block';
                } else {
                    extractedEmpty.style.display = 'block';
                    extractedContent.style.display = 'none';
                }

                resultContainer.scrollIntoView({ behavior: 'smooth' });
            });

            historyList.appendChild(itemEl);
        });
    };

    // Render history on page load
    renderHistory();

    // Display state helpers
    const showLoading = () => {
        // Automatically switch right card to 'Active Scan' tab
        const activeScanTabBtn = document.querySelector('[data-tab="active-tab"]');
        if (activeScanTabBtn) activeScanTabBtn.click();

        loadingEl.style.display = 'block';
        resultEmpty.style.display = 'none';
        resultContent.style.display = 'none';
        warningMsg.style.display = 'none';
        errorMsg.style.display = 'none';
        chartSection.style.display = 'none';
        predictionBadge.className = 'badge';
        predictionBadge.textContent = '';
        
        extractedEmpty.style.display = 'none';
        extractedContent.style.display = 'none';
    };

    const showResult = (data) => {
        loadingEl.style.display = 'none';
        resultContent.style.display = 'block';
        resultEmpty.style.display = 'none';

        const modelId = modelSelect.value;
        const modelName = modelSelect.options[modelSelect.selectedIndex]?.text || modelId;

        if (data.result.error) {
            errorMsg.textContent = data.result.error;
            errorMsg.style.display = 'block';
            predictionBadge.textContent = 'Error';
            predictionBadge.className = 'badge danger';
            confidenceText.textContent = '';
            chartSection.style.display = 'none';

            saveToHistory({
                title: data.title || 'Error processing article',
                body: data.body || '',
                prediction: 'Error',
                confidence: undefined,
                distance: undefined,
                isSuccess: false,
                errorMessage: data.result.error,
                modelId,
                modelName,
                url: urlInput.value.trim() && document.getElementById('url-tab').classList.contains('active') ? urlInput.value.trim() : null,
                timestamp: Date.now()
            });
        } else {
            const prediction = data.result.prediction;
            const formattedPrediction = prediction
                .toString()
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

            predictionBadge.textContent = formattedPrediction;
            predictionBadge.className = 'badge success';

            const isSvm = data.result.distance !== undefined && data.result.distance !== null;
            if (isSvm) {
                confidenceText.textContent = `Distance: ${data.result.distance.toFixed(4)}`;
            } else if (data.result.confidence !== undefined && data.result.confidence !== null) {
                confidenceText.textContent = `Confidence: ${(data.result.confidence * 100).toFixed(1)}%`;
            } else {
                confidenceText.textContent = '';
            }

            if (data.result.warning) {
                warningMsg.textContent = data.result.warning;
                warningMsg.style.display = 'block';
            }

            // Render score distribution chart
            renderScoresChart(data.result.scores, isSvm, prediction);

            saveToHistory({
                title: data.title || 'Untitled Article',
                body: data.body || '',
                prediction: prediction,
                confidence: data.result.confidence,
                distance: data.result.distance,
                scores: data.result.scores,
                isSuccess: true,
                warning: data.result.warning,
                modelId,
                modelName,
                url: urlInput.value.trim() && document.getElementById('url-tab').classList.contains('active') ? urlInput.value.trim() : null,
                timestamp: Date.now()
            });
        }

        extractedTitle.textContent = data.title || 'No title extracted';
        extractedBody.textContent = data.body || 'No content extracted';

        if (data.title || data.body) {
            extractedEmpty.style.display = 'none';
            extractedContent.style.display = 'block';
        } else {
            extractedEmpty.style.display = 'block';
            extractedContent.style.display = 'none';
        }
    };

    const showError = (message) => {
        loadingEl.style.display = 'none';
        resultEmpty.style.display = 'none';
        resultContent.style.display = 'block';
        errorMsg.textContent = message;
        errorMsg.style.display = 'block';
        predictionBadge.textContent = 'Failed';
        predictionBadge.className = 'badge danger';
        confidenceText.textContent = '';
        chartSection.style.display = 'none';
        extractedTitle.textContent = '';
        extractedBody.textContent = '';
        extractedEmpty.style.display = 'block';
        extractedContent.style.display = 'none';

        const modelId = modelSelect.value;
        const modelName = modelSelect.options[modelSelect.selectedIndex]?.text || modelId;

        saveToHistory({
            title: 'Failed to process',
            body: message,
            prediction: 'Failed',
            confidence: undefined,
            distance: undefined,
            isSuccess: false,
            errorMessage: message,
            modelId,
            modelName,
            url: urlInput.value.trim() && document.getElementById('url-tab').classList.contains('active') ? urlInput.value.trim() : null,
            timestamp: Date.now()
        });
    };

    // Predict URL
    predictUrlBtn.addEventListener('click', async () => {
        const url = urlInput.value.trim();
        const modelId = modelSelect.value;

        if (!url) {
            alert('Please enter a valid URL.');
            return;
        }

        showLoading();

        try {
            const response = await fetch(`${API_URL}/predict_url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url, model_id: modelId })
            });

            if (!response.ok) {
                let errorMessage = 'Failed to process URL';
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorMessage;
                } catch (err) {
                    const text = await response.text();
                    if (text) errorMessage = text;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            showResult(data);
        } catch (error) {
            showError(error.message);
        }
    });

    // Predict Text
    predictTextBtn.addEventListener('click', async () => {
        const title = titleInput.value.trim();
        const body = bodyInput.value.trim();
        const modelId = modelSelect.value;

        if (!title && !body) {
            alert('Please enter title or body.');
            return;
        }

        showLoading();

        try {
            const response = await fetch(`${API_URL}/predict_text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title, body, model_id: modelId })
            });

            if (!response.ok) {
                throw new Error('Failed to process text');
            }

            const data = await response.json();
            showResult(data);
        } catch (error) {
            showError(error.message);
        }
    });
});
