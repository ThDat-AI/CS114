const API_URL = 'http://127.0.0.1:8080';

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
    const predictionBadge = document.getElementById('prediction-badge');
    const confidenceText = document.getElementById('confidence-text');
    const warningMsg = document.getElementById('warning-msg');
    const errorMsg = document.getElementById('error-msg');
    const extractedTitle = document.getElementById('extracted-title');
    const extractedBody = document.getElementById('extracted-body');

    // Fetch available models
    fetch(`${API_URL}/models`)
        .then(res => res.json())
        .then(data => {
            modelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                // Models from backend will be fully enabled
                modelSelect.appendChild(option);
            });
            modelSelect.disabled = false;
        })
        .catch(err => {
            console.error('Failed to load models:', err);
            modelSelect.innerHTML = '<option value="">Error loading models</option>';
        });

    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });

    // Display state helpers
    const showLoading = () => {
        resultContainer.style.display = 'block';
        loadingEl.style.display = 'block';
        resultContent.style.display = 'none';
        warningMsg.style.display = 'none';
        errorMsg.style.display = 'none';
        predictionBadge.className = 'badge';
        predictionBadge.textContent = '';
    };

    const showResult = (data) => {
        loadingEl.style.display = 'none';
        resultContent.style.display = 'block';

        if (data.result.error) {
            errorMsg.textContent = data.result.error;
            errorMsg.style.display = 'block';
            predictionBadge.textContent = 'Error';
            predictionBadge.classList.add('danger');
            confidenceText.textContent = '';
        } else {
            const prediction = data.result.prediction;

            const formattedPrediction = prediction
                .toString()
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

            predictionBadge.textContent = formattedPrediction;
            predictionBadge.classList.add('success');

            if (data.result.confidence !== undefined) {
                confidenceText.textContent = `Confidence: ${(data.result.confidence * 100).toFixed(1)}%`;
            } else {
                confidenceText.textContent = '';
            }

            if (data.result.warning) {
                warningMsg.textContent = data.result.warning;
                warningMsg.style.display = 'block';
            }
        }

        extractedTitle.textContent = data.title || 'No title extracted';
        extractedBody.textContent = data.body || 'No content extracted';
    };

    const showError = (message) => {
        loadingEl.style.display = 'none';
        resultContent.style.display = 'block';
        errorMsg.textContent = message;
        errorMsg.style.display = 'block';
        predictionBadge.textContent = 'Failed';
        predictionBadge.classList.add('danger');
        confidenceText.textContent = '';
        extractedTitle.textContent = '';
        extractedBody.textContent = '';
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
