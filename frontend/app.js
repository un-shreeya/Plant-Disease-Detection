document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const resultsPanel = document.getElementById('results-panel');
    const chatInput = document.getElementById('chat-input');
    const sendChatBtn = document.getElementById('send-chat-btn');
    
    let currentImage = null;
    let currentContext = null;

    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        currentImage = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            dropZone.querySelector('.drop-zone-content').style.display = 'none';
            analyzeBtn.disabled = false;
            resultsPanel.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentImage) return;

        analyzeBtn.disabled = true;
        loading.style.display = 'block';
        resultsPanel.style.display = 'none';
        document.getElementById('chat-history').innerHTML = ''; 

        const formData = new FormData();
        formData.append('file', currentImage);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Analysis failed');
            
            const data = await response.json();
            displayResults(data);
            currentContext = data.recommendation;
        } catch (error) {
            alert(error.message || 'Failed to connect to the AI engine.');
        } finally {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        const detectionType = data.detection_type || 'leaf_disease';
        const isLeafDisease = detectionType === 'leaf_disease';

        const rawName = data.class_name || '';
        const displayName = rawName.includes('___')
            ? rawName.split('___').pop().replace(/_/g, ' ')
            : rawName;

        const prefix = detectionType === 'insect_pest' ? '🐛 ' : detectionType === 'not_a_plant' ? '⚠️ ' : '';
        document.getElementById('res-disease').textContent = prefix + displayName;

        const confPct = (data.confidence * 100).toFixed(1);
        document.getElementById('res-confidence').textContent = isLeafDisease ? confPct + '%' : 'Vision AI';

        const sevMeter = document.getElementById('res-sev-text').closest('.severity-meter') ||
                         document.querySelector('.severity-meter');
        if (isLeafDisease && data.severity && data.severity.severity !== 'N/A') {
            if (sevMeter) sevMeter.style.display = '';
            const sevVal = parseFloat(data.severity.percentage || 0).toFixed(1);
            const sevClass = data.severity.severity || 'Unknown';
            document.getElementById('res-sev-text').textContent = `${sevClass} (${sevVal}%)`;
            const fill = document.getElementById('res-sev-fill');
            fill.style.width = `${Math.min(sevVal, 100)}%`;
            fill.style.background = sevVal < 10 ? '#4caf50' : sevVal < 30 ? '#ff9800' : '#f44336';
        } else {
            if (sevMeter) sevMeter.style.display = 'none';
        }

        const gradcamSection = document.getElementById('res-gradcam').closest('div') ||
                               document.getElementById('res-gradcam').parentElement;
        if (isLeafDisease && data.gradcam_overlay) {
            gradcamSection.style.display = '';
            document.getElementById('res-gradcam').src = data.gradcam_overlay;
        } else {
            gradcamSection.style.display = 'none';
        }
        
        const rec = data.recommendation;
        document.getElementById('res-chemical').textContent = rec.chemical_treatment;
        document.getElementById('res-organic').textContent = rec.organic_remedy;
        document.getElementById('res-weather').textContent = rec.weather_precautions;

        const prevList = document.getElementById('res-prevention-list');
        prevList.innerHTML = '';
        (rec.prevention || []).forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            prevList.appendChild(li);
        });

        document.getElementById('chat-context-disease').textContent = rec.disease;

        resultsPanel.style.display = 'block';
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }

    sendChatBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });

    async function sendChatMessage() {
        const msg = chatInput.value.trim();
        if (!msg) return;

        appendMessage(msg, 'user-message');
        chatInput.value = '';

        try {
            const response = await fetch('/api/chatbot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, context: currentContext })
            });
            const data = await response.json();
            appendMessage(data.reply, 'ai-message');
        } catch (error) {
            appendMessage("Sorry, I'm having trouble connecting right now.", 'ai-message');
        }
    }

    function appendMessage(text, className) {
        const chatHistory = document.getElementById('chat-history');
        const div = document.createElement('div');
        div.className = `message ${className}`;
        div.textContent = text;
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    window.switchTab = function(tabName, evt) {
        document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + tabName).style.display = 'block';
        if (evt && evt.target) evt.target.classList.add('active');
    };
});
