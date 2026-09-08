document.getElementById('analysis-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI Elements
    const btn = document.getElementById('analyzeBtn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');
    const resultsPanel = document.getElementById('resultsPanel');
    
    // State: Loading
    btn.disabled = true;
    btnText.textContent = 'Analyzing...';
    spinner.classList.remove('hidden');
    resultsPanel.classList.add('hidden');
    
    // Gather Input
    const payload = {
        text: document.getElementById('emailText').value,
        url_length: parseFloat(document.getElementById('urlLength').value),
        subdomain_count: parseInt(document.getElementById('subdomainCount').value),
        https_presence: document.getElementById('httpsPresence').checked ? 1 : 0,
        spf_dkim_failure: document.getElementById('spfFailure').checked ? 1 : 0,
        sender_reputation: parseFloat(document.getElementById('senderRep').value),
        recipient_count: parseInt(document.getElementById('recipientCount').value)
    };
    
    try {
        const response = await fetch('https://spam-detection-api.onrender.com/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error('API Request Failed');
        }
        
        const data = await response.json();
        
        // Render Results
        renderResults(data);
        resultsPanel.classList.remove('hidden');
        
        // Smooth scroll to results on mobile
        if(window.innerWidth < 900) {
            resultsPanel.scrollIntoView({ behavior: 'smooth' });
        }
        
    } catch (error) {
        alert('Failed to connect to the Hybrid ML-DNN API. The server may be starting up (cold start can take ~30s on free tier). Please try again.');
        console.error(error);
    } finally {
        // State: Reset
        btn.disabled = false;
        btnText.textContent = 'Analyze Threat';
        spinner.classList.add('hidden');
    }
});

function renderResults(data) {
    // 1. Badge Classification
    const badge = document.getElementById('classificationBadge');
    badge.textContent = data.classification.toUpperCase();
    badge.className = 'badge'; // reset
    if(data.classification === 'Ham') badge.classList.add('safe');
    else if(data.classification === 'Spam') badge.classList.add('spam');
    else badge.classList.add('phishing');
    
    // 2. Epistemic Uncertainty
    document.getElementById('uncertaintyScore').textContent = data.epistemic_uncertainty.toFixed(4);
    const flag = document.getElementById('humanReviewFlag');
    if(data.requires_human_review) {
        flag.classList.remove('hidden');
    } else {
        flag.classList.add('hidden');
    }
    
    // 3. Class Probabilities Chart
    const probChart = document.getElementById('probChart');
    probChart.innerHTML = '';
    const colors = {'Ham': 'var(--success)', 'Spam': 'var(--warning)', 'Phishing': 'var(--danger)'};
    
    for (const [cls, prob] of Object.entries(data.probabilities)) {
        const pct = (prob * 100).toFixed(1);
        probChart.innerHTML += `
            <div class="bar-row">
                <div class="bar-label">${cls}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: ${pct}%; background: ${colors[cls] || 'var(--primary)'}"></div>
                </div>
                <div class="bar-value">${pct}%</div>
            </div>
        `;
    }
    
    // 4. Gating Weights Chart
    const gateChart = document.getElementById('gatingChart');
    gateChart.innerHTML = '';
    const gateColors = {'Text': '#3b82f6', 'Metadata': '#8b5cf6', 'RATI (Threat Intel)': '#ec4899'};
    
    for (const [branch, weight] of Object.entries(data.gating_weights)) {
        const pct = (weight * 100).toFixed(1);
        gateChart.innerHTML += `
            <div class="bar-row">
                <div class="bar-label" style="width: 140px">${branch}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: ${pct}%; background: ${gateColors[branch]}"></div>
                </div>
                <div class="bar-value">${pct}%</div>
            </div>
        `;
    }
    
    // 5. Explainable AI Breakdown
    const xaiList = document.getElementById('xaiList');
    xaiList.innerHTML = '';
    
    data.xai_breakdown.forEach(item => {
        xaiList.innerHTML += `
            <div class="xai-item">
                <div class="xai-header">
                    <div class="xai-feature">${item.feature}</div>
                    <div class="xai-impact">+${(item.impact * 100).toFixed(0)}% Impact</div>
                </div>
                <div class="xai-desc">${item.description}</div>
            </div>
        `;
    });
}
