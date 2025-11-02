
// Handle Predict form page
const form = document.getElementById('prediction-form');
if (form) {
    const errorDiv = document.getElementById('error');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const raw = Object.fromEntries(formData.entries());

        // Include checkbox booleans (FormData omits when unchecked)
        raw.alcohol_habit = document.getElementById('alcohol_habit')?.checked || false;

        // Build data object preserving types
        const data = {};
        
        // Numeric fields
        data.failures = Number(raw.failures || 0);
        data.absences = Number(raw.absences || 0);
        data.goout = Number(raw.goout || 0); // 0-20
        
        // String fields (dropdowns)
        data.studytime = raw.studytime || '4-8'; // dropdown: <4, 4-8, 8-12, >12
        data.health = raw.health || 'average'; // dropdown: very_bad, bad, average, good, very_good
        
        // Float field (GPA)
        data.G3 = parseFloat(raw.G3 || 5.0); // GPA 0-10.00
        
        // New fields: family and activity hours
        data.family_hours = parseFloat(raw.family_hours || 0);
        data.activity_hours = parseFloat(raw.activity_hours || 0);
        
        // Keep sex as string
        data.sex = raw.sex || 'M';
        data.age = Number(raw.age || 18);

        // Map alcohol habit -> Dalc/Walc expected by model
        const hasHabit = !!(document.getElementById('alcohol_habit')?.checked);
        data.Dalc = hasHabit ? 1 : 0;
        data.Walc = hasHabit ? 1 : 0;
        data.alcohol_habit = hasHabit;

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Bad response');
            }
            const result = await response.json();
            // Persist to localStorage and navigate to result page
            localStorage.setItem('cw_result', JSON.stringify(result));
            window.location.href = 'result.html';
        } catch (err) {
            console.error('Error:', err);
            if (errorDiv) errorDiv.textContent = 'Error: Could not get a prediction.';
        }
    });
}

// Handle Result page rendering
const riskEl = document.getElementById('risk');
const adviceListEl = document.getElementById('advice-list');
if (riskEl && adviceListEl) {
    const stored = localStorage.getItem('cw_result');
    if (stored) {
        try {
            const { prediction, advice } = JSON.parse(stored);
            const high = Number(prediction) === 1;
            riskEl.textContent = high ? 'High risk of depression' : 'Low risk of depression';
            riskEl.className = 'risk ' + (high ? 'high' : 'low');
            adviceListEl.innerHTML = '';
            (advice || []).forEach((tip) => {
                const li = document.createElement('li');
                li.textContent = tip;
                adviceListEl.appendChild(li);
            });
        } catch {}
    }
}

// Handle resources toggle on result page
const toggleResourcesBtn = document.getElementById('toggle-resources');
const resourcesSection = document.getElementById('resources-section');
if (toggleResourcesBtn && resourcesSection) {
    toggleResourcesBtn.addEventListener('click', () => {
        const isVisible = resourcesSection.style.display !== 'none';
        resourcesSection.style.display = isVisible ? 'none' : 'block';
        toggleResourcesBtn.textContent = isVisible ? 'View resources' : 'Hide resources';
    });
}

// How it works page: render metrics chart
async function renderMetricsChart() {
    const canvas = document.getElementById('modelChart');
    if (!canvas) return;
    const note = document.getElementById('metricsNote');
    try {
        const res = await fetch('http://127.0.0.1:8000/metrics');
        const metrics = await res.json();
        const labels = ['logistic_regression', 'naive_bayes', 'svm_rbf', 'knn'];
        const values = labels.map((k) => metrics[k] || 0).map((v) => typeof v === 'number' ? v : 0);

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: ['Logistic Regression', 'Naive Bayes', 'SVM (RBF)', 'KNN'],
                datasets: [{
                    label: 'Test Accuracy',
                    data: values,
                    backgroundColor: ['#0b5cff', '#22c55e', '#f59e0b', '#ef4444']
                }]
            },
            options: {
                responsive: true,
                scales: { 
                    y: { 
                        min: 0, 
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    } 
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Accuracy: ' + (context.parsed.y * 100).toFixed(2) + '%';
                            }
                        }
                    }
                }
            }
        });
        if (note) note.textContent = 'Metrics computed from studataset.csv using 80/20 train-test split.';
    } catch (e) {
        console.error('Chart error:', e);
        // Show placeholder chart even on error
        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: ['Logistic Regression', 'Naive Bayes', 'SVM (RBF)', 'KNN'],
                datasets: [{
                    label: 'Test Accuracy',
                    data: [0.85, 0.78, 0.84, 0.80],
                    backgroundColor: ['#0b5cff', '#22c55e', '#f59e0b', '#ef4444']
                }]
            },
            options: {
                scales: { y: { min: 0, max: 1 } }
            }
        });
        if (note) note.textContent = 'Using default values. Check API connection.';
    }
}

renderMetricsChart();
