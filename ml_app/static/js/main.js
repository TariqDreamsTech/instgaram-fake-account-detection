// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');

    // Activate button
    if (event && event.target) {
        event.target.classList.add('active');
    } else {
        // Find button by tab name
        document.querySelectorAll('.tab-btn').forEach(btn => {
            if (btn.getAttribute('onclick').includes(tabName)) {
                btn.classList.add('active');
            }
        });
    }

    // Load metrics if metrics tab
    if (tabName === 'metrics') {
        loadMetrics();
    }

    // Load dataset if dataset tab
    if (tabName === 'dataset') {
        loadDatasetInfo();
    }
}

// Training functions removed - models are already trained

// Metrics Functions
function showMetrics(type) {
    document.querySelectorAll('.metrics-content').forEach(div => {
        div.style.display = 'none';
    });

    document.querySelectorAll('.metrics-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    event.target.classList.add('active');

    if (type === 'basic') {
        document.getElementById('basic-metrics').style.display = 'block';
        loadBasicMetrics();
    } else {
        document.getElementById('advanced-metrics').style.display = 'block';
        loadAdvancedMetrics();
    }
}

async function loadMetrics() {
    loadBasicMetrics();
}

async function loadBasicMetrics() {
    const div = document.getElementById('basic-metrics');
    div.innerHTML = '<div class="loading">Loading basic models metrics...</div>';

    try {
        const response = await fetch('/metrics/basic');
        const data = await response.json();

        if (response.ok) {
            displayMetrics(data, 'basic');
        } else {
            div.innerHTML = `<div class="status-error">${data.detail || 'Failed to load metrics'}</div>`;
        }
    } catch (error) {
        div.innerHTML = `<div class="status-error">Error loading metrics: ${error.message}</div>`;
    }
}

async function loadAdvancedMetrics() {
    const div = document.getElementById('advanced-metrics');
    div.innerHTML = '<div class="loading">Loading advanced models metrics...</div>';

    try {
        const response = await fetch('/metrics/advanced');
        const data = await response.json();

        if (response.ok) {
            displayMetrics(data, 'advanced');
        } else {
            div.innerHTML = `<div class="status-error">${data.detail || 'Failed to load metrics'}</div>`;
        }
    } catch (error) {
        div.innerHTML = `<div class="status-error">Error loading metrics: ${error.message}</div>`;
    }
}

function displayMetrics(metrics, type) {
    const div = type === 'basic' ? document.getElementById('basic-metrics') :
        document.getElementById('advanced-metrics');

    let html = '<div class="metrics-header">';
    html += `<h3>Best Model: <span style="color: var(--primary-color);">${metrics.best_model || 'N/A'}</span></h3>`;
    html += `<p>Best F1-Score: <strong>${(metrics.best_f1_score || 0).toFixed(4)}</strong></p>`;
    html += '</div>';

    // Add comparison chart
    html += '<div class="chart-container" style="margin: 30px 0;">';
    html += `<canvas id="${type}MetricsChart"></canvas>`;
    html += '</div>';

    html += '<table class="metrics-table">';
    html += '<thead><tr>';
    html += '<th>Model Name</th>';
    html += '<th>Type</th>';
    html += '<th>Accuracy</th>';
    html += '<th>Precision</th>';
    html += '<th>Recall</th>';
    html += '<th>F1-Score</th>';
    if (type === 'advanced') {
        html += '<th>AUC-ROC</th>';
    }
    html += '</tr></thead><tbody>';

    for (const [modelName, modelInfo] of Object.entries(metrics)) {
        if (modelName === 'best_model' || modelName === 'best_f1_score' || modelName === 'scaler_file') {
            continue;
        }

        if (typeof modelInfo === 'object' && modelInfo.metrics) {
            const m = modelInfo.metrics;
            html += '<tr>';
            html += `<td><strong>${modelName}</strong></td>`;
            html += `<td>${modelInfo.model_type || 'N/A'}</td>`;
            html += `<td>${(m.accuracy || 0).toFixed(4)}</td>`;
            html += `<td>${(m.precision || 0).toFixed(4)}</td>`;
            html += `<td>${(m.recall || 0).toFixed(4)}</td>`;
            html += `<td><strong style="color: var(--primary-color);">${(m.f1_score || 0).toFixed(4)}</strong></td>`;
            if (type === 'advanced' && m.auc) {
                html += `<td>${m.auc.toFixed(4)}</td>`;
            } else if (type === 'advanced') {
                html += '<td>-</td>';
            }
            html += '</tr>';
        }
    }

    html += '</tbody></table>';
    div.innerHTML = html;

    // Render metrics chart
    setTimeout(() => {
        renderMetricsChart(metrics, type);
    }, 100);
}

function renderMetricsChart(metrics, type) {
    const ctx = document.getElementById(`${type}MetricsChart`);
    if (!ctx) return;

    const models = [];
    const accuracies = [];
    const f1Scores = [];
    const precisions = [];
    const recalls = [];

    for (const [modelName, modelInfo] of Object.entries(metrics)) {
        if (modelName === 'best_model' || modelName === 'best_f1_score' || modelName === 'scaler_file') {
            continue;
        }

        if (typeof modelInfo === 'object' && modelInfo.metrics) {
            models.push(modelName);
            accuracies.push(modelInfo.metrics.accuracy || 0);
            f1Scores.push(modelInfo.metrics.f1_score || 0);
            precisions.push(modelInfo.metrics.precision || 0);
            recalls.push(modelInfo.metrics.recall || 0);
        }
    }

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'Accuracy',
                    data: accuracies,
                    backgroundColor: 'rgba(99, 102, 241, 0.2)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(99, 102, 241, 1)'
                },
                {
                    label: 'F1-Score',
                    data: f1Scores,
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(16, 185, 129, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(16, 185, 129, 1)'
                },
                {
                    label: 'Precision',
                    data: precisions,
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(139, 92, 246, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(139, 92, 246, 1)'
                },
                {
                    label: 'Recall',
                    data: recalls,
                    backgroundColor: 'rgba(245, 158, 11, 0.2)',
                    borderColor: 'rgba(245, 158, 11, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(245, 158, 11, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(245, 158, 11, 1)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cbd5e1',
                        font: {
                            size: 12,
                            family: "'Poppins', sans-serif"
                        },
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        color: '#cbd5e1',
                        font: {
                            size: 10,
                            family: "'Poppins', sans-serif"
                        }
                    },
                    grid: {
                        color: 'rgba(71, 85, 105, 0.3)'
                    },
                    pointLabels: {
                        color: '#cbd5e1',
                        font: {
                            size: 10,
                            family: "'Poppins', sans-serif"
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

// Dataset Functions
async function loadDatasetInfo() {
    const div = document.getElementById('dataset-content');
    div.innerHTML = '<div class="loading">Loading dataset information...</div>';

    try {
        const response = await fetch('/dataset/info');
        const data = await response.json();

        if (response.ok) {
            displayDatasetInfo(data);
        } else {
            div.innerHTML = `<div class="status-error">${data.detail || 'Failed to load dataset information'}</div>`;
        }
    } catch (error) {
        div.innerHTML = `<div class="status-error">Error loading dataset information: ${error.message}</div>`;
    }
}

function displayDatasetInfo(info) {
    const div = document.getElementById('dataset-content');

    let html = '';

    // Overview Section with Chart
    html += '<div class="dataset-section">';
    html += '<h3><i class="fas fa-info-circle"></i> Dataset Overview</h3>';
    html += '<div class="stats-grid">';
    html += `<div class="stat-card animate-card"><div class="stat-value">${info.total_samples.toLocaleString()}</div><div class="stat-label">Total Samples</div></div>`;
    html += `<div class="stat-card fake animate-card"><div class="stat-value">${info.fake_count.toLocaleString()}</div><div class="stat-label">Fake Accounts (${info.fake_percentage}%)</div></div>`;
    html += `<div class="stat-card real animate-card"><div class="stat-value">${info.real_count.toLocaleString()}</div><div class="stat-label">Real Accounts (${info.real_percentage}%)</div></div>`;
    html += `<div class="stat-card animate-card"><div class="stat-value">${info.dataset_version}</div><div class="stat-label">Dataset Version</div></div>`;
    html += '</div>';

    // Distribution Pie Chart
    html += '<div class="chart-container">';
    html += '<canvas id="distributionChart"></canvas>';
    html += '</div>';
    html += '</div>';

    // Features Section
    html += '<div class="dataset-section">';
    html += '<h3><i class="fas fa-list"></i> Dataset Features</h3>';
    html += '<div class="features-list">';
    info.feature_list.forEach(feature => {
        html += `<div class="feature-item">`;
        html += `<div class="feature-name"><code>${feature.name}</code></div>`;
        html += `<div class="feature-desc">${feature.description}</div>`;
        html += `</div>`;
    });
    html += '</div>';
    html += '</div>';

    // Categorical Statistics
    html += '<div class="dataset-section">';
    html += '<h3><i class="fas fa-chart-pie"></i> Categorical Features Distribution</h3>';

    // Profile Picture
    html += '<div class="categorical-stat">';
    html += '<h4>Profile Picture</h4>';
    html += '<div class="stat-bars">';
    const picTotal = info.categorical_stats.user_has_profil_pic.has_pic + info.categorical_stats.user_has_profil_pic.no_pic;
    const picHas = ((info.categorical_stats.user_has_profil_pic.has_pic / picTotal) * 100).toFixed(1);
    const picNo = ((info.categorical_stats.user_has_profil_pic.no_pic / picTotal) * 100).toFixed(1);
    html += `<div class="stat-bar-item"><span class="bar-label">Has Picture</span><div class="bar"><div class="bar-fill" style="width: ${picHas}%"></div></div><span class="bar-value">${info.categorical_stats.user_has_profil_pic.has_pic.toLocaleString()} (${picHas}%)</span></div>`;
    html += `<div class="stat-bar-item"><span class="bar-label">No Picture</span><div class="bar"><div class="bar-fill" style="width: ${picNo}%"></div></div><span class="bar-value">${info.categorical_stats.user_has_profil_pic.no_pic.toLocaleString()} (${picNo}%)</span></div>`;
    html += '</div>';
    html += '</div>';

    // Private/Public
    html += '<div class="categorical-stat">';
    html += '<h4>Account Privacy</h4>';
    html += '<div class="stat-bars">';
    const privTotal = info.categorical_stats.user_is_private.private + info.categorical_stats.user_is_private.public;
    const privPrivate = ((info.categorical_stats.user_is_private.private / privTotal) * 100).toFixed(1);
    const privPublic = ((info.categorical_stats.user_is_private.public / privTotal) * 100).toFixed(1);
    html += `<div class="stat-bar-item"><span class="bar-label">Private</span><div class="bar"><div class="bar-fill" style="width: ${privPrivate}%"></div></div><span class="bar-value">${info.categorical_stats.user_is_private.private.toLocaleString()} (${privPrivate}%)</span></div>`;
    html += `<div class="stat-bar-item"><span class="bar-label">Public</span><div class="bar"><div class="bar-fill" style="width: ${privPublic}%"></div></div><span class="bar-value">${info.categorical_stats.user_is_private.public.toLocaleString()} (${privPublic}%)</span></div>`;
    html += '</div>';
    html += '</div>';
    html += '</div>';

    // Numeric Features Statistics
    html += '<div class="dataset-section">';
    html += '<h3><i class="fas fa-chart-line"></i> Numeric Features Statistics</h3>';
    html += '<div class="feature-stats-table">';
    html += '<table class="stats-table">';
    html += '<thead><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead>';
    html += '<tbody>';

    Object.entries(info.feature_stats).forEach(([feature, stats]) => {
        html += '<tr>';
        html += `<td><code>${feature}</code></td>`;
        html += `<td>${stats.mean.toFixed(2)}</td>`;
        html += `<td>${stats.median.toFixed(2)}</td>`;
        html += `<td>${stats.std.toFixed(2)}</td>`;
        html += `<td>${stats.min.toFixed(2)}</td>`;
        html += `<td>${stats.max.toFixed(2)}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    html += '</div>';
    html += '</div>';

    // Numeric Features Bar Chart
    html += '<div class="dataset-section">';
    html += '<h3><i class="fas fa-chart-bar"></i> Feature Statistics Visualization</h3>';
    html += '<div class="chart-container">';
    html += '<canvas id="featuresChart"></canvas>';
    html += '</div>';
    html += '</div>';

    div.innerHTML = html;

    // Render charts after DOM is updated
    setTimeout(() => {
        renderDatasetCharts(info);
    }, 100);
}

function renderDatasetCharts(info) {
    // Distribution Pie Chart
    const ctx1 = document.getElementById('distributionChart');
    if (ctx1) {
        new Chart(ctx1, {
            type: 'doughnut',
            data: {
                labels: ['Fake Accounts', 'Real Accounts'],
                datasets: [{
                    data: [info.fake_count, info.real_count],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(16, 185, 129, 0.8)'
                    ],
                    borderColor: [
                        'rgba(239, 68, 68, 1)',
                        'rgba(16, 185, 129, 1)'
                    ],
                    borderWidth: 3,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#cbd5e1',
                            font: {
                                size: 14,
                                family: "'Poppins', sans-serif"
                            },
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#cbd5e1',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function (context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value.toLocaleString()} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true,
                    duration: 2000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    // Features Statistics Bar Chart
    const ctx2 = document.getElementById('featuresChart');
    if (ctx2) {
        const features = Object.keys(info.feature_stats);
        const means = features.map(f => info.feature_stats[f].mean);
        const maxs = features.map(f => info.feature_stats[f].max);

        new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: features.map(f => f.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())),
                datasets: [
                    {
                        label: 'Mean Value',
                        data: means,
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 2,
                        borderRadius: 8
                    },
                    {
                        label: 'Max Value',
                        data: maxs,
                        backgroundColor: 'rgba(139, 92, 246, 0.7)',
                        borderColor: 'rgba(139, 92, 246, 1)',
                        borderWidth: 2,
                        borderRadius: 8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#cbd5e1',
                            font: {
                                size: 13,
                                family: "'Poppins', sans-serif"
                            },
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#cbd5e1',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                        padding: 12
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#cbd5e1',
                            font: {
                                size: 11,
                                family: "'Poppins', sans-serif"
                            }
                        },
                        grid: {
                            color: 'rgba(71, 85, 105, 0.3)'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#cbd5e1',
                            font: {
                                size: 11,
                                family: "'Poppins', sans-serif"
                            }
                        },
                        grid: {
                            color: 'rgba(71, 85, 105, 0.3)'
                        },
                        beginAtZero: true
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }
}

// Input Mode Toggle
function toggleInputMode() {
    const manualMode = document.getElementById('input-mode-manual').checked;
    const manualForm = document.getElementById('prediction-form');
    const urlForm = document.getElementById('url-prediction-form');

    if (manualMode) {
        manualForm.style.display = 'block';
        urlForm.style.display = 'none';
    } else {
        manualForm.style.display = 'none';
        urlForm.style.display = 'block';
    }
}

// URL Prediction Form Handler
document.getElementById('url-prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const profileUrl = formData.get('profile_url');
    const predictionMode = formData.get('url_prediction_mode') || 'basic-best';

    const resultsDiv = document.getElementById('prediction-results');
    resultsDiv.innerHTML = '<div class="loading">Fetching profile data and predicting...</div>';
    resultsDiv.classList.add('show');

    try {
        let url;
        let options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (predictionMode === 'basic-best') {
            url = '/predict/from-url/basic';
        } else if (predictionMode === 'advanced-best') {
            url = '/predict/from-url/advanced';
        } else {
            url = '/predict/from-url/all';
        }

        options.body = JSON.stringify({
            url: profileUrl,
            prediction_mode: predictionMode
        });

        const response = await fetch(url, options);
        const data = await response.json();

        if (response.ok) {
            // Display fetched profile data
            let html = '<div class="profile-info" style="margin-bottom: 20px; padding: 20px; background: var(--dark-card); border-radius: 12px; border: 1px solid var(--border-color);">';
            html += `<h3 style="margin-bottom: 15px; color: var(--primary-color);"><i class="fas fa-user"></i> Profile: @${data.username}</h3>`;
            html += '<div class="profile-stats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">';

            if (data.profile_data) {
                const pd = data.profile_data;
                html += `<div class="stat-item"><strong>Posts:</strong> ${pd.user_media_count.toLocaleString()}</div>`;
                html += `<div class="stat-item"><strong>Followers:</strong> ${pd.user_follower_count.toLocaleString()}</div>`;
                html += `<div class="stat-item"><strong>Following:</strong> ${pd.user_following_count.toLocaleString()}</div>`;
                html += `<div class="stat-item"><strong>Private:</strong> ${pd.user_is_private ? 'Yes' : 'No'}</div>`;
                html += `<div class="stat-item"><strong>Has Profile Pic:</strong> ${pd.user_has_profil_pic ? 'Yes' : 'No'}</div>`;
                html += `<div class="stat-item"><strong>Bio Length:</strong> ${pd.user_biography_length} chars</div>`;
            }
            html += '</div></div>';

            // Display prediction results
            if (data.prediction) {
                // Single prediction
                html += displaySinglePredictionResult(data.prediction, data.username);
            } else if (data.predictions) {
                // Multiple predictions
                html += displayMultiplePredictionResults(data.predictions);
            }

            resultsDiv.innerHTML = html;
        } else {
            throw new Error(data.detail || 'Failed to fetch profile or predict');
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="status-error">Error: ${error.message}</div>`;
    }
});

// Add smooth scroll to results
function scrollToResults() {
    const resultsContainer = document.getElementById('prediction-results') || document.getElementById('results');
    if (resultsContainer) {
        // Add highlight effect before scrolling
        resultsContainer.style.animation = 'glow 1s ease-in-out';

        setTimeout(() => {
            resultsContainer.scrollIntoView({
                behavior: 'smooth',
                block: 'start',
                inline: 'nearest'
            });
        }, 100);

        setTimeout(() => {
            resultsContainer.style.animation = '';
        }, 1100);
    }
}

function displaySinglePredictionResult(result, username) {
    const predictionStr = result.is_fake ? 'FAKE ACCOUNT' : 'REAL ACCOUNT';
    const predictionClass = result.is_fake ? 'fake' : 'real';
    const badgeClass = result.is_fake ? 'badge-fake' : 'badge-real';

    let html = `<div class="result-card ${predictionClass}">`;
    html += '<div class="result-header">';
    html += `<div class="result-title">${result.model_name || 'Best Model'}</div>`;
    html += `<span class="result-badge ${badgeClass}">${predictionStr}</span>`;
    html += '</div>';

    if (result.is_neural_network !== undefined) {
        html += `<p style="margin-bottom: 10px; color: var(--text-secondary);">Type: ${result.is_neural_network ? 'Neural Network' : 'Traditional ML'}</p>`;
    }

    html += '<div class="result-metrics">';
    html += '<div class="metric-item">';
    html += '<div class="metric-label">Confidence</div>';
    html += `<div class="metric-value">${(result.confidence || Math.max(result.fake_probability, result.real_probability)).toFixed(2)}%</div>`;
    html += '</div>';
    html += '<div class="metric-item">';
    html += '<div class="metric-label">Fake Probability</div>';
    html += `<div class="metric-value">${result.fake_probability.toFixed(2)}%</div>`;
    html += '</div>';
    html += '<div class="metric-item">';
    html += '<div class="metric-label">Real Probability</div>';
    html += `<div class="metric-value">${result.real_probability.toFixed(2)}%</div>`;
    html += '</div>';
    html += '</div>';
    html += '</div>';

    return html;
}

function displayMultiplePredictionResults(results) {
    let html = '<h3 style="margin-top: 20px; margin-bottom: 15px;">Prediction Results from All Models</h3>';
    html += '<table class="comparison-table">';
    html += '<thead><tr>';
    html += '<th>Model</th>';
    html += '<th>Type</th>';
    html += '<th>Prediction</th>';
    html += '<th>Confidence</th>';
    html += '<th>Fake %</th>';
    html += '<th>Real %</th>';
    html += '</tr></thead><tbody>';

    results.sort((a, b) => b.confidence - a.confidence);

    results.forEach(result => {
        const predictionStr = result.is_fake ? 'FAKE' : 'REAL';
        const predictionClass = result.is_fake ? 'badge-fake' : 'badge-real';
        const modelType = result.is_neural_network ? 'Neural Network' : 'Traditional ML';

        html += '<tr>';
        html += `<td><strong>${result.model_name}</strong></td>`;
        html += `<td>${modelType}</td>`;
        html += `<td><span class="result-badge ${predictionClass}">${predictionStr}</span></td>`;
        html += `<td>${result.confidence.toFixed(2)}%</td>`;
        html += `<td>${result.fake_probability.toFixed(2)}%</td>`;
        html += `<td>${result.real_probability.toFixed(2)}%</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';

    // Summary
    const fakeCount = results.filter(r => r.is_fake).length;
    const realCount = results.length - fakeCount;
    const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;

    html += '<div class="metrics-header" style="margin-top: 20px;">';
    html += `<p><strong>Summary:</strong> ${fakeCount} models predict FAKE, ${realCount} models predict REAL</p>`;
    html += `<p><strong>Average Confidence:</strong> ${avgConfidence.toFixed(2)}%</p>`;
    html += '</div>';

    return html;
}

// Prediction Functions
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const predictionMode = formData.get('prediction_mode');

    // Get account data
    const accountData = {
        user_media_count: parseInt(formData.get('user_media_count')),
        user_follower_count: parseInt(formData.get('user_follower_count')),
        user_following_count: parseInt(formData.get('user_following_count')),
        user_has_profil_pic: parseInt(formData.get('user_has_profil_pic')),
        user_is_private: parseInt(formData.get('user_is_private')),
        user_biography_length: parseInt(formData.get('user_biography_length')),
        username_length: parseInt(formData.get('username_length')),
        username_digit_count: parseInt(formData.get('username_digit_count')),
    };

    const resultsDiv = document.getElementById('prediction-results');
    resultsDiv.innerHTML = '<div class="loading">Predicting...</div>';
    resultsDiv.classList.add('show');

    try {
        let url;
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (predictionMode === 'basic-best') {
            url = '/predict/basic/best';
            options.body = JSON.stringify(accountData);
        } else if (predictionMode === 'advanced-best') {
            url = '/predict/advanced/best';
            options.body = JSON.stringify(accountData);
        } else if (predictionMode === 'basic-all') {
            url = '/predict/basic/all';
            options.body = JSON.stringify(accountData);
        } else if (predictionMode === 'advanced-all') {
            url = '/predict/advanced/all';
            options.body = JSON.stringify(accountData);
        } else if (predictionMode === 'basic-single') {
            const modelName = document.getElementById('basic-model-dropdown').value;
            if (!modelName) {
                throw new Error('Please select a model');
            }
            url = '/predict/basic/single';
            const formDataToSend = new FormData();
            Object.entries(accountData).forEach(([k, v]) => {
                formDataToSend.append(k, v);
            });
            formDataToSend.append('model_name', modelName);
            options.headers = {};
            options.body = formDataToSend;
        } else if (predictionMode === 'advanced-single') {
            const modelName = document.getElementById('advanced-model-dropdown').value;
            if (!modelName) {
                throw new Error('Please select a model');
            }
            url = '/predict/advanced/single';
            const formDataToSend = new FormData();
            Object.entries(accountData).forEach(([k, v]) => {
                formDataToSend.append(k, v);
            });
            formDataToSend.append('model_name', modelName);
            options.headers = {};
            options.body = formDataToSend;
        }

        const response = await fetch(url, options);
        const data = await response.json();

        if (response.ok) {
            displayPredictionResults(data.data, predictionMode.includes('all'));
        } else {
            throw new Error(data.detail || 'Prediction failed');
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="status-error">Error: ${error.message}</div>`;
    }
});

function displayPredictionResults(results, isMultiple) {
    const resultsDiv = document.getElementById('prediction-results');

    if (isMultiple) {
        // Display multiple results in a table
        let html = '<h3>Prediction Results from All Models</h3>';
        html += '<table class="comparison-table">';
        html += '<thead><tr>';
        html += '<th>Model</th>';
        html += '<th>Type</th>';
        html += '<th>Prediction</th>';
        html += '<th>Confidence</th>';
        html += '<th>Fake %</th>';
        html += '<th>Real %</th>';
        html += '</tr></thead><tbody>';

        results.sort((a, b) => b.confidence - a.confidence);

        results.forEach(result => {
            const predictionStr = result.is_fake ? 'FAKE' : 'REAL';
            const predictionClass = result.is_fake ? 'badge-fake' : 'badge-real';
            const modelType = result.is_neural_network ? 'Neural Network' : 'Traditional ML';

            html += '<tr>';
            html += `<td><strong>${result.model_name}</strong></td>`;
            html += `<td>${modelType}</td>`;
            html += `<td><span class="result-badge ${predictionClass}">${predictionStr}</span></td>`;
            html += `<td>${result.confidence.toFixed(2)}%</td>`;
            html += `<td>${result.fake_probability.toFixed(2)}%</td>`;
            html += `<td>${result.real_probability.toFixed(2)}%</td>`;
            html += '</tr>';
        });

        html += '</tbody></table>';

        // Summary
        const fakeCount = results.filter(r => r.is_fake).length;
        const realCount = results.length - fakeCount;
        const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;

        html += '<div class="metrics-header" style="margin-top: 20px;">';
        html += `<p><strong>Summary:</strong> ${fakeCount} models predict FAKE, ${realCount} models predict REAL</p>`;
        html += `<p><strong>Average Confidence:</strong> ${avgConfidence.toFixed(2)}%</p>`;
        html += '</div>';

        resultsDiv.innerHTML = html;

        // Animate results
        setTimeout(() => {
            const cards = resultsDiv.querySelectorAll('.result-card');
            cards.forEach((card, index) => {
                card.style.animationDelay = `${index * 0.1}s`;
                card.classList.add('animate-card');
            });
            // Scroll to results
            scrollToResults();
        }, 100);
    } else {
        // Display single result
        const result = results;
        const predictionStr = result.is_fake ? 'FAKE ACCOUNT' : 'REAL ACCOUNT';
        const predictionClass = result.is_fake ? 'fake' : 'real';
        const badgeClass = result.is_fake ? 'badge-fake' : 'badge-real';

        let html = `<div class="result-card ${predictionClass}">`;
        html += '<div class="result-header">';
        html += `<div class="result-title">${result.model_name || 'Best Model'}</div>`;
        html += `<span class="result-badge ${badgeClass}">${predictionStr}</span>`;
        html += '</div>';

        if (result.is_neural_network !== undefined) {
            html += `<p style="margin-bottom: 10px; color: var(--text-secondary);">Type: ${result.is_neural_network ? 'Neural Network' : 'Traditional ML'}</p>`;
        }

        html += '<div class="result-metrics">';
        html += '<div class="metric-item">';
        html += '<div class="metric-label">Confidence</div>';
        html += `<div class="metric-value">${(result.confidence || Math.max(result.fake_probability, result.real_probability)).toFixed(2)}%</div>`;
        html += '</div>';
        html += '<div class="metric-item">';
        html += '<div class="metric-label">Fake Probability</div>';
        html += `<div class="metric-value">${result.fake_probability.toFixed(2)}%</div>`;
        html += '</div>';
        html += '<div class="metric-item">';
        html += '<div class="metric-label">Real Probability</div>';
        html += `<div class="metric-value">${result.real_probability.toFixed(2)}%</div>`;
        html += '</div>';
        html += '</div>';
        html += '</div>';

        resultsDiv.innerHTML = html;

        // Animate single result
        setTimeout(() => {
            const card = resultsDiv.querySelector('.result-card');
            if (card) {
                card.classList.add('animate-card');
            }
            // Scroll to results
            scrollToResults();
        }, 100);
    }

    resultsDiv.classList.add('show');
}

// Update model dropdowns when metrics are loaded
document.querySelectorAll('input[name="prediction_mode"]').forEach(radio => {
    radio.addEventListener('change', function () {
        const basicSelect = document.getElementById('basic-model-select');
        const advancedSelect = document.getElementById('advanced-model-select');

        basicSelect.style.display = 'none';
        advancedSelect.style.display = 'none';

        if (this.value === 'basic-single') {
            basicSelect.style.display = 'block';
            loadBasicModelsDropdown();
        } else if (this.value === 'advanced-single') {
            advancedSelect.style.display = 'block';
            loadAdvancedModelsDropdown();
        }
    });
});

async function loadBasicModelsDropdown() {
    const dropdown = document.getElementById('basic-model-dropdown');
    dropdown.innerHTML = '<option value="">Loading...</option>';

    try {
        const response = await fetch('/metrics/basic');
        const data = await response.json();

        dropdown.innerHTML = '<option value="">Select a model...</option>';

        for (const [modelName, modelInfo] of Object.entries(data)) {
            if (modelName === 'best_model' || modelName === 'best_f1_score' || modelName === 'scaler_file') {
                continue;
            }
            if (typeof modelInfo === 'object' && modelInfo.model_file) {
                dropdown.innerHTML += `<option value="${modelName}">${modelName}</option>`;
            }
        }
    } catch (error) {
        dropdown.innerHTML = '<option value="">Error loading models</option>';
    }
}

async function loadAdvancedModelsDropdown() {
    const dropdown = document.getElementById('advanced-model-dropdown');
    dropdown.innerHTML = '<option value="">Loading...</option>';

    try {
        const response = await fetch('/metrics/advanced');
        const data = await response.json();

        dropdown.innerHTML = '<option value="">Select a model...</option>';

        for (const [modelName, modelInfo] of Object.entries(data)) {
            if (modelName === 'best_model' || modelName === 'best_f1_score' || modelName === 'scaler_file') {
                continue;
            }
            if (typeof modelInfo === 'object' && modelInfo.model_file) {
                dropdown.innerHTML += `<option value="${modelName}">${modelName}</option>`;
            }
        }
    } catch (error) {
        dropdown.innerHTML = '<option value="">Error loading models</option>';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    // Load metrics on page load
    loadMetrics();
});

