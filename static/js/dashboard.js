/**
 * In-Game Cheat Detection System
 * Interactive Telemetry Presets & Dashboard Interactivity
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Telemetry Preset Autofillers
    const presetButtons = document.querySelectorAll('[data-preset]');
    if (presetButtons) {
        presetButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const type = btn.getAttribute('data-preset');
                applyTelemetryPreset(type);
            });
        });
    }

    // 2. Moderator Dashboard Table Filter
    const riskFilter = document.getElementById('riskFilter');
    const searchInput = document.getElementById('sessionSearch');
    if (riskFilter || searchInput) {
        if (riskFilter) riskFilter.addEventListener('change', filterTable);
        if (searchInput) searchInput.addEventListener('input', filterTable);
    }
});

/**
 * Fetches an authentic row from final_dataset.csv via backend API and populates the form
 */
async function loadRealSample(type) {
    try {
        const response = await fetch(`/api/sample-row?type=${type}`);
        const data = await response.json();
        if (data.status === 'success') {
            const row = data.row;
            for (const key in row) {
                const el = document.getElementById(key);
                if (el) {
                    el.value = row[key];
                }
            }
            const labelStr = row.cheat === 1 ? "Cheater (cheat=1)" : "Legitimate (cheat=0)";
            showToast(`Loaded authentic row from final_dataset.csv [${labelStr}] (UUID: ${row.uuid.substring(0,8)}...)`, type === 'cheat' ? 'danger' : 'success');
        } else {
            showToast("Failed to load sample row: " + data.message, "danger");
        }
    } catch (err) {
        showToast("Error contacting API: " + err, "danger");
    }
}

/**
 * Injects calibrated telemetry profiles for instant college project demonstrations
 */
function applyTelemetryPreset(type) {
    const fields = {
        yaw: document.getElementById('yaw'),
        pitch: document.getElementById('pitch'),
        delta_yaw: document.getElementById('delta_yaw'),
        delta_pitch: document.getElementById('delta_pitch'),
        accel_yaw: document.getElementById('accel_yaw'),
        accel_pitch: document.getElementById('accel_pitch'),
        target_x: document.getElementById('target_x'),
        target_y: document.getElementById('target_y'),
        target_z: document.getElementById('target_z'),
        position_x: document.getElementById('position_x'),
        position_y: document.getElementById('position_y'),
        position_z: document.getElementById('position_z'),
        sensitivity: document.getElementById('sensitivity'),
        new_sequence: document.getElementById('new_sequence')
    };

    if (!fields.yaw) return;

    if (type === 'legitimate') {
        // Natural human smooth wrist tracking
        fields.yaw.value = (Math.random() * 40 - 20).toFixed(2);
        fields.pitch.value = (Math.random() * 20 - 10).toFixed(2);
        fields.delta_yaw.value = (Math.random() * 8 + 4).toFixed(2);       // Low human delta
        fields.delta_pitch.value = (Math.random() * 5 + 2).toFixed(2);
        fields.accel_yaw.value = (Math.random() * 3 - 1.5).toFixed(2);     // Natural biomechanical inertia
        fields.accel_pitch.value = (Math.random() * 2 - 1).toFixed(2);
        fields.target_x.value = "12.45";
        fields.target_y.value = "1.80";
        fields.target_z.value = "34.12";
        fields.position_x.value = "5.20";
        fields.position_y.value = "1.00";
        fields.position_z.value = "10.40";
        fields.sensitivity.value = "2.0";
        fields.new_sequence.value = "0";
        showToast("Loaded Legitimate Human Telemetry Profile", "success");
    } 
    else if (type === 'aimbot') {
        // Instantaneous robotic crosshair snap
        fields.yaw.value = "45.20";
        fields.pitch.value = "-15.30";
        fields.delta_yaw.value = (Math.random() * 30 + 55).toFixed(2);     // High delta > 55 deg/tick
        fields.delta_pitch.value = (Math.random() * 20 + 32).toFixed(2);
        fields.accel_yaw.value = (Math.random() * 5 + 12).toFixed(2);      // Physics violation acceleration
        fields.accel_pitch.value = (Math.random() * 4 + 7.5).toFixed(2);
        fields.target_x.value = "0.00";
        fields.target_y.value = "0.00";
        fields.target_z.value = "15.00";
        fields.position_x.value = "10.00";
        fields.position_y.value = "1.20";
        fields.position_z.value = "5.00";
        fields.sensitivity.value = "2.5";
        fields.new_sequence.value = "1";
        showToast("Loaded Aggressive Aimbot Snap Telemetry", "danger");
    }
    else if (type === 'spinbot') {
        // 360 degree spinbot continuous anomalous rotation
        fields.yaw.value = "179.80";
        fields.pitch.value = "-88.50";
        fields.delta_yaw.value = "112.40";                                 // Max angular velocity
        fields.delta_pitch.value = "68.20";
        fields.accel_yaw.value = "18.50";                                  // Extreme acceleration
        fields.accel_pitch.value = "10.20";
        fields.target_x.value = "-22.10";
        fields.target_y.value = "12.40";
        fields.target_z.value = "-45.00";
        fields.position_x.value = "-30.00";
        fields.position_y.value = "0.00";
        fields.position_z.value = "-12.00";
        fields.sensitivity.value = "3.0";
        fields.new_sequence.value = "1";
        showToast("Loaded Blatant Spinbot Telemetry", "danger");
    }
}

/**
 * Dynamic filtering for the moderator session queue
 */
function filterTable() {
    const riskVal = document.getElementById('riskFilter')?.value.toUpperCase() || 'ALL';
    const query = document.getElementById('sessionSearch')?.value.toLowerCase() || '';
    const rows = document.querySelectorAll('#sessionTableBody tr');

    rows.forEach(row => {
        const riskBadge = row.querySelector('.risk-cell')?.textContent.trim().toUpperCase() || '';
        const username = row.querySelector('.user-cell')?.textContent.trim().toLowerCase() || '';

        const matchesRisk = (riskVal === 'ALL' || riskBadge.includes(riskVal));
        const matchesQuery = (username.includes(query));

        if (matchesRisk && matchesQuery) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

function showToast(message, type = "info") {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type}`;
    toast.style.position = 'fixed';
    toast.style.bottom = '20px';
    toast.style.right = '20px';
    toast.style.zIndex = '9999';
    toast.style.boxShadow = '0 10px 25px rgba(0,0,0,0.5)';
    toast.innerText = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 3000);
}
