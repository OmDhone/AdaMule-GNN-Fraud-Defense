// AdaMule Cyber War Room & Red vs Blue Heist Arena Logic

const canvas = document.getElementById('networkCanvas');
const ctx = canvas.getContext('2d');

let width, height;
function resizeCanvas() {
    if (!canvas.parentElement) return;
    width = canvas.parentElement.clientWidth;
    height = canvas.parentElement.clientHeight;
    canvas.width = width;
    canvas.height = height;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// --- Graph Model & Physics State ---
const nodes = [
    { id: 0, label: "Victim Acc #1", x: 120, y: 150, vx: 0, vy: 0, type: "victim", acc: "ACC_990112", color: "#38bdf8", radius: 18, kyc: "Tier 2 (Full)", risk: 1.2, flow: "₹45,000 Out", dwell: "N/A", frozen: false },
    { id: 1, label: "Victim Acc #2", x: 120, y: 350, vx: 0, vy: 0, type: "victim", acc: "ACC_990145", color: "#38bdf8", radius: 18, kyc: "Tier 2 (Full)", risk: 0.8, flow: "₹30,000 Out", dwell: "N/A", frozen: false },
    { id: 2, label: "Mule A (Student)", x: 300, y: 220, vx: 0, vy: 0, type: "mule", acc: "ACC_008892", color: "#ef4444", radius: 22, kyc: "Tier 1 (Partial)", risk: 98.4, flow: "₹75k In / ₹72k Out", dwell: "< 8 Mins", frozen: false },
    { id: 3, label: "Mule B (Dormant)", x: 470, y: 200, vx: 0, vy: 0, type: "mule", acc: "ACC_004521", color: "#ef4444", radius: 22, kyc: "Tier 1 (Partial)", risk: 94.2, flow: "₹72k In / ₹70k Out", dwell: "< 14 Mins", frozen: false },
    { id: 4, label: "Sharma Kirana (Merchant)", x: 420, y: 410, vx: 0, vy: 0, type: "merchant", acc: "MERCH_SK889", color: "#10b981", radius: 28, kyc: "Verified GST / Merchant", risk: 2.1, flow: "₹2.4L In / ₹2.1L Out", dwell: "Aggregator Hub", frozen: false },
    { id: 5, label: "Mule C (Layered)", x: 630, y: 230, vx: 0, vy: 0, type: "mule", acc: "ACC_007319", color: "#ef4444", radius: 22, kyc: "Tier 1 (Partial)", risk: 96.7, flow: "₹70k In / ₹68k Out", dwell: "< 11 Mins", frozen: false },
    { id: 6, label: "ATM Cashout Hub", x: 790, y: 240, vx: 0, vy: 0, type: "cashout", acc: "ATM_TERM_04", color: "#f59e0b", radius: 26, kyc: "High-Risk Physical Rail", risk: 99.8, flow: "₹68,000 Withdrawn", dwell: "Cash Conversion", frozen: false },
    { id: 7, label: "Retail User D", x: 280, y: 460, vx: 0, vy: 0, type: "victim", acc: "ACC_124500", color: "#38bdf8", radius: 15, kyc: "Tier 2 (Full)", risk: 0.4, flow: "₹150 Out", dwell: "Retail Purchase", frozen: false },
    { id: 8, label: "Retail User E", x: 570, y: 460, vx: 0, vy: 0, type: "victim", acc: "ACC_124501", color: "#38bdf8", radius: 15, kyc: "Tier 2 (Full)", risk: 0.5, flow: "₹420 Out", dwell: "Retail Purchase", frozen: false }
];

const edges = [
    { from: 0, to: 2, amount: 45000, type: "illicit", label: "₹45k" },
    { from: 1, to: 2, amount: 30000, type: "illicit", label: "₹30k" },
    { from: 2, to: 3, amount: 72000, type: "illicit", label: "₹72k" },
    { from: 2, to: 4, amount: 250, type: "camouflage", label: "₹250" }, // Camouflage to grocery
    { from: 7, to: 4, amount: 150, type: "legitimate", label: "₹150" },
    { from: 8, to: 4, amount: 420, type: "legitimate", label: "₹420" },
    { from: 3, to: 5, amount: 70000, type: "illicit", label: "₹70k" },
    { from: 5, to: 6, amount: 68000, type: "cashout", label: "₹68k" }
];

// Cash Particles
let particles = [];
function initParticles(count = 28) {
    particles = [];
    for (let i = 0; i < count; i++) {
        const e = edges[i % edges.length];
        particles.push({
            edge: e,
            progress: Math.random(),
            speed: 0.003 + Math.random() * 0.005,
            size: e.type === "camouflage" ? 3 : 5
        });
    }
}
initParticles();

// Interaction Variables
let selectedNode = nodes[2]; // Default select Mule A
let draggedNode = null;
let hoveredNode = null;
let mouseX = 0, mouseY = 0;
let legitimacyLens = 100; // 0 = GCN, 100 = AdaMule
let isAudioMuted = false;
let currentMode = "heist"; // 'heist', 'sentinel', 'upi'
let selectedHops = 3;

// --- Mouse Drag & Hover Physics Handling ---
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    nodes.forEach(n => {
        const dx = n.x - mx;
        const dy = n.y - my;
        if (Math.hypot(dx, dy) < n.radius + 6) {
            draggedNode = n;
            selectedNode = n;
            updateDossierHUD(n);
            audio.playClick();
        }
    });
});

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;

    if (draggedNode) {
        draggedNode.x = mouseX;
        draggedNode.y = mouseY;
        draggedNode.vx = 0;
        draggedNode.vy = 0;
    }

    // Hover detection
    hoveredNode = null;
    nodes.forEach(n => {
        const dx = n.x - mouseX;
        const dy = n.y - mouseY;
        if (Math.hypot(dx, dy) < n.radius + 6) {
            hoveredNode = n;
        }
    });
});

window.addEventListener('mouseup', () => {
    draggedNode = null;
});

// --- Gentle Force Physics Step ---
function updatePhysics() {
    if (draggedNode) return;

    // Center gravity
    const cx = width / 2;
    const cy = height / 2;

    nodes.forEach(n => {
        // Pull to center
        n.vx += (cx - n.x) * 0.0002;
        n.vy += (cy - n.y) * 0.0002;

        // Node repulsion
        nodes.forEach(other => {
            if (n === other) return;
            const dx = n.x - other.x;
            const dy = n.y - other.y;
            const dist = Math.hypot(dx, dy) || 1;
            if (dist < 180) {
                const force = (180 - dist) / dist * 0.04;
                n.vx += dx * force;
                n.vy += dy * force;
            }
        });
    });

    // Spring forces on edges
    edges.forEach(e => {
        const u = nodes[e.from];
        const v = nodes[e.to];
        const dx = v.x - u.x;
        const dy = v.y - u.y;
        const dist = Math.hypot(dx, dy) || 1;
        const targetDist = e.type === "camouflage" ? 140 : 160;
        const spring = (dist - targetDist) * 0.001;
        u.vx += dx * spring;
        u.vy += dy * spring;
        v.vx -= dx * spring;
        v.vy -= dy * spring;
    });

    // Apply velocities with damping & bounds
    nodes.forEach(n => {
        n.vx *= 0.88;
        n.vy *= 0.88;
        n.x += n.vx;
        n.y += n.vy;

        // Bounding box
        n.x = Math.max(n.radius + 20, Math.min(width - n.radius - 20, n.x));
        n.y = Math.max(n.radius + 40, Math.min(height - n.radius - 20, n.y));
    });
}

// --- Main Render Loop (60 FPS) ---
let pulseAngle = 0;

function drawGraph() {
    updatePhysics();
    ctx.clearRect(0, 0, width, height);
    pulseAngle += 0.04;

    // Draw Edges
    edges.forEach(e => {
        const u = nodes[e.from];
        const v = nodes[e.to];

        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);

        if (legitimacyLens < 40) {
            // Low Lens = Standard GCN view (baffled by camouflage)
            ctx.strokeStyle = e.type === "camouflage" ? "rgba(56, 189, 248, 0.7)" : "rgba(148, 163, 184, 0.35)";
            ctx.lineWidth = 1.5;
        } else {
            // High Lens = AdaMule unmasks the illicit structuring path
            if (e.type === "illicit" || e.type === "cashout") {
                ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
                ctx.lineWidth = 2.5;
            } else if (e.type === "camouflage") {
                ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 4]);
            } else {
                ctx.strokeStyle = "rgba(16, 185, 129, 0.6)";
                ctx.lineWidth = 1.5;
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Edge label (amount badge)
        const mx = (u.x + v.x) / 2;
        const my = (u.y + v.y) / 2;
        ctx.font = "9px 'Segoe UI', sans-serif";
        ctx.fillStyle = "rgba(148, 163, 184, 0.8)";
        ctx.textAlign = "center";
        ctx.fillText(e.label, mx, my - 6);
    });

    // Animate Cash Particles
    particles.forEach(p => {
        p.progress += p.speed;
        if (p.progress > 1) p.progress = 0;

        const u = nodes[p.edge.from];
        const v = nodes[p.edge.to];
        const px = u.x + (v.x - u.x) * p.progress;
        const py = u.y + (v.y - u.y) * p.progress;

        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, Math.PI * 2);

        if (legitimacyLens >= 50 && (p.edge.type === "illicit" || p.edge.type === "cashout")) {
            ctx.fillStyle = "#ef4444";
            ctx.shadowColor = "#ef4444";
            ctx.shadowBlur = 12;
        } else if (p.edge.type === "camouflage") {
            ctx.fillStyle = "#f59e0b";
            ctx.shadowColor = "#f59e0b";
            ctx.shadowBlur = 8;
        } else {
            ctx.fillStyle = "#38bdf8";
            ctx.shadowColor = "#38bdf8";
            ctx.shadowBlur = 6;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
    });

    // Draw Nodes
    nodes.forEach(n => {
        let displayColor = n.color;

        // Legitimacy Lens Interpolation:
        // In GCN (lens < 40), Sharma Kirana (Node 4) is mistakenly colored RED (False Positive!)
        // Mules blend in greyish-blue.
        if (legitimacyLens < 40) {
            if (n.id === 4) displayColor = "#ef4444"; // GCN False Alarm!
            if (n.type === "mule") displayColor = "#94a3b8"; // Mules blend in
        }

        // Frozen Account Overlay
        if (n.frozen) {
            displayColor = "#38bdf8";
        }

        // Pulsing selection ring for selected node
        if (n === selectedNode || n === hoveredNode) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.radius + 10 + Math.sin(pulseAngle) * 3, 0, Math.PI * 2);
            ctx.strokeStyle = n.frozen ? "#38bdf8" : displayColor;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Node Glow Ring
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius + 5, 0, Math.PI * 2);
        ctx.fillStyle = displayColor.replace("#ef4444", "rgba(239, 68, 68, 0.2)")
                                    .replace("#10b981", "rgba(16, 185, 129, 0.2)")
                                    .replace("#38bdf8", "rgba(56, 189, 248, 0.2)")
                                    .replace("#f59e0b", "rgba(245, 158, 11, 0.2)")
                                    .replace("#94a3b8", "rgba(148, 163, 184, 0.2)");
        ctx.fill();

        // Main Node Circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = displayColor;
        ctx.shadowColor = displayColor;
        ctx.shadowBlur = 16;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Frozen Lock Badge
        if (n.frozen) {
            ctx.font = "14px 'Segoe UI', sans-serif";
            ctx.fillStyle = "#fff";
            ctx.textAlign = "center";
            ctx.fillText("❄️", n.x, n.y + 5);
        }

        // Node Label
        ctx.font = (n === selectedNode ? "bold 12px" : "11px") + " 'Segoe UI', sans-serif";
        ctx.fillStyle = "#f8fafc";
        ctx.textAlign = "center";
        ctx.fillText(n.label, n.x, n.y + n.radius + 16);
    });

    requestAnimationFrame(drawGraph);
}
requestAnimationFrame(drawGraph);

// --- Mode Switcher ---
function switchMode(mode) {
    currentMode = mode;
    audio.playClick();

    document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active', 'active-red'));
    document.getElementById('deckHeist').style.display = 'none';
    document.getElementById('deckSentinel').style.display = 'none';
    document.getElementById('deckUPI').style.display = 'none';

    if (mode === 'heist') {
        document.getElementById('tabHeist').classList.add('active-red');
        document.getElementById('deckHeist').style.display = 'flex';
        addChatMessage("⚔️ <b>Red Team Heist Console Active:</b> Configure evasion structuring tactics and launch the laundering pipeline.", 'alert');
    } else if (mode === 'sentinel') {
        document.getElementById('tabSentinel').classList.add('active');
        document.getElementById('deckSentinel').style.display = 'flex';
        addChatMessage("🛡️ <b>Blue Team Sentinel Active:</b> Fourier continuous-time encoder & legitimacy regularizer guarding the payment rail.", 'bot');
    } else if (mode === 'upi') {
        document.getElementById('tabUPI').classList.add('active');
        document.getElementById('deckUPI').style.display = 'block';
        addChatMessage("📱 <b>UPI Simulator Active:</b> Send simulated real-time payments across the network graph.", 'bot');
    }
}

// --- Red Team Heist Logic ---
function updateLootAmount(val) {
    const num = parseInt(val);
    document.getElementById('lootDisplay').innerText = `₹${num.toLocaleString('en-IN')}`;
    updateHeistOdds();
}

function setHops(hops, el) {
    audio.playClick();
    selectedHops = hops;
    el.parentElement.querySelectorAll('button').forEach(b => {
        b.style.borderColor = 'var(--panel-border)';
        b.style.color = 'var(--text-main)';
    });
    el.style.borderColor = 'var(--neon-cyan)';
    el.style.color = 'var(--neon-cyan)';
    updateHeistOdds();
}

function updateHeistOdds() {
    const isSmurf = document.getElementById('toggleSmurf').checked;
    const isCamo = document.getElementById('toggleCamouflage').checked;
    const isJitter = document.getElementById('toggleJitter').checked;

    let gcnEvasion = 40;
    if (isSmurf) gcnEvasion += 25;
    if (isCamo) gcnEvasion += 25;
    if (selectedHops >= 3) gcnEvasion += 9;
    gcnEvasion = Math.min(98, gcnEvasion);

    let adamuleEvasion = 5;
    if (selectedHops >= 5) adamuleEvasion += 6;
    if (isJitter) adamuleEvasion += 4;
    adamuleEvasion = Math.min(15, adamuleEvasion);

    document.getElementById('gcnOdds').innerText = `${gcnEvasion}% (${gcnEvasion > 70 ? 'High' : 'Moderate'})`;
    document.getElementById('adamuleOdds').innerText = `${adamuleEvasion}% (Blocked)`;
}

function executeSyndicateHeist() {
    audio.playLaserPulse();
    const loot = document.getElementById('lootRange').value;
    const isSmurf = document.getElementById('toggleSmurf').checked;
    const isCamo = document.getElementById('toggleCamouflage').checked;

    // Dramatic particle swarm
    initParticles(60);

    const modal = document.getElementById("scannerModal");
    const scanStatus = document.getElementById("scanStatus");
    const verdictCard = document.getElementById("verdictCard");
    const verdictContent = document.getElementById("verdictContent");

    modal.classList.add("active");
    scanStatus.innerText = "🚀 INJECTING HEIST PACKETS ACROSS UPI PAYMENT RAIL...";
    verdictCard.style.display = "none";

    setTimeout(() => {
        audio.playAlarm();
        verdictCard.style.display = "block";
        verdictCard.style.border = "2px solid #ef4444";
        verdictContent.innerHTML = `
            <div style="font-size: 2.2rem; margin-bottom: 8px;">⚔️ vs 🛡️</div>
            <h3 style="color: #f87171; margin-bottom: 8px;">SYNDICATE HEIST OUTCOME</h3>
            <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 12px;">
                Loot Injected: <b>₹${parseInt(loot).toLocaleString('en-IN')}</b> | Hops: <b>${selectedHops}</b> | Camouflage: <b>${isCamo ? 'Active' : 'Off'}</b>
            </p>
            <div style="background: rgba(15, 23, 42, 0.9); padding: 12px; border-radius: 8px; font-size: 0.8rem; text-align: left; margin-bottom: 14px; border: 1px solid rgba(255, 255, 255, 0.1);">
                ⚠️ <b>Legacy Bank GCN Result:</b> <span style="color: #f87171;">FOOLED (94% Evasion)</span><br>
                • Legacy model misclassified Sharma Kirana as the primary aggregator hub and locked customer accounts.<br><br>
                🛡️ <b>AdaMule Result:</b> <span style="color: #34d399;">INTERCEPTED & NEUTRALIZED (92% Catch Rate)</span><br>
                • Legitimacy module preserved Sharma Kirana (Zero False Positive).<br>
                • Layering ring unmasked: Funds frozen at Hop #3 before ATM cashout!
            </div>
            <button class="btn-pay" style="width: 100%; background: #ef4444;" onclick="closeModal()">Dismiss & Inspect Graph</button>
        `;

        addChatMessage(`🚨 <b>HEIST SIMULATION:</b> ₹${parseInt(loot).toLocaleString('en-IN')} structuring attack executed. Legacy GCN blinded by camouflage; AdaMule locked exit Node #6 (ATM Hub).`, 'alert');
        addLiveTx(`[HEIST ATTEMPT] ₹${parseInt(loot).toLocaleString('en-IN')} • Syndicate Ring → ATM Cashout • Intercepted by AdaMule`, 'busted');
    }, 1400);
}

// --- Legitimacy Lens Slider ---
function adjustLegitimacyLens(val) {
    legitimacyLens = parseInt(val);
    document.getElementById('lensPercent').innerText = `${legitimacyLens}%`;

    const label = document.getElementById("visionLabel");
    const gcnBadge = document.getElementById("gcnVerdictBadge");
    const adamuleBadge = document.getElementById("adamuleVerdictBadge");

    if (legitimacyLens < 40) {
        label.innerText = "Mode: Standard GCN (Vulnerable to Camouflage & High False Alarms)";
        label.style.color = "#ef4444";
        gcnBadge.innerText = "FOOLED ⚠️";
        adamuleBadge.innerText = "OFFLINE";
    } else {
        label.innerText = "Mode: AdaMule Robust Defense (Legitimacy Regularizer Active)";
        label.style.color = "#10b981";
        gcnBadge.innerText = "FOOLED ⚠️";
        adamuleBadge.innerText = "BUSTED 🛡️";
    }
}

// --- Entity Dossier HUD ---
function updateDossierHUD(node) {
    document.getElementById('dossierName').innerText = node.label;
    document.getElementById('dossierSub').innerText = `Account: ${node.acc}`;
    document.getElementById('dossierKYC').innerText = node.kyc;
    document.getElementById('dossierRisk').innerText = `${node.risk}% (${node.risk > 80 ? 'Critical' : node.risk > 30 ? 'Suspicious' : 'Clean'})`;
    document.getElementById('dossierFlow').innerText = node.flow;
    document.getElementById('dossierDwell').innerText = node.dwell;

    const tag = document.getElementById('dossierTag');
    tag.className = 'dossier-tag';
    if (node.type === 'mule') { tag.classList.add('tag-mule'); tag.innerText = 'MULE SUSPECT'; }
    else if (node.type === 'merchant') { tag.classList.add('tag-merchant'); tag.innerText = 'VERIFIED MERCHANT'; }
    else if (node.type === 'victim') { tag.classList.add('tag-victim'); tag.innerText = 'RETAIL VICTIM'; }
    else if (node.type === 'cashout') { tag.classList.add('tag-cashout'); tag.innerText = 'ATM CASHOUT'; }

    const freezeBtn = document.getElementById('freezeBtn');
    freezeBtn.innerText = node.frozen ? "🔓 Unfreeze Account" : "❄️ Freeze Account";
}

function freezeSelectedNode() {
    if (!selectedNode) return;
    selectedNode.frozen = !selectedNode.frozen;
    if (selectedNode.frozen) {
        audio.playFreeze();
        addChatMessage(`❄️ <b>COMPLIANCE ACTION:</b> Precautionary freeze applied to ${selectedNode.label} (${selectedNode.acc}). All outgoing settlement rails halted.`, 'alert');
        addLiveTx(`[DEBIT FREEZE] ${selectedNode.acc} (${selectedNode.label}) Locked by Investigator`, 'busted');
    } else {
        audio.playClick();
        addChatMessage(`🔓 <b>COMPLIANCE ACTION:</b> Freeze lifted for ${selectedNode.label} (${selectedNode.acc}).`, 'bot');
    }
    updateDossierHUD(selectedNode);
}

function traceRingFromSelected() {
    audio.playScan();
    addChatMessage(`🕸️ <b>RING TRACE:</b> Tracing 3-hop topological dependencies for ${selectedNode.label}... Found direct link to ATM Cashout Hub #04.`, 'alert');
}

// --- Blue Team Actions ---
function runContinuousXRay() {
    audio.playScan();
    addChatMessage("⚡ <b>GRAPH X-RAY COMPLETE:</b> Scanned 9 nodes, 8 edges. 3-hop mule ring isolated. Legitimacy regularizer verified 1 hard-negative merchant.", 'bot');
}

function freezeAllMules() {
    audio.playFreeze();
    nodes.forEach(n => {
        if (n.type === 'mule' || n.type === 'cashout') {
            n.frozen = true;
        }
    });
    updateDossierHUD(selectedNode);
    addChatMessage("❄️ <b>MASS LOCKDOWN:</b> All flagged mule nodes (Mule A, Mule B, Mule C, ATM Hub) have been frozen!", 'alert');
    addLiveTx("[EMERGENCY LOCKDOWN] 4 Illicit Nodes Frozen via FIU Directive", 'busted');
}

function unmaskCamouflageTrails() {
    audio.playScan();
    legitimacyLens = 100;
    document.getElementById('lensSlider').value = 100;
    adjustLegitimacyLens(100);
    addChatMessage("👁️ <b>CAMOUFLAGE UNMASKED:</b> Synthetic ₹250 micro-payment to Sharma Kirana isolated as adversarial camouflage noise.", 'bot');
}

// --- Phone Payment Simulation ---
let selectedRecipient = { name: "Rohan V. (Student Mule)", upi: "rohan88@okaxis", type: "mule" };

function selectRecipient(name, upi, type, el) {
    audio.playClick();
    selectedRecipient = { name, upi, type };
    document.querySelectorAll('.recipient-card').forEach(c => c.style.borderColor = 'rgba(255, 255, 255, 0.1)');
    el.style.borderColor = '#38bdf8';
    document.getElementById("selectedRecipName").innerText = name;
    document.getElementById("selectedRecipUPI").innerText = upi;
}

function setQuickAmount(amt) {
    audio.playClick();
    document.getElementById("paymentAmount").value = amt;
}

function executeUPIPayment() {
    const amt = parseFloat(document.getElementById("paymentAmount").value) || 5000;
    audio.playScan();

    const modal = document.getElementById("scannerModal");
    const scanStatus = document.getElementById("scanStatus");
    const verdictContent = document.getElementById("verdictContent");
    const verdictCard = document.getElementById("verdictCard");

    modal.classList.add("active");
    scanStatus.innerText = "Running AdaMule Graph Neural Network X-Ray...";
    verdictCard.style.display = "none";

    setTimeout(() => {
        verdictCard.style.display = "block";
        if (selectedRecipient.type === "mule") {
            audio.playAlarm();
            verdictCard.style.border = "2px solid #ef4444";
            verdictContent.innerHTML = `
                <div style="font-size: 2.5rem; margin-bottom: 8px;">🚨</div>
                <h3 style="color: #ef4444; margin-bottom: 8px;">PAYMENT INTERCEPTED</h3>
                <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 12px;">
                    Amount: <b>₹${amt.toLocaleString('en-IN')}</b> to <b>${selectedRecipient.name}</b>
                </p>
                <div style="background: rgba(239, 68, 68, 0.15); padding: 10px; border-radius: 8px; font-size: 0.78rem; text-align: left; margin-bottom: 14px;">
                    ⚠️ <b>Threat Detected:</b> Recipient is Node #2 in a 3-hop money mule ring.<br>
                    🧩 <b>Structuring Anomaly:</b> Dwell time < 8 mins. Downstream destination: ATM Cashout Hub.<br>
                    🛡️ <b>Verdict:</b> Blocked by AdaMule Min-Max Defense.
                </div>
                <button class="btn-pay" style="width: 100%; background: #ef4444;" onclick="closeModal()">Dismiss Alert</button>
            `;
            addChatMessage(`🚨 <b>INTERCEPTION ALERT:</b> ₹${amt.toLocaleString('en-IN')} payment to ${selectedRecipient.name} blocked! Multi-hop pass-through structuring detected.`, 'alert');
            addLiveTx(`[INTERCEPTED] ₹${amt.toLocaleString('en-IN')} to ${selectedRecipient.name} • Structuring Mule Intercepted`, 'busted');
        } else {
            audio.playSuccess();
            verdictCard.style.border = "2px solid #10b981";
            verdictContent.innerHTML = `
                <div style="font-size: 2.5rem; margin-bottom: 8px;">✅</div>
                <h3 style="color: #10b981; margin-bottom: 8px;">PAYMENT APPROVED</h3>
                <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 12px;">
                    Amount: <b>₹${amt.toLocaleString('en-IN')}</b> to <b>${selectedRecipient.name}</b>
                </p>
                <div style="background: rgba(16, 185, 129, 0.15); padding: 10px; border-radius: 8px; font-size: 0.78rem; text-align: left; margin-bottom: 14px;">
                    ✅ <b>Verified Merchant Profile:</b> Legitimate small business aggregator.<br>
                    🛡️ <b>AdaMule Protection:</b> Exempt from false positive lockouts.<br>
                    ⚡ <b>Settlement Rail:</b> Immediate clearance.
                </div>
                <button class="btn-pay" style="width: 100%; background: #10b981;" onclick="closeModal()">Complete Transaction</button>
            `;
            addChatMessage(`✅ <b>PAYMENT APPROVED:</b> ₹${amt.toLocaleString('en-IN')} to ${selectedRecipient.name}. Verified merchant aggregator; zero false positive.`, 'bot');
            addLiveTx(`[APPROVED] ₹${amt.toLocaleString('en-IN')} to ${selectedRecipient.name} • Verified Merchant Rail`, 'clean');
        }
    }, 1200);
}

function closeModal() {
    audio.playClick();
    document.getElementById("scannerModal").classList.remove("active");
}

// --- Live Ticker Feed Generator ---
function addLiveTx(text, type = 'clean') {
    const marquee = document.getElementById('txMarquee');
    const span = document.createElement('span');
    span.className = 'ticker-item';
    span.innerHTML = `<span class="${type === 'busted' ? 'ticker-tag-busted' : 'ticker-tag-clean'}">${text.split('•')[0]}</span> • ${text.split('•').slice(1).join('•')}`;
    marquee.prepend(span);
}

// Stream random transactions every 4 seconds
const sampleTxPool = [
    { text: "[APPROVED] ₹180.00 • Chai Point • Retail P2M Micro-Tx", type: "clean" },
    { text: "[APPROVED] ₹2,450.00 • Apollo Pharmacy • Verified Retail", type: "clean" },
    { text: "[INTERCEPTED] ₹49,990.00 • ACC_004521 (Mule B) • Below Reporting Threshold Smurf", type: "busted" },
    { text: "[APPROVED] ₹8,200.00 • Wholesale Provisions • Merchant Settlement", type: "clean" },
    { text: "[CAMOUFLAGE SUPPRESSED] ₹120.00 • Synthetic Edge to Sharma Kirana Neutralized", type: "busted" }
];
let txIndex = 0;
setInterval(() => {
    const item = sampleTxPool[txIndex % sampleTxPool.length];
    txIndex++;
    addLiveTx(item.text, item.type);
}, 4500);

// --- Official Legal SAR Report Generator Modal ---
function openSARModal() {
    audio.playClick();
    const modal = document.getElementById('sarModal');
    const content = document.getElementById('sarModalContent');

    const timestamp = new Date().toISOString();
    const hash = Array.from(crypto.getRandomValues(new Uint8Array(16))).map(b => b.toString(16).padStart(2, '0')).join('');

    content.innerHTML = `
        <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 6px; margin-bottom: 12px; font-family: monospace; font-size: 0.75rem;">
            <b>ELECTRONIC FILING HASH:</b> 0x${hash}<br>
            <b>TIMESTAMP:</b> ${timestamp}<br>
            <b>REPORTING ENTITY:</b> ADAMULE AUTONOMOUS CYBER DEFENSE SENTINEL (FINTECH RAIL)
        </div>

        <h4 style="color: #38bdf8; margin-bottom: 6px;">PART I: SUBJECT INFORMATION</h4>
        <table class="sar-table">
            <tr><th>Account ID</th><th>Account Holder</th><th>KYC Tier</th><th>Typology Classification</th></tr>
            <tr><td>ACC_008892</td><td>Rohan V.</td><td>Tier 1 Partial</td><td style="color:#f87171;">Primary Funnel Mule (Node #2)</td></tr>
            <tr><td>ACC_004521</td><td>Dormant Shell #4</td><td>Tier 1 Partial</td><td style="color:#f87171;">Intermediary Layer Mule (Node #3)</td></tr>
            <tr><td>ACC_007319</td><td>Anonymous Prepaid</td><td>Unverified</td><td style="color:#f87171;">Exit Layer Mule (Node #5)</td></tr>
            <tr><td>MERCH_SK889</td><td>Sharma Kirana Store</td><td>Verified GST</td><td style="color:#34d399;">EXEMPT: Hard-Negative Merchant</td></tr>
        </table>

        <h4 style="color: #38bdf8; margin: 12px 0 6px;">PART II: SUSPICIOUS ACTIVITY SUMMARY & NARRATIVE</h4>
        <p style="margin-bottom: 8px;">
            The AdaMule Graph Neural Network detected an organized, 3-hop structuring network executing rapid pass-through transfers. Total illicit volume under investigation: <b>₹75,000.00 INR</b>. Funds entered through victim phishing funnels (ACC_990112, ACC_990145) with a dwell time of under <b>8 minutes</b> before forwarding to downstream layered mule accounts.
        </p>
        <p style="margin-bottom: 8px;">
            <b>Adversarial Camouflage Analysis:</b> The syndicate injected synthetic ₹250 micro-transactions to a high-degree local retail merchant (Sharma Kirana Store) to disguise degree distribution. Legacy GCN baseline erroneously flagged the merchant (False Positive); however, the <b>AdaMule Legitimacy Discrimination Module</b> confirmed genuine retail dispersion, preserving the merchant while freezing the true mule exit path before ATM conversion.
        </p>

        <h4 style="color: #38bdf8; margin: 12px 0 6px;">PART III: RECOMMENDED ACTIONS</h4>
        <ul style="padding-left: 20px; font-size: 0.8rem;">
            <li>Immediate debit block on destination accounts ACC_008892, ACC_004521, and ACC_007319.</li>
            <li>Issue Section 91 CrPC notice to intermediary bank for device IMEI correlation.</li>
            <li>Exempt Sharma Kirana Store (MERCH_SK889) from automated AML lockouts.</li>
        </ul>
    `;

    modal.classList.add('active');
}

function closeSARModal() {
    audio.playClick();
    document.getElementById('sarModal').classList.remove('active');
}

function printSAR() {
    audio.playClick();
    window.print();
}

// --- AI Copilot Chat ---
function addChatMessage(msg, type) {
    const history = document.getElementById("chatHistory");
    const div = document.createElement("div");
    div.className = `chat-msg ${type === 'alert' ? 'msg-alert' : 'msg-bot'}`;
    div.innerHTML = msg;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function askPreset(type) {
    audio.playClick();
    if (type === 'ring') {
        addChatMessage("🔍 <b>Investigating Active Ring #1:</b><br>Detected a 3-hop linear funnel originating from Victim accounts (ACC_990112, ACC_990145) routing into Student Mule ACC_008892 (Rohan V.). Camouflage transaction injected to Sharma Kirana to disguise high fan-in velocity.", 'bot');
    } else if (type === 'sharma') {
        addChatMessage("🏬 <b>Entity Dossier: Sharma Kirana Store (Node #4)</b><br>• Status: <b>Protected Hard Negative</b><br>• Reason: Demonstrates genuine P2M retail dispersion across 15+ diverse consumer accounts.<br>• Standard GNN False Positive: <b>Eliminated by AdaMule</b> (FPR: 0.0000).", 'bot');
    } else if (type === 'fourier') {
        addChatMessage("📈 <b>Fourier Continuous Time Encoding:</b><br>$\phi(\Delta t) = [\cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \dots]$. Unlike static graphs, AdaMule encodes sub-hour dwell time intervals, preventing mules from evading detection by adding temporal hop delays.", 'bot');
    }
}

function sendUserChat() {
    const input = document.getElementById("userChatInput");
    const val = input.value.trim();
    if (!val) return;

    audio.playClick();
    addChatMessage(`<b>Investigator:</b> ${val}`, 'bot');
    input.value = "";

    setTimeout(() => {
        addChatMessage("🤖 <b>AdaMule Intelligence Analysis:</b> Identified pattern correlation with FATF Typology R.16 (Pass-Through Mule Structuring). Recommend placing immediate precautionary debit freeze on destination account.", 'alert');
    }, 600);
}

function toggleAudio() {
    isAudioMuted = audio.toggleMute();
    document.getElementById("audioBtn").innerText = isAudioMuted ? "🔇 Audio Off" : "🔊 Audio On";
}

