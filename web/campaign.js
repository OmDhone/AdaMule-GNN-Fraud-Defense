// Gamified Cyber Detective Campaign (Story Mode)

const campaignMissions = [
    {
        id: 1,
        title: "Mission 1: The Campus Smurfs",
        difficulty: "ROOKIE",
        diffColor: "#34d399",
        xp: 250,
        objective: "Intercept the 2-hop student mule ring laundering phished tuition fees before ATM cashout.",
        targetNode: 2,
        unmaskRequired: false,
        briefing: "Syndicate recruiters targeted college students to split ₹75,000 into pass-through accounts. Find and freeze the primary funnel mule (Node #2)."
    },
    {
        id: 2,
        title: "Mission 2: The Kirana Smokescreen",
        difficulty: "DETECTIVE",
        diffColor: "#f59e0b",
        xp: 500,
        objective: "Unmask the camouflage transactions to Sharma Kirana without generating a false positive lock on the merchant!",
        targetNode: 4,
        unmaskRequired: true,
        briefing: "Fraudsters injected micro-payments to Sharma Kirana Store to mimic retail degree distributions. Legacy GCN panics and locks the merchant. Use AdaMule Legitimacy Lens to exonerate the merchant and freeze Mule C."
    },
    {
        id: 3,
        title: "Mission 3: Operation Ghost Mixer",
        difficulty: "MASTER",
        diffColor: "#ef4444",
        xp: 1000,
        objective: "Defeat a 5-hop RL-evasion structuring ring using Fourier temporal encoding and Min-Max robust retrained defense.",
        targetNode: 6,
        unmaskRequired: true,
        briefing: "High-level syndicate deploying gradient-evasion attacks with 4-hour temporal jitter. Execute full network lockdown before the illicit funds convert at the physical ATM hub."
    }
];

let currentMissionIndex = 0;
let detectiveXP = 0;
let completedMissions = [];

function openCampaignModal() {
    audio.playClick();
    const modal = document.getElementById('campaignModal');
    renderCampaignUI();
    modal.classList.add('active');
}

function closeCampaignModal() {
    audio.playClick();
    document.getElementById('campaignModal').classList.remove('active');
}

function renderCampaignUI() {
    const m = campaignMissions[currentMissionIndex];
    const card = document.getElementById('campaignCardContent');
    
    let missionSelectorHTML = campaignMissions.map((mis, idx) => `
        <button class="btn-ctrl" style="flex: 1; ${idx === currentMissionIndex ? 'border-color: var(--neon-cyan); color: var(--neon-cyan); background: rgba(56,189,248,0.2);' : ''}" onclick="selectMission(${idx})">
            ${mis.title.split(':')[0]} (${mis.difficulty})
        </button>
    `).join('');

    card.innerHTML = `
        <div style="display: flex; gap: 8px; margin-bottom: 14px;">
            ${missionSelectorHTML}
        </div>

        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 14px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 style="color: #fff; font-size: 1.1rem;">${m.title}</h3>
                <span style="background: ${m.diffColor}22; color: ${m.diffColor}; border: 1px solid ${m.diffColor}; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 800;">${m.difficulty} • +${m.xp} XP</span>
            </div>
            <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 10px; line-height: 1.5;">${m.briefing}</p>
            <div style="background: rgba(30, 41, 59, 0.6); padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; border-left: 3px solid var(--neon-cyan);">
                🎯 <b>Primary Objective:</b> ${m.objective}
            </div>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div style="font-size: 0.85rem;">
                Detective Rank: <b style="color: #38bdf8;">${detectiveXP >= 1500 ? 'Chief Inspector 🎖️' : detectiveXP >= 750 ? 'Senior Analyst 🔍' : 'Field Agent 🛡️'}</b>
            </div>
            <div style="font-size: 0.85rem;">
                Total Score: <b style="color: #10b981;">${detectiveXP} XP</b>
            </div>
        </div>

        <button class="btn-pay" style="width: 100%;" onclick="startCurrentMission()">
            🚀 START INVESTIGATION MISSION
        </button>
    `;
}

function selectMission(idx) {
    audio.playClick();
    currentMissionIndex = idx;
    renderCampaignUI();
}

function startCurrentMission() {
    audio.playLaserPulse();
    const m = campaignMissions[currentMissionIndex];
    closeCampaignModal();
    
    addChatMessage(`🎯 <b>MISSION STARTED: ${m.title}</b><br>Objective: ${m.objective}`, 'alert');
    if (typeof speakAI === 'function') {
        speakAI(`Starting ${m.title}. Good luck, detective.`);
    }

    // Auto-select initial clue node
    selectedNode = nodes[m.targetNode];
    updateDossierHUD(selectedNode);

    // Prompt user to solve
    setTimeout(() => {
        if (!completedMissions.includes(m.id)) {
            detectiveXP += m.xp;
            completedMissions.push(m.id);
            audio.playSuccess();
            addChatMessage(`🏆 <b>MISSION OBJECTIVE COMPLETE!</b><br>+${m.xp} XP Earned. Total Clearance Score: <b>${detectiveXP} XP</b>.`, 'bot');
            if (typeof speakAI === 'function') {
                speakAI(`Mission complete. You earned ${m.xp} experience points.`);
            }
        }
    }, 4000);
}
