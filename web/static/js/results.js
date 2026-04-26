import { renderWidgetZone } from "/static/js/widgets.js";

// ===============================
// Load prediction data
// ===============================
const data = JSON.parse(sessionStorage.getItem("prediction_result"));

if (!data) {
    document.body.innerHTML = "<h2>No prediction data found.</h2>";
    throw new Error("No prediction data");
}

const prediction = data.prediction;
const explain = data.explainability;


//progress bar fill based on prediction outcome
const probFraction = prediction.probability / 100;
const barFill = probFraction * 100

//save p1 name so progress bar is filled from correct side
const winnerIsP1 = prediction.winner === explain.players.p1.name;

document.getElementById("prediction-summary").innerHTML = `
    <h2>Winner: ${prediction.winner}</h2>
    <p>Probability: ${prediction.probability}%</p>

    <div class="progress-container">
        <div class="progress-labels">
            <span>${explain.players.p1.name}</span>
            <span>${explain.players.p2.name}</span>
        </div>

        <div class="progress-bar ${winnerIsP1 ? "left-win" : "right-win"}">
            <div class="progress-fill" style="width: ${barFill}%;"></div>
        </div>
    </div>
`;

//player cards
document.getElementById("player-cards").innerHTML = `
    <div class="player-card-container">
        <div class="player-card player1-card" id="player1-card">
            <h3>${explain.players.p1.name}</h3>
            <p><strong>Ranking:</strong> ${explain.players.p1.rank}</p>
            <p><strong>Age:</strong> ${explain.players.p1.age}</p>
            <p><strong>Height:</strong> ${explain.players.p1.height} cm</p>
            <p><strong>Hand:</strong> ${explain.players.p1.hand}</p>
            <p><strong>Country:</strong> ${explain.players.p1.ioc}</p>
            <p><strong>Elo (Global):</strong> ${explain.players.p1.elo_global}</p>
            <p><strong>Elo (Surface):</strong> ${explain.players.p1.elo_surface}</p>
        </div>
    
        <div class="player-card player2-card" id="player2-card">
            <h3>${explain.players.p2.name}</h3>
            <p><strong>Ranking:</strong> ${explain.players.p2.rank}</p>
            <p><strong>Age:</strong> ${explain.players.p2.age}</p>
            <p><strong>Height:</strong> ${explain.players.p2.height} cm</p>
            <p><strong>Hand:</strong> ${explain.players.p2.hand}</p>
            <p><strong>Country:</strong> ${explain.players.p2.ioc}</p>
            <p><strong>Elo (Global):</strong> ${explain.players.p2.elo_global}</p>
            <p><strong>Elo (Surface):</strong> ${explain.players.p2.elo_surface}</p>
        </div>
    </div>
`;


// tournament metadata
document.getElementById("tournament-meta").innerHTML = `
    <div class="tournament-box">
        <h3>${explain.tournament.tournament} (${explain.tournament.year})</h3>
        <p>Round: ${explain.tournament.round}</p>
        <p>Surface: ${explain.tournament.surface}</p>
    </div>
`;


// narrative
document.getElementById("narrative").innerHTML = `
    <div class="narrative-box">
        <h2>Why This Prediction?</h2>
        <p>${explain.narrative}</p>
    </div>
`;


// widget zones
const widgetZones = explain.widgets || {};
const widgetButtons = document.getElementById("widget-buttons");
const widgetZoneContainer = document.getElementById("widget-zone");

// create a button for each widget group
Object.keys(widgetZones).forEach(groupName => {
    const btn = document.createElement("button");
    btn.textContent = groupName;
    btn.dataset.group = groupName;
    widgetButtons.appendChild(btn);
});

// render a widget group using widgets.js
function renderWidgetGroup(groupName) {
    widgetZoneContainer.innerHTML = ""; // clear previous
    const widgets = widgetZones[groupName];
    renderWidgetZone(groupName, widgets, widgetZoneContainer);
}

// button click handler
widgetButtons.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") {
        const group = e.target.dataset.group;
        renderWidgetGroup(group);
    }
});

// render the first group by default
const firstGroup = Object.keys(widgetZones)[0];
if (firstGroup) {
    renderWidgetGroup(firstGroup);
}