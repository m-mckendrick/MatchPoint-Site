//load prediction data
const data = JSON.parse(sessionStorage.getItem("prediction_result"));

if (!data) {
    document.body.innerHTML = "<h2>No prediction data found.</h2>";
    throw new Error("No prediction data");
}

//extract sections from decoupled JSON
const prediction = data.prediction;
const explain = data.explainability;

//progress bar fill based on prediction outcome
const probFraction = prediction.probability / 100;
const barFill = probFraction * 100

//save p1 name so progress bar is filled from correct side
const winnerIsP1 = prediction.winner === explain.players.p1.name;

//prediction sumaary
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
    <div class="player-card">
        <h3>${explain.players.p1.name}</h3>
        <p>Ranking: ${explain.players.p1.rank}</p>
        <p>Age: ${explain.players.p1.age}</p>
        <p>Height: ${explain.players.p1.height + " cm"}</p>
        <p>Dominant Hand: ${explain.players.p1.hand}</p>
    </div>

    <div class="player-card">
        <h3>${explain.players.p2.name}</h3>
        <p>Ranking: ${explain.players.p2.rank}</p>
        <p>Age: ${explain.players.p2.age}</p>
        <p>Height: ${explain.players.p2.height + " cm"}</p>
        <p>Dominant Hand: ${explain.players.p2.hand}</p>
    </div>
`;


// tourny metadata
document.getElementById("tournament-meta").innerHTML = `
    <div class="tournament-box">
        <h3>${explain.tournament.tournament} (${explain.tournament.year})</h3>
        <p>Round: ${explain.tournament.round}</p>
        <p>Surface: ${explain.tournament.surface}</p>
    </div>
`;

//narrative explanation
document.getElementById("narrative").innerHTML = `
    <div class="narrative-box">
        <h2>Why This Prediction?</h2>
        <p>${explain.narrative}</p>
    </div>
`;


//widget switching (empty for now)
const widgetZone = document.getElementById("widget-zone");
const widgets = explain.widgets || [];

//render widgets based on its type
function renderWidget(type) {
    const widget = widgets.find(w => w.type === type);

    if (!widget) {
        widgetZone.innerHTML = "<p>No data available for this widget.</p>";
        return;
    }

    //handle widget types
    switch (type) {
        //top features widget
        case "top_features":
            widgetZone.innerHTML = `
                <h3>${widget.title}</h3>
                <ul>
                    ${widget.data.map(f => `
                        <li><strong>${f.feature}</strong>: ${f.value}</li>
                    `).join("")}
                </ul>
            `;
            break;

        //fallback
        default:
            widgetZone.innerHTML = `
                <h3>${widget.title}</h3>
                <pre>${JSON.stringify(widget.data, null, 2)}</pre>
            `;
            break;
    }

}
//render widgets based on button click
document.querySelectorAll(".widget-buttons button").forEach(btn => {
    btn.addEventListener("click", () => {
        renderWidget(btn.dataset.widget);
    });
});

//render first widget by default (for prototype)
if (widgets.length > 0) {
    renderWidget(widgets[0].type);
}