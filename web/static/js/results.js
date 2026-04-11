//load prediction data
const data = JSON.parse(sessionStorage.getItem("prediction_result"));

if (!data) {
    document.body.innerHTML = "<h2>No prediction data found.</h2>";
    throw new Error("No prediction data");
}

//progress bar fill based on prediction outcome
const probFraction = data.probability / 100;
const barFill = probFraction * 100

//save p1 name so progress bar is filled from correct side
const winnerIsP1 = data.winner === data.players.p1.name;

//prediction sumaary
document.getElementById("prediction-summary").innerHTML = `
    <h2>Winner: ${data.winner}</h2>
    <p>Probability: ${data.probability}%</p>
    
    <div class="progress-container">
        <div class="progress-labels">
            <span>${data.players.p1.name}</span>
            <span>${data.players.p2.name}</span>
        </div>

        <div class="progress-bar ${winnerIsP1 ? "left-win" : "right-win"}">
            <div class="progress-fill" style="width: ${barFill}%;"></div>
        </div>
    </div>
`;


//player cards
document.getElementById("player-cards").innerHTML = `
    <div class="player-card">
        <h3>${data.players.p1.name}</h3>
        <p>Ranking: ${data.players.p1.rank}</p>
        <p>Age: ${data.players.p1.age}</p>
        <p>Height: ${data.players.p1.height + " cm"}</p>
        <p>Dominant Hand: ${data.players.p1.hand}</p>
    </div>

    <div class="player-card">
        <h3>${data.players.p2.name}</h3>
        <p>Ranking: ${data.players.p2.rank}</p>
        <p>Age: ${data.players.p2.age}</p>
        <p>Height: ${data.players.p2.height + " cm"}</p>
        <p>Dominant Hand: ${data.players.p2.hand}</p>
    </div>
`;


// tourny metadata
document.getElementById("tournament-meta").innerHTML = `
    <div class="tournament-box">
        <h3>${data.tournament.tournament} (${data.tournament.year})</h3>
        <p>Round: ${data.tournament.round}</p>
        <p>Surface: ${data.tournament.surface}</p>
    </div>
`;

//narrative explanation
document.getElementById("narrative").innerHTML = `
    <div class="narrative-box">
        <h2>Why This Prediction?</h2>
        <p>${data.narrative}</p>
    </div>
`;


//widget switching (empty for now)
const widgetZone = document.getElementById("widget-zone");

const widgets = data.widgets || [];

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