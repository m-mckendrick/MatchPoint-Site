//load prediction data
const data = JSON.parse(sessionStorage.getItem("prediction_result"));

if (!data) {
    document.body.innerHTML = "<h2>No prediction data found.</h2>";
    throw new Error("No prediction data");
}

//prediction sumaary
document.getElementById("prediction-summary").innerHTML = `
    <h2>Winner: ${data.winner}</h2>
    <p>Probability: ${data.probability}%</p>
`;


// Player cards
document.getElementById("player-cards").innerHTML = `
    <div class="player-card">
        <h3>${data.players.p1.name}</h3>
        <p>Ranking: ${data.players.p1.ranking}</p>
        <p>Age: ${data.players.p1.age}</p>
    </div>

    <div class="player-card">
        <h3>${data.players.p2.name}</h3>
        <p>Ranking: ${data.players.p2.ranking}</p>
        <p>Age: ${data.players.p2.age}</p>
    </div>
`;


// tourny metadata
document.getElementById("tournament-meta").innerHTML = `
    <h3>${data.tournament.name} (${data.tournament.year})</h3>
    <p>Round: ${data.tournament.round}</p>
    <p>Surface: ${data.tournament.surface}</p>
`;


// Widget switching (empty for now)
const widgetZone = document.getElementById("widget-zone");

//later on will have relevant logic
function renderWidget(type) {
    widgetZone.innerHTML = "<p></p>";
}
//render widgets based on button click
document.querySelectorAll(".widget-buttons button").forEach(btn => {
    btn.addEventListener("click", () => {
        renderWidget(btn.dataset.widget);
    });
});
