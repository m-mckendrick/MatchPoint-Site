const form = document.getElementById("predict-form");

//when submitted get form data create json and store it
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    const payload = {
        p1_name: formData.get("p1_name"),
        p2_name: formData.get("p2_name"),
        tournament: formData.get("tournament"),
        year: Number(formData.get("year")),
        round_label: formData.get("round_label"),
    };

    const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    const data = await response.json();

    // Store result for results page
    sessionStorage.setItem("prediction_result", JSON.stringify(data));

    // Redirect to results page
    window.location.href = "/results";
});
