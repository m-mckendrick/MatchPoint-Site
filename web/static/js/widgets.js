//render zone
export function renderWidgetZone(groupName, widgets, container) {
    const zone = document.createElement("div");
    zone.className = "widget-zone";

    const header = document.createElement("h2");
    header.className = "widget-zone-title";
    header.textContent = groupName;
    zone.appendChild(header);

    widgets.forEach(widget => {
        const widgetEl = renderWidget(widget);
        if (widgetEl) zone.appendChild(widgetEl);
    });

    container.appendChild(zone);
}


//get widget dependant on type
function renderWidget(widget) {
    switch (widget.type) {

        case "group_narrative":
            return renderNarrativeWidget(widget);

        case "comparison":
            return renderComparisonWidget(widget);

        case "stat_card":
            return renderStatCardWidget(widget);

        case "shap_bar":
            return renderShapBarWidget(widget);

        default:
            console.warn("Unknown widget type:", widget.type);
            return renderUnknownWidget(widget);
    }
}



function renderUnknownWidget(widget) {
    const el = document.createElement("pre");
    el.textContent = JSON.stringify(widget, null, 2);
    return el;
}

//narrative widget
function renderNarrativeWidget(widget) {
    const el = document.createElement("div");
    el.className = "widget narrative-widget";

    const title = document.createElement("h3");
    title.textContent = widget.title;
    el.appendChild(title);

    const text = document.createElement("p");
    text.innerHTML = widget.text;
    el.appendChild(text);

    return el;
}

//comparison
function renderComparisonWidget(widget) {
    const el = document.createElement("div");
    el.className = "widget comparison-widget";

    const title = document.createElement("h3");
    title.textContent = widget.title;
    el.appendChild(title);

    const data = widget.data;
    const p1Name = data.label_p1;
    const p2Name = data.label_p2;

    const table = document.createElement("table");
    table.className = "comparison-table";

    table.innerHTML = `
        <tr>
            <th>${p1Name}</th>
            <th>Metric</th>
            <th>${p2Name}</th>
        </tr>
    `;

    // Raw keys: p1_xxx / p2_xxx
    const rawP1 = Object.keys(data).filter(k => k.startsWith("p1_"));
    const rawP2 = Object.keys(data).filter(k => k.startsWith("p2_"));

    if (rawP1.length > 0) {
        rawP1.forEach(p1Key => {
            const metric = p1Key.replace("p1_", "");
            const p2Key = "p2_" + metric;

            const row = document.createElement("tr");
            row.innerHTML = `
                <td class="p1-value">${data[p1Key]}</td>
                <td class="metric-label">${metric}</td>
                <td class="p2-value">${data[p2Key] ?? "-"}</td>
            `;
            table.appendChild(row);
        });

        el.appendChild(table);
        return el;
    }

    // case "diff"
    if ("p1" in data && "p2" in data) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td class="p1-value">${data.p1}</td>
            <td class="metric-label">Value</td>
            <td class="p2-value">${data.p2}</td>
        `;
        table.appendChild(row);
    }

    el.appendChild(table);
    return el;
}



//stat card widget
function renderStatCardWidget(widget) {
    const el = document.createElement("div");
    el.className = "widget stat-card-widget";

    const title = document.createElement("h3");
    title.textContent = widget.title;
    el.appendChild(title);

    const list = document.createElement("ul");

    const stats = widget.data || {};

    if (Object.keys(stats).length === 0) {
        list.innerHTML = "<li>No data available.</li>";
        el.appendChild(list);
        return el;
    }

    Object.entries(stats).forEach(([key, value]) => {
        const li = document.createElement("li");

        if (typeof value === "object" && value !== null) {
            li.innerHTML = `<strong>${key}:</strong> ${JSON.stringify(value)}`;
        } else {
            li.innerHTML = `<strong>${key}:</strong> ${value}`;
        }

        list.appendChild(li);
    });

    el.appendChild(list);
    return el;
}


// shap bar widget
function renderShapBarWidget(widget) {
    const el = document.createElement("div");
    el.className = "widget shap-bar-widget";

    const title = document.createElement("h3");
    title.textContent = widget.title;
    el.appendChild(title);

    const canvas = document.createElement("canvas");
    el.appendChild(canvas);

    const features = widget.data.features;
    const shapValues = features.map(f => widget.data.raw_shap_values[f]);

    new Chart(canvas, {
        type: "bar",
        data: {
            labels: features,
            datasets: [{
                label: "SHAP Value",
                data: shapValues,
                backgroundColor: shapValues.map(v =>
                    v >= 0 ? "rgb(98,189,74)" : "rgb(184,71,71)"
                )
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            plugins: { legend: { display: false } }
        }
    });

    return el;
}
