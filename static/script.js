let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let isDrawing = false;
let loader = document.getElementById("loader");

// Initialize canvas
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth = 15;
ctx.lineJoin = "round";
ctx.lineCap = "round";

// Start drawing
canvas.addEventListener("mousedown", () => { isDrawing = true; });
canvas.addEventListener("mouseup", () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener("mousemove", draw);

function draw(event) {
    if (!isDrawing) return;
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.offsetX, event.offsetY);
}

// Convert canvas to image & send to Flask backend
function predictDigit() {
    let imageData = canvas.toDataURL("image/png");
    loader.style.display = "block";

    fetch("/predict", {
        method: "POST",
        body: JSON.stringify({ image: imageData }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Predicted Digit: " + data.prediction;
    })
    .catch(error => console.error("Error:", error))
    .finally(() => loader.style.display = "none");
}

// Clear the canvas
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    document.getElementById("result").innerText = "";
}
