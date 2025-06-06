let ctxMap = {};

window.addEventListener("load", function () {
  document.querySelectorAll(".drawingCanvas").forEach(canvas => {
    const id = canvas.id;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctxMap[id] = ctx;

    canvas.addEventListener("mousedown", e => startDrawing(e, id));
    canvas.addEventListener("mouseup", e => stopDrawing(e, id));
    canvas.addEventListener("mouseout", e => stopDrawing(e, id));
    canvas.addEventListener("mousemove", e => draw(e, id));
  });
});

let drawingStates = {};

function startDrawing(e, id) {
  drawingStates[id] = true;
  ctxMap[id].beginPath();
  ctxMap[id].moveTo(e.offsetX, e.offsetY);
}

function stopDrawing(e, id) {
  drawingStates[id] = false;
  ctxMap[id].closePath();
}

function draw(e, id) {
  if (!drawingStates[id]) return;
  const ctx = ctxMap[id];
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.strokeStyle = "black";
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
}

function clearCanvas(id) {
  const ctx = ctxMap[id];
  const canvas = document.getElementById(id);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function submitDrawing(name) {
  const canvas = document.getElementById(name);
  const imageData = canvas.toDataURL('image/jpeg');
  fetch('/upload_drawing', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'drawing=' + encodeURIComponent(imageData) + '&name=' + name
  }).then(() => {
    alert(`🎨 ${name} 그림이 저장되었습니다.`);
  });
}
