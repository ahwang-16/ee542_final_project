function switchTab(tabName, btn) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`tab-${tabName}`).classList.add("active");
  btn.classList.add("active");
  if (tabName === "manager") loadTickets();
}

async function updatePersonaScale(scaleVal) {
  await fetch("/api/persona/scale", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scale: parseFloat(scaleVal) }),
  });
}

async function sendQuestion(e) {
  e.preventDefault();
  const nameInput = document.getElementById("customerName");
  const qInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const chatBox = document.getElementById("chatMessages");

  const question = qInput.value.trim();
  if (!question) return;

  chatBox.innerHTML += `<div class="msg user"><strong>${nameInput.value}:</strong>\n${question}</div>`;
  qInput.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "Thinking...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_name: nameInput.value, question }),
    });
    const data = await res.json();
    const sourcesText = (data.sources || []).map(s => s.source).join(", ") || "None";
    chatBox.innerHTML += `
      <div class="msg bot">
        <strong>TwinRep Agent (Ticket #${data.ticket_id}):</strong>\n${data.ai_draft}
        <div class="sources-pill">RAG Sources Referenced: ${sourcesText}</div>
      </div>`;
  } catch (err) {
    chatBox.innerHTML += `<div class="msg bot">Error contacting TwinRep server: ${err.message}</div>`;
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send Inquiry";
    chatBox.scrollTop = chatBox.scrollHeight;
  }
}

async function uploadManual() {
  const fileInput = document.getElementById("manualFile");
  const statusEl = document.getElementById("uploadStatus");
  if (!fileInput.files.length) return alert("Select a PDF, TXT, or MD file first.");

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  statusEl.textContent = "Chunking and embedding document...";

  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await res.json();
  statusEl.textContent = `Indexed "${data.filename}" (${data.chunks_indexed} chunks added to FAISS).`;
}

async function syncFolder() {
  const statusEl = document.getElementById("uploadStatus");
  statusEl.textContent = "Indexing all files in /data/manuals...";
  const res = await fetch("/api/ingest-folder", { method: "POST" });
  const data = await res.json();
  statusEl.textContent = `Sync complete: ${data.total_chunks_indexed} total chunks indexed.`;
}

async function loadTickets() {
  const container = document.getElementById("ticketList");
  const res = await fetch("/api/tickets");
  const tickets = await res.json();

  if (!tickets.length) {
    container.innerHTML = "<p>No customer tickets yet. Ask a question in Tab 1 first.</p>";
    return;
  }

  container.innerHTML = tickets.map(t => `
    <div class="ticket-card">
      <div class="ticket-meta">
        <span><strong>Ticket #${t.id}</strong> — ${t.customer_name}</span>
        <span class="status-tag ${t.status}">${t.status.replace("_", " ")}</span>
      </div>
      <p><strong>Customer Question:</strong> ${t.question}</p>
      <label><strong>AI Drafted Response (Editable):</strong></label>
      <textarea id="draft-${t.id}">${t.final_response || t.ai_draft}</textarea>
      <div class="ticket-actions">
        <button class="approve" onclick="approveTicket(${t.id})">Approve & Index</button>
        <button onclick="editTicket(${t.id})">Edit & Teach AI</button>
      </div>
    </div>
  `).join("");
}

async function approveTicket(id) {
  await fetch(`/api/tickets/${id}/approve`, { method: "POST" });
  loadTickets();
}

async function editTicket(id) {
  const edited_response = document.getElementById(`draft-${id}`).value;
  await fetch(`/api/tickets/${id}/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edited_response }),
  });
  alert(`Ticket #${id} saved and vectorized into the live RAG Knowledge Base!`);
  loadTickets();
}
