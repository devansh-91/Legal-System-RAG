(() => {
  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const clearBtn = document.getElementById("clear");

  let history = [];
  let busy = false;

  // ---------- tiny, safe markdown renderer ----------
  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  function inline(text) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>')
      .replace(/(https?:\/\/[^\s<)]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  }

  function renderMarkdown(src) {
    const lines = escapeHtml(src).split("\n");
    let html = "";
    let list = null; // "ul" | "ol"
    let para = [];
    let table = [];

    const flushPara = () => {
      if (para.length) html += `<p>${inline(para.join("<br>"))}</p>`;
      para = [];
    };
    const flushList = () => {
      if (list) html += `</${list}>`;
      list = null;
    };
    const flushTable = () => {
      if (!table.length) return;
      const rows = table.filter((r) => !/^\|?\s*:?-{2,}/.test(r));
      html += "<table>";
      rows.forEach((row, i) => {
        const cells = row.replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
        const tag = i === 0 ? "th" : "td";
        html += "<tr>" + cells.map((c) => `<${tag}>${c}</${tag}>`).join("") + "</tr>";
      });
      html += "</table>";
      table = [];
    };

    for (const raw of lines) {
      const line = raw.trimEnd();
      if (/^\s*\|.*\|\s*$/.test(line)) { flushPara(); flushList(); table.push(line.trim()); continue; }
      flushTable();
      let m;
      if ((m = line.match(/^#{1,6}\s+(.*)/))) { flushPara(); flushList(); html += `<h4>${inline(m[1])}</h4>`; }
      else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) {
        flushPara(); if (list !== "ul") { flushList(); html += "<ul>"; list = "ul"; }
        html += `<li>${inline(m[1])}</li>`;
      } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
        flushPara(); if (list !== "ol") { flushList(); html += "<ol>"; list = "ol"; }
        html += `<li>${inline(m[1])}</li>`;
      } else if (!line.trim()) { flushPara(); flushList(); }
      else { flushList(); para.push(line); }
    }
    flushTable(); flushPara(); flushList();
    return html;
  }

  // ---------- UI helpers ----------
  function addMessage(role, content = "") {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "user") bubble.textContent = content;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scroll();
    return bubble;
  }

  const scroll = () => { messagesEl.scrollTop = messagesEl.scrollHeight; };

  function renderSources(sources) {
    if (!sources || !sources.length) return null;
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${sources.length})`;
    details.appendChild(summary);
    for (const s of sources) {
      const div = document.createElement("div");
      div.className = "source";
      const title = document.createElement("div");
      title.className = "title";
      title.textContent = `[${s.id}] ${s.title}${s.section ? " › " + s.section.split(" › ").slice(-1)[0] : ""}${s.page ? ` (p. ${s.page})` : ""}`;
      div.appendChild(title);
      if (s.reference) {
        const ref = document.createElement("div");
        ref.className = "ref";
        ref.textContent = s.reference;
        div.appendChild(ref);
      }
      const snip = document.createElement("div");
      snip.className = "snippet";
      snip.textContent = s.snippet + (s.snippet.length >= 400 ? "…" : "");
      div.appendChild(snip);
      details.appendChild(div);
    }
    return details;
  }

  function setBusy(value) {
    busy = value;
    sendBtn.disabled = value;
    input.disabled = value;
  }

  // ---------- chat ----------
  async function ask(question) {
    if (busy || !question.trim()) return;
    setBusy(true);
    addMessage("user", question);
    const bubble = addMessage("assistant");
    bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';

    const notices = document.createElement("div");
    const body = document.createElement("div");
    let answer = "";
    let meta = null;
    let started = false;

    const start = () => {
      if (started) return;
      started = true;
      bubble.innerHTML = "";
      bubble.appendChild(notices);
      bubble.appendChild(body);
    };

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ? JSON.stringify(err.detail) : `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const chunk = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          if (!chunk.startsWith("data: ")) continue;
          const event = JSON.parse(chunk.slice(6));
          if (event.type === "meta") {
            meta = event;
            start();
            for (const n of event.notices || []) {
              const div = document.createElement("div");
              div.className = "notice";
              div.textContent = "⚠️ " + n;
              notices.appendChild(div);
            }
          } else if (event.type === "token") {
            start();
            answer += event.content;
            body.innerHTML = renderMarkdown(answer);
            scroll();
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        }
      }
      start();
      const src = renderSources(meta && meta.sources);
      if (src) bubble.appendChild(src);
      history.push({ role: "user", content: question }, { role: "assistant", content: answer });
      history = history.slice(-10);
    } catch (err) {
      start();
      const p = document.createElement("p");
      p.className = "error";
      p.textContent = "Sorry, something went wrong: " + err.message;
      body.appendChild(p);
    } finally {
      setBusy(false);
      input.focus();
      scroll();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value;
    input.value = "";
    autoresize();
    ask(q);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  const autoresize = () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
  };
  input.addEventListener("input", autoresize);

  document.getElementById("topics").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-q]");
    if (btn) ask(btn.dataset.q);
  });

  clearBtn.addEventListener("click", () => {
    history = [];
    messagesEl.querySelectorAll(".msg:not(.welcome)").forEach((el) => el.remove());
    input.focus();
  });
})();
