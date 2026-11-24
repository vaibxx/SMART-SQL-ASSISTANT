// Smart SQL (web): plain-English questions -> Gemini -> SQL -> real PostgreSQL in the browser.
// The user's Gemini key lives only in the `apiKey` variable below. It is never written to
// storage, cookies or the URL, and it is only ever sent to Google's Gemini endpoint.
import { PGlite } from 'https://cdn.jsdelivr.net/npm/@electric-sql/pglite@0.5.17/dist/index.js';

const MODEL = 'gemini-3.5-flash-lite';
const ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;
const MAX_ROWS_SHOWN = 200;

let apiKey = null;
let db = null;
let schema = [];
let mode = 'ask';
let busy = false;
const history = []; // [{q, sql, explanation}] for follow-up questions

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const qi = (name) => (/^[a-z_][a-z0-9_]*$/.test(name) ? name : `"${name.replace(/"/g, '""')}"`);

/* ------------------------------------------------------------------ */
/* Sample database                                                     */
/* ------------------------------------------------------------------ */
function rng(seed) {
  return () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
}

function sampleSQL() {
  const r = rng(7);
  const pick = (a) => a[Math.floor(r() * a.length)];
  const first = ['Aarav', 'Diya', 'Kabir', 'Meera', 'Rohan', 'Ananya', 'Vihaan', 'Isha', 'Arjun', 'Saanvi', 'Liam', 'Emma', 'Noah', 'Olivia', 'Lucas', 'Mia', 'Kenji', 'Yuki', 'Omar', 'Layla', 'Sofia', 'Mateo', 'Zara', 'Ethan'];
  const last = ['Sharma', 'Patel', 'Iyer', 'Khan', 'Gupta', 'Reddy', 'Smith', 'Garcia', 'Muller', 'Tanaka', 'Rossi', 'Silva', 'Nguyen', 'Brown'];
  const cities = [['Mumbai', 'India'], ['Bengaluru', 'India'], ['Delhi', 'India'], ['Pune', 'India'], ['London', 'UK'], ['Manchester', 'UK'], ['Berlin', 'Germany'], ['New York', 'USA'], ['Austin', 'USA'], ['San Francisco', 'USA'], ['Tokyo', 'Japan'], ['Singapore', 'Singapore'], ['Dubai', 'UAE'], ['Toronto', 'Canada']];
  const products = [
    ['Aero 14 Laptop', 'Laptops', 1199], ['Titan 16 Laptop', 'Laptops', 1899], ['Nova X Phone', 'Phones', 899], ['Nova Lite Phone', 'Phones', 449],
    ['Pulse Earbuds', 'Audio', 129], ['Studio Headphones', 'Audio', 249], ['Boom Speaker', 'Audio', 89], ['Orbit Watch', 'Wearables', 299],
    ['Fit Band', 'Wearables', 79], ['USB-C Hub', 'Accessories', 49], ['Mech Keyboard', 'Accessories', 139], ['Ergo Mouse', 'Accessories', 59],
    ['4K Webcam', 'Accessories', 119], ['Laptop Stand', 'Accessories', 39], ['Fast Charger', 'Accessories', 35], ['Smart Bulb 4-pack', 'Smart Home', 45],
    ['Air Purifier', 'Smart Home', 229], ['Robot Vacuum', 'Smart Home', 399], ['Game Console', 'Gaming', 499], ['Pro Controller', 'Gaming', 69],
    ['VR Headset', 'Gaming', 549], ['27in 4K Monitor', 'Displays', 379], ['34in Ultrawide', 'Displays', 649], ['Portable Monitor', 'Displays', 199]
  ];
  const day = (start, span) => new Date(Date.UTC(2025, 0, 1) + Math.floor(start + r() * span) * 864e5).toISOString().slice(0, 10);

  let sql = `
    CREATE TABLE customers (customer_id SERIAL PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE, city TEXT, country TEXT, signup_date DATE);
    CREATE TABLE products (product_id SERIAL PRIMARY KEY, name TEXT NOT NULL, category TEXT, price NUMERIC(10,2), cost NUMERIC(10,2));
    CREATE TABLE orders (order_id SERIAL PRIMARY KEY, customer_id INT REFERENCES customers, order_date DATE, status TEXT, channel TEXT);
    CREATE TABLE order_items (order_item_id SERIAL PRIMARY KEY, order_id INT REFERENCES orders, product_id INT REFERENCES products, quantity INT, unit_price NUMERIC(10,2), discount NUMERIC(4,2));
  `;
  const cust = [];
  for (let i = 1; i <= 80; i++) {
    const f = pick(first), l = pick(last), [city, country] = pick(cities);
    cust.push(`('${f} ${l}','${f.toLowerCase()}.${l.toLowerCase()}${i}@example.com','${city}','${country}','${day(-400, 990)}')`);
  }
  sql += `INSERT INTO customers (name,email,city,country,signup_date) VALUES ${cust.join(',')};`;
  sql += `INSERT INTO products (name,category,price,cost) VALUES ${products.map(([n, c, p]) => `('${n}','${c}',${p},${(p * (0.55 + r() * 0.2)).toFixed(2)})`).join(',')};`;

  const orders = [], items = [];
  for (let o = 1; o <= 650; o++) {
    const s = r();
    const status = s < 0.78 ? 'delivered' : s < 0.86 ? 'shipped' : s < 0.94 ? 'returned' : 'cancelled';
    orders.push(`(${1 + Math.floor(r() * 80)},'${day(0, 690)}','${status}','${pick(['web', 'web', 'app', 'app', 'store'])}')`);
    const n = 1 + Math.floor(r() * r() * 4);
    for (let k = 0; k < n; k++) {
      const pi = Math.floor(Math.pow(r(), 1.4) * products.length);
      const disc = r() < 0.25 ? pick([0.05, 0.1, 0.15, 0.2]) : 0;
      items.push(`(${o},${pi + 1},${1 + Math.floor(r() * r() * 3)},${products[pi][2]},${disc})`);
    }
  }
  sql += `INSERT INTO orders (customer_id,order_date,status,channel) VALUES ${orders.join(',')};`;
  sql += `INSERT INTO order_items (order_id,product_id,quantity,unit_price,discount) VALUES ${items.join(',')};`;
  return sql;
}

const SAMPLE_NOTES = `Business rules for the sample store:
- Revenue of a line = quantity * unit_price * (1 - discount), from order_items.
- Exclude orders with status 'cancelled' from revenue unless asked. 'returned' orders count as returned, not revenue.
- Profit of a line = quantity * (unit_price * (1 - discount) - products.cost).`;

async function startDb(withSample = true) {
  setPill('#db-status', 'Starting…', '');
  if (db) await db.close().catch(() => {});
  db = new PGlite();
  if (withSample) await db.exec(sampleSQL());
  await loadSchema();
}

async function loadSchema() {
  const cols = await db.query(`
    SELECT c.table_name, c.column_name, c.data_type
    FROM information_schema.columns c
    JOIN information_schema.tables t ON t.table_schema = c.table_schema AND t.table_name = c.table_name
    WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
    ORDER BY c.table_name, c.ordinal_position`);
  const byTable = new Map();
  for (const row of cols.rows) {
    if (!byTable.has(row.table_name)) byTable.set(row.table_name, []);
    byTable.get(row.table_name).push({ name: row.column_name, type: row.data_type });
  }
  schema = [];
  for (const [name, columns] of byTable) {
    const count = Number((await db.query(`SELECT count(*)::int AS n FROM ${qi(name)}`)).rows[0].n);
    const sample = (await db.query(`SELECT * FROM ${qi(name)} LIMIT 3`)).rows;
    schema.push({ name, columns, count, sample });
  }
  renderSchema();
  setPill('#db-status', `${schema.length} table${schema.length === 1 ? '' : 's'}`, 'ok');
}

function renderSchema() {
  $('#schema').innerHTML = schema.map((t) => `
    <details class="tbl">
      <summary>${esc(t.name)} <span>${t.count.toLocaleString()} rows</span></summary>
      <ul>${t.columns.map((c) => `<li><b>${esc(c.name)}</b><span>${esc(c.type)}</span></li>`).join('')}</ul>
    </details>`).join('');
}

function schemaForPrompt() {
  const fmt = (v) => (v instanceof Date ? v.toISOString().slice(0, 10) : v);
  const parts = schema.map((t) => {
    const cols = t.columns.map((c) => `${c.name} ${c.type}`).join(', ');
    const rows = t.sample.map((r) => JSON.stringify(Object.fromEntries(Object.entries(r).map(([k, v]) => [k, fmt(v)])))).join('\n    ');
    return `TABLE ${t.name} (${cols}) -- ${t.count} rows\n  sample rows:\n    ${rows}`;
  });
  const hasSample = schema.some((t) => t.name === 'order_items');
  return parts.join('\n\n') + (hasSample ? `\n\n${SAMPLE_NOTES}` : '');
}

/* ------------------------------------------------------------------ */
/* Gemini                                                              */
/* ------------------------------------------------------------------ */
async function askGemini(question, repair) {
  const system = `You translate questions into ONE PostgreSQL query for the database below.
Rules:
- Exactly one SQL statement. Prefer SELECT. Use only the tables and columns listed.
- For "top N" use ORDER BY with LIMIT. Round money to 2 decimals. Give result columns readable aliases.
- If the user asks to change data, write the INSERT, UPDATE or DELETE. Never write DROP, TRUNCATE, ALTER, CREATE, GRANT or REVOKE.
- If the question cannot be answered from this schema, return an empty sql and explain why.
- chart: choose "bar", "line" (time series) or "pie" (share of a whole, few slices) when the result has one label/date column and one numeric column; set x and y to those result column names. Otherwise "none".
- explanation: one or two plain-English sentences describing what the query returns.
Today's date: ${new Date().toISOString().slice(0, 10)}.

DATABASE
${schemaForPrompt()}`;

  const contents = [];
  for (const h of history.slice(-4)) {
    contents.push({ role: 'user', parts: [{ text: h.q }] });
    contents.push({ role: 'model', parts: [{ text: JSON.stringify({ sql: h.sql, explanation: h.explanation }) }] });
  }
  contents.push({ role: 'user', parts: [{ text: question }] });
  if (repair) {
    contents.push({ role: 'model', parts: [{ text: JSON.stringify({ sql: repair.sql }) }] });
    contents.push({ role: 'user', parts: [{ text: `That query failed in PostgreSQL with: ${repair.error}\nReturn a corrected query.` }] });
  }

  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents,
      generationConfig: {
        temperature: 0.1,
        responseMimeType: 'application/json',
        responseSchema: {
          type: 'OBJECT',
          properties: {
            sql: { type: 'STRING' },
            explanation: { type: 'STRING' },
            chart: {
              type: 'OBJECT',
              properties: { type: { type: 'STRING', enum: ['none', 'bar', 'line', 'pie'] }, x: { type: 'STRING' }, y: { type: 'STRING' } },
              required: ['type']
            }
          },
          required: ['sql', 'explanation', 'chart']
        }
      }
    })
  });
  if (!res.ok) {
    let msg = '';
    try { msg = (await res.json()).error?.message || ''; } catch { /* not json */ }
    if (res.status === 400 && /api key/i.test(msg)) throw new Error('Google rejected this API key. Check it and set it again.');
    if (res.status === 403) throw new Error('This key is not allowed to use the Gemini API. Enable it in Google AI Studio.');
    if (res.status === 429) throw new Error('Your key hit its rate limit. Wait a minute and try again.');
    if (res.status === 404) throw new Error(`The model ${MODEL} is not available for this key.`);
    throw new Error(msg || `Gemini returned an error (${res.status}).`);
  }
  const data = await res.json();
  const text = (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || '').join('');
  if (!text) throw new Error('Gemini returned an empty answer. Try rephrasing the question.');
  const out = JSON.parse(text);
  out.sql = (out.sql || '').trim();
  return out;
}

/* ------------------------------------------------------------------ */
/* SQL safety                                                          */
/* ------------------------------------------------------------------ */
function classify(sql) {
  const clean = sql.replace(/--.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '').replace(/'(?:[^']|'')*'/g, "''").trim();
  const stmts = clean.split(';').map((s) => s.trim()).filter(Boolean);
  if (stmts.length === 0) return { kind: 'empty' };
  if (stmts.length > 1) return { kind: 'multi' };
  const verb = stmts[0].split(/\s+/)[0].toUpperCase();
  if (['SELECT', 'WITH', 'VALUES', 'TABLE', 'EXPLAIN', 'SHOW'].includes(verb)) {
    if (/\b(insert|update|delete)\b/i.test(clean) && verb === 'WITH') return { kind: 'write', verb: 'WITH' };
    return { kind: 'read', verb };
  }
  if (['INSERT', 'UPDATE', 'DELETE'].includes(verb)) return { kind: 'write', verb };
  return { kind: 'ddl', verb };
}

async function dryRun(sql, verb) {
  // Run inside a transaction and roll back: exact affected-row count, nothing changes.
  const base = sql.trim().replace(/;\s*$/, '');
  await db.exec('BEGIN');
  try {
    if (verb !== 'WITH' && !/\breturning\b/i.test(base)) {
      try {
        const r = await db.query(`${base} RETURNING *`);
        return { affected: r.affectedRows ?? r.rows.length, rows: r.rows.slice(0, 5), fields: r.fields.map((f) => f.name) };
      } catch {
        await db.exec('ROLLBACK');
        await db.exec('BEGIN');
      }
    }
    const r = await db.query(base);
    return { affected: r.affectedRows ?? 0, rows: [], fields: [] };
  } finally {
    await db.exec('ROLLBACK').catch(() => {});
  }
}

async function execute(sql) {
  const t = performance.now();
  const r = await db.query(sql.trim().replace(/;\s*$/, ''));
  return { rows: r.rows, fields: r.fields.map((f) => f.name), affected: r.affectedRows, ms: performance.now() - t };
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */
const KW = 'SELECT|FROM|WHERE|GROUP|BY|ORDER|LIMIT|JOIN|LEFT|RIGHT|INNER|OUTER|FULL|ON|AS|AND|OR|NOT|IN|IS|NULL|SUM|COUNT|AVG|MIN|MAX|ROUND|CASE|WHEN|THEN|ELSE|END|DISTINCT|HAVING|WITH|INSERT|INTO|VALUES|UPDATE|SET|DELETE|DESC|ASC|BETWEEN|LIKE|ILIKE|EXTRACT|DATE_TRUNC|COALESCE|OVER|PARTITION|UNION|ALL|RETURNING|CAST|FILTER|INTERVAL';
function highlight(sql) {
  return esc(sql).replace(new RegExp(`('(?:[^']|'')*')|\\b(\\d+(?:\\.\\d+)?)\\b|\\b(${KW})\\b`, 'gi'), (m, s, n, k) =>
    s ? `<span class="str">${s}</span>` : n ? `<span class="num">${n}</span>` : `<span class="kw">${k.toUpperCase()}</span>`);
}

const isNum = (v) => typeof v === 'number' || typeof v === 'bigint' || (typeof v === 'string' && /^-?\d+(\.\d+)?$/.test(v));
const cell = (v) => (v === null || v === undefined ? '' : v instanceof Date ? v.toISOString().slice(0, 10) : typeof v === 'object' ? JSON.stringify(v) : String(v));

function sqlBlock(sql) {
  const wrap = document.createElement('div');
  wrap.className = 'sql';
  wrap.innerHTML = `<pre>${highlight(sql)}</pre><div class="tools"><button class="btn ghost small" data-act="copy">Copy</button><button class="btn ghost small" data-act="edit">Edit</button></div>`;
  wrap.querySelector('[data-act="copy"]').onclick = (e) => { navigator.clipboard.writeText(sql); e.target.textContent = 'Copied'; setTimeout(() => (e.target.textContent = 'Copy'), 1200); };
  wrap.querySelector('[data-act="edit"]').onclick = () => { setMode('sql'); $('#q').value = sql; autosize(); $('#q').focus(); };
  return wrap;
}

function resultBlock(res, chart) {
  const frag = document.createDocumentFragment();
  const meta = document.createElement('div');
  meta.className = 'meta';
  if (!res.fields.length) {
    meta.innerHTML = `<span>${res.affected ?? 0} row(s) affected · ${res.ms.toFixed(0)} ms</span>`;
    frag.append(meta);
    return frag;
  }
  const shown = res.rows.slice(0, MAX_ROWS_SHOWN);
  meta.innerHTML = `<span>${res.rows.length.toLocaleString()} row${res.rows.length === 1 ? '' : 's'}${res.rows.length > MAX_ROWS_SHOWN ? ` (showing ${MAX_ROWS_SHOWN})` : ''} · ${res.ms.toFixed(0)} ms</span>`;
  const dl = document.createElement('button');
  dl.className = 'btn ghost small';
  dl.textContent = 'Download CSV';
  dl.onclick = () => {
    const csv = window.Papa.unparse({ fields: res.fields, data: res.rows.map((r) => res.fields.map((f) => cell(r[f]))) });
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })), download: 'result.csv' });
    a.click();
    URL.revokeObjectURL(a.href);
  };
  meta.append(dl);
  frag.append(meta);

  if (chart && chart.type !== 'none' && res.fields.includes(chart.x) && res.fields.includes(chart.y) && res.rows.length > 1 && res.rows.length <= 80 && res.rows.every((r) => isNum(r[chart.y]))) {
    const box = document.createElement('div');
    box.className = 'chart';
    const canvas = document.createElement('canvas');
    box.append(canvas);
    frag.append(box);
    requestAnimationFrame(() => drawChart(canvas, chart, res.rows));
  }

  const wrap = document.createElement('div');
  wrap.className = 'tablewrap';
  wrap.innerHTML = `<table><thead><tr>${res.fields.map((f) => `<th>${esc(f)}</th>`).join('')}</tr></thead><tbody>${shown
    .map((r) => `<tr>${res.fields.map((f) => `<td class="${isNum(r[f]) ? 'n' : ''}">${esc(cell(r[f]))}</td>`).join('')}</tr>`)
    .join('')}</tbody></table>`;
  frag.append(wrap);
  return frag;
}

const PALETTE = ['#7c6cff', '#2fd9c4', '#ff8bd1', '#ffcf6b', '#6cb6ff', '#3ddc84', '#ff8a5c', '#b39dff', '#5ee7ff', '#ffd1f0'];
function drawChart(canvas, chart, rows) {
  const labels = rows.map((r) => cell(r[chart.x]));
  const values = rows.map((r) => Number(r[chart.y]));
  const pie = chart.type === 'pie';
  new window.Chart(canvas, {
    type: chart.type,
    data: {
      labels,
      datasets: [{
        label: chart.y, data: values,
        backgroundColor: pie ? PALETTE : 'rgba(124,108,255,0.55)', borderColor: pie ? '#141722' : '#7c6cff',
        borderWidth: pie ? 2 : 2, borderRadius: chart.type === 'bar' ? 6 : 0, tension: 0.35, fill: chart.type === 'line',
        pointRadius: chart.type === 'line' ? 2 : 0
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: pie, position: 'right', labels: { color: '#c8cbe0' } } },
      scales: pie ? {} : {
        x: { ticks: { color: '#8c93a6', maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#8c93a6' }, grid: { color: 'rgba(255,255,255,0.06)' } }
      }
    }
  });
}

/* ------------------------------------------------------------------ */
/* Conversation flow                                                   */
/* ------------------------------------------------------------------ */
function newTurn(text, isSql) {
  $('#welcome')?.remove();
  const node = $('#tpl-turn').content.firstElementChild.cloneNode(true);
  const q = node.querySelector('.q');
  q.textContent = text;
  if (isSql) q.classList.add('sql');
  $('#thread').append(node);
  scrollDown();
  const a = node.querySelector('.a');
  return {
    a,
    status: (t) => { a.querySelector('.status-text').textContent = t; },
    done: () => a.querySelector('.status')?.remove(),
    add: (el) => { a.append(el); scrollDown(); },
    text: (t, cls = 'explain') => { const p = document.createElement('p'); p.className = cls; p.textContent = t; a.append(p); scrollDown(); }
  };
}
const scrollDown = () => { const t = $('#thread'); t.scrollTop = t.scrollHeight; };

async function handle(input) {
  if (busy || !input.trim() || !db) return;
  busy = true;
  $('#send').disabled = true;
  const isSql = mode === 'sql';
  const turn = newTurn(input, isSql);
  try {
    if (isSql) {
      await runWithChecks(turn, input, null, true);
    } else {
      if (!apiKey) {
        turn.done();
        turn.text('Add your Gemini API key on the left first, or switch to "Write SQL" to query without AI.', 'err');
        $('#key-input').focus();
        return;
      }
      turn.status(`Asking ${MODEL}…`);
      let ai = await askGemini(input);
      if (!ai.sql) { turn.done(); turn.text(ai.explanation || 'I could not turn that into a query for this database.'); return; }
      turn.done();
      turn.text(ai.explanation);
      turn.add(sqlBlock(ai.sql));
      const ok = await runWithChecks(turn, ai.sql, ai.chart, false, async (error) => {
        turn.text(`That query failed (${error}). Asking Gemini to fix it…`, 'note');
        const fixed = await askGemini(input, { sql: ai.sql, error });
        if (!fixed.sql) return null;
        ai = fixed;
        turn.add(sqlBlock(fixed.sql));
        return fixed;
      });
      if (ok) history.push({ q: input, sql: ai.sql, explanation: ai.explanation });
    }
  } catch (e) {
    turn.done();
    turn.text(e.message || String(e), 'err');
  } finally {
    busy = false;
    $('#send').disabled = false;
  }
}

async function runWithChecks(turn, sql, chart, userWritten, repair) {
  const c = classify(sql);
  turn.done();
  if (c.kind === 'empty') { turn.text('Nothing to run.', 'err'); return false; }
  if (c.kind === 'multi') { turn.text('Run one statement at a time.', 'err'); return false; }
  if (c.kind === 'ddl' && !userWritten) { turn.text(`Blocked: the model suggested a ${c.verb} statement. Schema changes are only allowed in "Write SQL" mode.`, 'err'); return false; }

  if (c.kind === 'write' || c.kind === 'ddl') {
    let preview;
    try { preview = c.kind === 'write' ? await dryRun(sql, c.verb) : null; } catch (e) { turn.text(e.message, 'err'); return false; }
    const go = await confirmBox(turn, c, preview);
    if (!go) { turn.text('Cancelled. Nothing was changed.', 'note'); return false; }
  }

  try {
    const res = await execute(sql);
    turn.add(resultBlock(res, chart));
    if (c.kind !== 'read') await loadSchema();
    return true;
  } catch (e) {
    if (repair) {
      const fixed = await repair(e.message);
      if (fixed) return runWithChecks(turn, fixed.sql, fixed.chart, false, null);
    }
    turn.text(e.message, 'err');
    return false;
  }
}

function confirmBox(turn, c, preview) {
  return new Promise((resolve) => {
    const box = document.createElement('div');
    box.className = 'warnbox';
    const what = c.kind === 'ddl'
      ? `This ${c.verb} statement changes the database structure.`
      : `This ${c.verb} will change ${preview.affected} row${preview.affected === 1 ? '' : 's'}.`;
    box.innerHTML = `<div><b>Confirm change.</b> ${esc(what)} Changes only affect the database in this tab, and "Reset sample data" undoes them.</div>`;
    if (preview && preview.rows.length) {
      const p = document.createElement('div');
      p.append(resultBlock({ rows: preview.rows, fields: preview.fields, ms: 0 }, null));
      p.querySelector('.meta span').textContent = `Preview of affected rows (first ${preview.rows.length})`;
      p.querySelector('.meta button')?.remove();
      box.append(p);
    }
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<button class="btn danger small">Run it</button><button class="btn ghost small">Cancel</button>`;
    const [yes, no] = row.querySelectorAll('button');
    const end = (v) => { yes.disabled = no.disabled = true; resolve(v); };
    yes.onclick = () => end(true);
    no.onclick = () => end(false);
    box.append(row);
    turn.add(box);
  });
}

/* ------------------------------------------------------------------ */
/* CSV upload                                                          */
/* ------------------------------------------------------------------ */
function ident(raw, taken) {
  let s = String(raw || 'col').toLowerCase().normalize('NFKD').replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'col';
  if (/^\d/.test(s)) s = `c_${s}`;
  let out = s, i = 2;
  while (taken.has(out)) out = `${s}_${i++}`;
  taken.add(out);
  return out;
}

function inferType(values) {
  const v = values.filter((x) => x !== null && x !== undefined && String(x).trim() !== '').slice(0, 2000).map((x) => String(x).trim());
  if (!v.length) return 'TEXT';
  if (v.every((x) => /^-?\d{1,15}$/.test(x))) return 'BIGINT';
  if (v.every((x) => /^-?\d*\.?\d+(e[-+]?\d+)?$/i.test(x))) return 'DOUBLE PRECISION';
  if (v.every((x) => /^(true|false)$/i.test(x))) return 'BOOLEAN';
  if (v.every((x) => /^\d{4}-\d{2}-\d{2}$/.test(x))) return 'DATE';
  return 'TEXT';
}

async function importCSV(file) {
  setPill('#db-status', 'Importing…', 'warn');
  const parsed = await new Promise((resolve, reject) =>
    window.Papa.parse(file, { header: true, skipEmptyLines: true, complete: resolve, error: reject }));
  const rows = parsed.data.slice(0, 50000);
  if (!rows.length) throw new Error('That CSV has no rows.');
  const taken = new Set();
  const headers = parsed.meta.fields;
  const cols = headers.map((h) => ({ src: h, name: ident(h, taken) }));
  cols.forEach((c) => (c.type = inferType(rows.map((r) => r[c.src]))));
  const tables = new Set(schema.map((t) => t.name));
  const table = ident(file.name.replace(/\.[^.]+$/, ''), tables);

  await db.exec(`CREATE TABLE ${qi(table)} (${cols.map((c) => `${qi(c.name)} ${c.type}`).join(', ')})`);
  const per = Math.max(1, Math.floor(30000 / cols.length));
  for (let i = 0; i < rows.length; i += per) {
    const chunk = rows.slice(i, i + per);
    const params = [];
    const tuples = chunk.map((r) => `(${cols.map((c) => {
      const raw = r[c.src];
      params.push(raw === undefined || String(raw).trim() === '' ? null : String(raw).trim());
      return `$${params.length}`;
    }).join(',')})`);
    await db.query(`INSERT INTO ${qi(table)} (${cols.map((c) => qi(c.name)).join(',')}) VALUES ${tuples.join(',')}`, params);
  }
  await loadSchema();
  return { table, rows: rows.length, cols: cols.length, truncated: parsed.data.length > rows.length };
}

/* ------------------------------------------------------------------ */
/* UI wiring                                                           */
/* ------------------------------------------------------------------ */
function setPill(sel, text, cls) {
  const el = $(sel);
  el.textContent = text;
  el.className = `pill ${cls || ''}`;
}

function setMode(m) {
  mode = m;
  document.querySelectorAll('.mode button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  const q = $('#q');
  q.classList.toggle('sqlmode', m === 'sql');
  q.placeholder = m === 'sql' ? 'SELECT * FROM products ORDER BY price DESC LIMIT 5;' : 'e.g. Which 5 products earned the most revenue in 2026?';
}

function autosize() {
  const q = $('#q');
  q.style.height = 'auto';
  q.style.height = Math.min(q.scrollHeight, 180) + 'px';
}

$('#key-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const input = $('#key-input');
  const v = input.value.trim();
  input.value = '';
  if (!v) return;
  apiKey = v;
  setPill('#key-status', 'Ready · in memory', 'ok');
  $('#key-forget').hidden = false;
  input.placeholder = 'Key set. Paste a new one to replace it';
});
$('#key-forget').addEventListener('click', () => {
  apiKey = null;
  setPill('#key-status', 'Not set', '');
  $('#key-forget').hidden = true;
  $('#key-input').placeholder = 'Paste your Gemini API key';
});

document.querySelectorAll('.mode button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
$('#q').addEventListener('input', autosize);
$('#q').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); }
});
$('#composer').addEventListener('submit', (e) => {
  e.preventDefault();
  const v = $('#q').value;
  if (!v.trim() || busy) return;
  $('#q').value = '';
  autosize();
  handle(v);
});

$('#csv').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  const turn = newTurn(`Upload ${file.name}`, false);
  turn.status('Importing…');
  try {
    const r = await importCSV(file);
    turn.done();
    turn.text(`Imported ${r.rows.toLocaleString()} rows into table "${r.table}" (${r.cols} columns)${r.truncated ? ', first 50,000 rows only' : ''}. Ask anything about it.`);
  } catch (err) {
    turn.done();
    turn.text(`Import failed: ${err.message}`, 'err');
    await loadSchema().catch(() => {});
  }
});

$('#reset-db').addEventListener('click', async () => {
  if (busy) return;
  busy = true;
  try { await startDb(true); history.length = 0; } finally { busy = false; }
});

const EXAMPLES = [
  'Top 5 products by revenue in 2026',
  'Monthly revenue trend for 2026',
  'Revenue share by category',
  'Who are our 10 biggest customers?',
  'Return rate by sales channel',
  'Average order value by country'
];
$('#chips').innerHTML = EXAMPLES.map((x) => `<button class="chip" type="button">${esc(x)}</button>`).join('');
$('#chips').addEventListener('click', (e) => {
  const b = e.target.closest('.chip');
  if (!b) return;
  setMode('ask');
  handle(b.textContent);
});

startDb(true).catch((e) => {
  setPill('#db-status', 'Failed', 'warn');
  $('#schema').innerHTML = `<p class="err">Could not start the in-browser database: ${esc(e.message)}</p>`;
});
