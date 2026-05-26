#!/usr/bin/env node
/**
 * Local server for AI Engineering from Scratch
 * - Serves static site (same as Vercel)
 * - Provides /browse and /file routes for local file exploration
 * - All lesson content is embedded in lessons-content.js (no network calls)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const ROOT = __dirname;
const PORT = 3000;
const SITE_DIR = path.join(ROOT, 'site');

const mimeTypes = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.md': 'text/markdown',
  '.txt': 'text/plain',
};

function serveFile(res, filePath, mime) {
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  });
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function buildBrowseHtml(relPath, entries, parentPath) {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>📂 ${escapeHtml(relPath || 'AI Engineering')}</title>
  <style>
    body { font-family: -apple-system, system-ui; padding: 24px; background: #0a0a0a; color: #e0e0e0; }
    h1 { color: #3553ff; font-size: 1.1rem; margin-bottom: 20px; }
    a { color: #7faaff; text-decoration: none; display: block; padding: 8px 0; border-bottom: 1px solid #222; }
    a:hover { color: #fff; }
    .back { color: #888; margin-bottom: 20px; }
    .folder { color: #fbbf24; }
  </style>
</head>
<body>
  <h1>📂 /${escapeHtml(relPath || 'ai-engineering-from-scratch')}</h1>
  ${parentPath ? `<a href="/browse?path=${escapeHtml(parentPath)}" class="back">⬆ Back</a>` : ''}
  ${entries.map(e => {
    const href = e.isDirectory()
      ? `/browse?path=${relPath ? relPath + '/' + e.name : e.name}`
      : `/file?path=${relPath ? relPath + '/' + e.name : e.name}`;
    const icon = e.isDirectory() ? '📁' : '📄';
    const cls = e.isDirectory() ? 'folder' : '';
    return `<a href="${href}" class="${cls}">${icon} ${e.name}</a>`;
  }).join('')}
</body>
</html>`;
}

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);
  const pathname = parsed.pathname;
  const query = parsed.query;

  // Root → index.html
  if (pathname === '/') {
    serveFile(res, path.join(SITE_DIR, 'index.html'), 'text/html');
    return;
  }

  // Lesson page (fully static with embedded content)
  if (pathname === '/lesson.html') {
    serveFile(res, path.join(SITE_DIR, 'lesson.html'), 'text/html');
    return;
  }

  // File browse
  if (pathname === '/browse') {
    const relPath = (query.path || '').replace(/\/$/, '');
    const fullPath = path.join(ROOT, relPath);
    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isDirectory()) {
      res.writeHead(404); res.end('Not a directory'); return;
    }
    const entries = fs.readdirSync(fullPath, { withFileTypes: true });
    const parentPath = relPath.split('/').slice(0, -1).join('/');
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(buildBrowseHtml(relPath, entries, parentPath));
    return;
  }

  // Individual file view
  if (pathname === '/file') {
    const relPath = query.path || '';
    const fullPath = path.join(ROOT, relPath);
    if (!fs.existsSync(fullPath)) { res.writeHead(404); res.end('not found'); return; }

    const ext = path.extname(fullPath);
    const mime = mimeTypes[ext] || 'text/plain';

    if (mime.startsWith('text/') || ext === '.md') {
      const content = fs.readFileSync(fullPath, 'utf-8');
      const escapedPath = relPath.replace(/'/g, "\\'");
      const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${escapeHtml(relPath.split('/').pop())}</title>
  <style>
    body { font-family: -apple-system, system-ui; padding: 24px; background: #0a0a0a; color: #e0e0e0; max-width: 900px; margin: 0 auto; }
    .toolbar { position: sticky; top: 0; background: #0a0a0a; padding: 12px 0; border-bottom: 1px solid #333; margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .toolbar span { color: #888; font-size: 13px; }
    button { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-family: monospace; }
    .btn-finder { background: #3553ff; color: white; }
    .btn-vscode { background: #333; color: white; }
    pre { background: #111; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-all; }
    a.back { color: #888; display: inline-block; margin-bottom: 16px; font-size: 14px; }
  </style>
</head>
<body>
  <a href="/browse?path=${escapeHtml(path.dirname(relPath))}" class="back">⬅ Back</a>
  <div class="toolbar">
    <span>📄 ${escapeHtml(relPath)}</span>
    <button class="btn-finder" onclick="openFinder()">📂 Finder</button>
    <button class="btn-vscode" onclick="openVSCode()">💻 VS Code</button>
  </div>
  <pre><code>${escapeHtml(content)}</code></pre>
  <script>
    var filePath = '${escapedPath}';
    var rootDir = '${ROOT.replace(/'/g, "\\'")}';
    function openFinder() { window.open('file://'+rootDir+'/'+filePath, '_self'); }
    function openVSCode() { window.open('vscode://file/'+rootDir+'/'+filePath, '_blank'); }
  </script>
</body>
</html>`;
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    } else {
      serveFile(res, fullPath, mime);
    }
    return;
  }

  // Static files from site/
  let filePath = path.join(SITE_DIR, pathname);
  if (!fs.existsSync(filePath)) filePath = path.join(ROOT, pathname);
  if (!fs.existsSync(filePath)) { res.writeHead(404); res.end('Not found: ' + pathname); return; }
  const ext = path.extname(filePath);
  const mime = mimeTypes[ext] || 'application/octet-stream';
  serveFile(res, filePath, mime);
});

server.listen(PORT, () => {
  console.log('');
  console.log('  🎓 AI Engineering from Scratch — Local Server');
  console.log('');
  console.log('  📖 Course site:   http://localhost:' + PORT);
  console.log('  📂 File browser:  http://localhost:' + PORT + '/browse');
  console.log('');
  console.log('  ✅ Lesson content embedded (no GitHub calls)');
  console.log('  🌐 Language toggle in each lesson (EN / 中文)');
  console.log('');
});