#!/usr/bin/env node
/**
 * translate-all.js
 * Translate all docs/en.md → docs/zh.md using Google Translate.
 * Safe to re-run: skips lessons that already have a zh.md.
 *
 * Usage: node scripts/translate-all.js [from-phase-num] [--dry]
 *   --dry    preview what would be translated without writing files
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// ── Google Translate via REST (no API key needed) ──────────────────────
const GOOGLE_TRANSLATE_URL = 'https://translate.googleapis.com/translate_a/single';
const USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36';

function translateText(text, sourceLang = 'en', targetLang = 'zh-CN') {
  return new Promise((resolve, reject) => {
    if (!text || !text.trim()) { resolve(''); return; }

    const params = new URLSearchParams({
      client: 'gtx',
      sl: sourceLang,
      tl: targetLang,
      dt: 't',
      q: text,
    });

    const url = `${GOOGLE_TRANSLATE_URL}?${params.toString()}`;
    const options = {
      headers: { 'User-Agent': USER_AGENT },
      method: 'GET',
    };

    const req = https.request(url, options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          if (json[0]) {
            resolve(json[0].map(s => s[0]).join(''));
          } else {
            resolve('');
          }
        } catch (e) {
          reject(e);
        }
      });
    });

    req.on('error', reject);
    req.setTimeout(15000, () => { req.destroy(); reject(new Error('timeout')); });
    req.end();
  });
}

// ── Markdown-aware translation ──────────────────────────────────────────
async function translateMarkdown(mdText) {
  // Split into blocks: code blocks, headings, blockquotes, regular paragraphs
  const blocks = [];
  const codeBlockRegex = /```[\s\S]*?```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(mdText)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({ type: 'code', content: mdText.slice(lastIndex, match.index) });
    }
    blocks.push({ type: 'code', content: match[0] });
    lastIndex = codeBlockRegex.lastIndex;
  }
  if (lastIndex < mdText.length) {
    blocks.push({ type: 'text', content: mdText.slice(lastIndex) });
  }

  const translated = await Promise.all(blocks.map(async (block) => {
    if (block.type === 'code') return block.content;

    // For text blocks, translate heading lines separately for accuracy
    const lines = block.content.split('\n');
    const translatedLines = [];
    for (const line of lines) {
      if (!line.trim()) { translatedLines.push(line); continue; }
      // Table rows and list items translate as whole
      translatedLines.push(await translateText(line));
    }
    return translatedLines.join('\n');
  }));

  return translated.join('');
}

// ── Traverse phases and translate ──────────────────────────────────────
function findLessons(baseDir) {
  const lessons = [];
  if (!fs.existsSync(baseDir)) return lessons;
  const phaseDirs = fs.readdirSync(baseDir, { withFileTypes: true });
  for (const phaseEntry of phaseDirs) {
    if (!phaseEntry.isDirectory()) continue;
    const phasePath = path.join(baseDir, phaseEntry.name);
    const lessonDirs = fs.readdirSync(phasePath, { withFileTypes: true });
    for (const lessonEntry of lessonDirs) {
      if (!lessonEntry.isDirectory()) continue;
      const docsDir = path.join(phasePath, lessonEntry.name, 'docs');
      const enPath = path.join(docsDir, 'en.md');
      const zhPath = path.join(docsDir, 'zh.md');
      lessons.push({
        phase: phaseEntry.name,
        lesson: lessonEntry.name,
        enPath,
        zhPath,
        exists: fs.existsSync(enPath),
      });
    }
  }
  return lessons;
}

// ── Rate-limited translator ────────────────────────────────────────────
let requestCount = 0;
let lastRequestTime = 0;
const MIN_INTERVAL_MS = 350;

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function translateWithRetry(text, retries = 3) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      // Rate limit
      const elapsed = Date.now() - lastRequestTime;
      if (elapsed < MIN_INTERVAL_MS) {
        await sleep(MIN_INTERVAL_MS - elapsed);
      }
      lastRequestTime = Date.now();
      const result = await translateText(text);
      return result;
    } catch (err) {
      if (attempt === retries - 1) throw err;
      await sleep(1500 * (attempt + 1));
    }
  }
}

// ── Main ───────────────────────────────────────────────────────────────
const REPO_ROOT = path.resolve(__dirname, '..');
const PHASES_DIR = path.join(REPO_ROOT, 'phases');
const DRY = process.argv.includes('--dry');
const START_PHASE = parseInt(process.argv.find(a => /^\d+$/.test(a)) || '0');

const lessons = findLessons(PHASES_DIR);

console.log(`\n📚 Translating AI Engineering lessons → Chinese (zh-CN)`);
console.log(`   Total lesson docs found: ${lessons.length}`);
console.log(`   Dry run: ${DRY ? 'YES (no files written)' : 'NO (will write zh.md)'}`);
console.log(`   Start from phase: ${START_PHASE}\n`);

let done = 0, skipped = 0, failed = 0;
const failedLessons = [];

async function processAll() {
  for (const lesson of lessons) {
    const phaseNum = parseInt(lesson.phase.split('-')[0]);
    if (phaseNum < START_PHASE) { skipped++; continue; }

    // Skip if zh.md already exists
    if (fs.existsSync(lesson.zhPath)) {
      skipped++;
      process.stdout.write(`\r⏭  [${done + skipped + failed}] ${lesson.phase}/${lesson.lesson} — already translated, skipping`);
      continue;
    }

    if (!lesson.exists) {
      skipped++;
      continue;
    }

    try {
      const md = fs.readFileSync(lesson.enPath, 'utf-8');
      const translated = await translateMarkdown(md);

      if (!DRY) {
        // Ensure docs dir exists
        const docsDir = path.dirname(lesson.zhPath);
        if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir, { recursive: true });
        fs.writeFileSync(lesson.zhPath, translated, 'utf-8');
      }

      done++;
      process.stdout.write(`\r🌐 [${done + skipped + failed}] ${lesson.phase}/${lesson.lesson} — translated ✅`);
    } catch (err) {
      failed++;
      failedLessons.push({ lesson, err: err.message });
      process.stdout.write(`\r❌ [${done + skipped + failed}] ${lesson.phase}/${lesson.lesson} — FAILED: ${err.message}`);
    }

    // Progress dot every 10
    if ((done + skipped) % 10 === 0 && !DRY) {
      process.stdout.write(' .');
    }
  }
}

processAll().then(() => {
  console.log('\n');
  console.log('─'.repeat(50));
  console.log(`  ✅ Done: ${done} translated`);
  console.log(`  ⏭  Skipped (already exists): ${skipped}`);
  console.log(`  ❌ Failed: ${failed}`);
  if (failed > 0) {
    console.log('\n  Failed lessons:');
    failedLessons.forEach(f => console.log(`    - ${f.lesson.phase}/${f.lesson.lesson}: ${f.err}`));
    console.log('\n  Re-run to retry failed: node scripts/translate-all.js');
  }
  console.log('─'.repeat(50));
  process.exit(failed > 0 ? 1 : 0);
}).catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});