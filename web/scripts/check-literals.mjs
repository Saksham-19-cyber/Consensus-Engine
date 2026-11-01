import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const componentsDir = path.resolve(__dirname, '../components');

const BANNED_LITERALS = [
  '0.9156', '0.0654', '0.3990', '0.0725', '0.3885',
  '0.9082', '0.9525', '0.4532', '0.9991', '0.5481',
  '0.8205', '0.2887', '0.0017', '0.0015', '0.8312',
  '0.4472', '0.918', '0.364', '0.6266', '0.0753',
];

let failed = false;

function scanDir(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      scanDir(fullPath);
    } else if (entry.isFile() && (entry.name.endsWith('.tsx') || entry.name.endsWith('.ts'))) {
      const content = fs.readFileSync(fullPath, 'utf8');
      for (const literal of BANNED_LITERALS) {
        if (content.includes(literal)) {
          console.error(`[FAIL] Found hardcoded benchmark literal "${literal}" in ${fullPath}`);
          failed = true;
        }
      }
    }
  }
}

console.log('Scanning web/components for hardcoded benchmark literals...');
scanDir(componentsDir);

if (failed) {
  console.error('\nERROR: Hardcoded benchmark literals detected in web components! All numbers must be computed dynamically from API endpoints.');
  process.exit(1);
} else {
  console.log('[PASS] No hardcoded benchmark literals found in web/components.');
}
