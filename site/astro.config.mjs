// @ts-check
import { defineConfig } from 'astro/config';

// Configurable so a future custom-domain switch is a one-line env change
// rather than a code change. Defaults match a GitHub Pages *project* page
// (https://<user>.github.io/<repo>/) rather than a user/org page.
const site = process.env.SITE_URL || 'https://doraeminemon.github.io';
const base = process.env.SITE_BASE ?? '/github-stars-pages';

export default defineConfig({
  site,
  base,
  output: 'static',
});
