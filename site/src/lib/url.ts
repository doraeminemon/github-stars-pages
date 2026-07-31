/** Prefix an absolute app path with Astro's configured base path, so links
 * work whether the site is served at `/` or a GitHub Pages project path
 * like `/github-stars-pages/`. */
export function url(path: string): string {
  const base = import.meta.env.BASE_URL;
  const trimmedBase = base.endsWith('/') ? base.slice(0, -1) : base;
  const trimmedPath = path.startsWith('/') ? path : `/${path}`;
  return `${trimmedBase}${trimmedPath}`;
}
