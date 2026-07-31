import reposJson from '../data/repos.json';
import topicsJson from '../data/topics.json';
import tagsJson from '../data/tags.json';
import statsJson from '../data/stats.json';

export interface Related {
  repo_id: string;
  score: number;
}

export interface Repo {
  id: string;
  full_name: string;
  name: string;
  owner: string;
  description: string | null;
  html_url: string;
  homepage: string | null;
  language: string | null;
  stargazers_count: number;
  forks_count: number;
  open_issues_count: number;
  topics: string[];
  license: string | null;
  archived: boolean;
  fork: boolean;
  created_at: string;
  pushed_at: string;
  starred_at: string;
  topic_slug: string | null;
  tags: string[];
  related: Related[];
  readme_excerpt: string;
}

export interface Topic {
  slug: string;
  label: string;
  keywords: string[];
  size: number;
  parent_group: number;
  parent_label: string;
  repo_ids: string[];
}

export interface Tag {
  slug: string;
  label: string;
  repo_ids: string[];
}

export interface Stats {
  total_repos: number;
  total_topics: number;
  total_tags: number;
  archived_count: number;
  fork_count: number;
  language_histogram: Record<string, number>;
  license_histogram: Record<string, number>;
  stars_by_month: Record<string, number>;
}

export const repos = reposJson as unknown as Repo[];
export const topics = topicsJson as unknown as Topic[];
export const tags = tagsJson as unknown as Tag[];
export const stats = statsJson as unknown as Stats;

const repoById = new Map<string, Repo>(repos.map((r) => [r.id, r]));
const topicBySlug = new Map<string, Topic>(topics.map((t) => [t.slug, t]));
const tagBySlug = new Map<string, Tag>(tags.map((t) => [t.slug, t]));

export function getRepo(id: string): Repo | undefined {
  return repoById.get(id);
}

export function getTopic(slug: string): Topic | undefined {
  return topicBySlug.get(slug);
}

export function getTag(slug: string): Tag | undefined {
  return tagBySlug.get(slug);
}

export function reposFor(ids: string[]): Repo[] {
  return ids.map((id) => repoById.get(id)).filter((r): r is Repo => Boolean(r));
}

export function relatedRepos(repo: Repo, limit = 12): Repo[] {
  return repo.related
    .slice(0, limit)
    .map((r) => repoById.get(r.repo_id))
    .filter((r): r is Repo => Boolean(r));
}

export function siblingTopics(topic: Topic): Topic[] {
  return topics.filter((t) => t.parent_group === topic.parent_group && t.slug !== topic.slug);
}

export interface ParentGroup {
  id: number;
  label: string;
  topics: Topic[];
  size: number;
}

export function parentGroups(): ParentGroup[] {
  const byGroup = new Map<number, Topic[]>();
  for (const t of topics) {
    const list = byGroup.get(t.parent_group) ?? [];
    list.push(t);
    byGroup.set(t.parent_group, list);
  }
  return Array.from(byGroup.entries())
    .map(([id, ts]) => ({
      id,
      label: ts[0]?.parent_label ?? `Group ${id}`,
      topics: ts.sort((a, b) => b.size - a.size),
      size: ts.reduce((sum, t) => sum + t.size, 0),
    }))
    .sort((a, b) => b.size - a.size);
}

export function recentlyStarred(limit = 20): Repo[] {
  return [...repos]
    .sort((a, b) => (a.starred_at < b.starred_at ? 1 : -1))
    .slice(0, limit);
}
