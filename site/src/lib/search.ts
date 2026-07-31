import MiniSearch from 'minisearch';
import { url } from './url';

export interface SearchDoc {
  id: string;
  full_name: string;
  description: string;
  tags: string[];
  topic_slug: string | null;
}

let miniSearchPromise: Promise<MiniSearch<SearchDoc>> | null = null;
let docsById: Map<string, SearchDoc> | null = null;

export function loadLexicalIndex(): Promise<MiniSearch<SearchDoc>> {
  if (!miniSearchPromise) {
    miniSearchPromise = (async () => {
      const res = await fetch(url('/search/index.json'));
      const docs: SearchDoc[] = await res.json();
      docsById = new Map(docs.map((d) => [d.id, d]));
      const ms = new MiniSearch<SearchDoc>({
        fields: ['full_name', 'description', 'tags'],
        storeFields: ['full_name', 'description', 'tags', 'topic_slug'],
        searchOptions: { prefix: true, fuzzy: 0.2, boost: { full_name: 2 } },
      });
      ms.addAll(docs);
      return ms;
    })();
  }
  return miniSearchPromise;
}

export function getDoc(id: string): SearchDoc | undefined {
  return docsById?.get(id);
}

interface VectorIndex {
  n: number;
  dim: number;
  order: string[];
  vectors: Int8Array; // flattened n*dim, symmetric int8 quantized (v * 127)
}

let vectorIndexPromise: Promise<VectorIndex> | null = null;

function loadVectorIndex(): Promise<VectorIndex> {
  if (!vectorIndexPromise) {
    vectorIndexPromise = (async () => {
      const [meta, buf] = await Promise.all([
        fetch(url('/search/vectors.json')).then((r) => r.json()),
        fetch(url('/search/vectors.bin')).then((r) => r.arrayBuffer()),
      ]);
      return { ...meta, vectors: new Int8Array(buf) } as VectorIndex;
    })();
  }
  return vectorIndexPromise;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let extractorPromise: Promise<any> | null = null;

/** Lazily loads the ~23MB quantized ONNX model, only on first search —
 * never blocks lexical search, which works with zero model download. */
async function loadExtractor() {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      // No CDN/HF Hub calls at runtime: the model was fetched at build time
      // (scripts/fetch_onnx.py) into public/models/, served from this
      // site's own origin.
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = url('/models/');
      return pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
    })();
  }
  return extractorPromise;
}

export async function isExtractorReady(): Promise<boolean> {
  return extractorPromise !== null;
}

export async function embedQuery(text: string): Promise<Float32Array> {
  const extractor = await loadExtractor();
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return output.data as Float32Array;
}

export interface ScoredResult {
  id: string;
  score: number;
}

export async function semanticSearch(query: string, limit = 30): Promise<ScoredResult[]> {
  const [queryVec, index] = await Promise.all([embedQuery(query), loadVectorIndex()]);
  const { n, dim, order, vectors } = index;
  const scores: ScoredResult[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const offset = i * dim;
    let dot = 0;
    for (let d = 0; d < dim; d++) {
      dot += queryVec[d] * (vectors[offset + d] / 127);
    }
    scores[i] = { id: order[i], score: dot };
  }
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, limit);
}

export function lexicalSearch(ms: MiniSearch<SearchDoc>, query: string, limit = 30): ScoredResult[] {
  return ms
    .search(query)
    .slice(0, limit)
    .map((r) => ({ id: String(r.id), score: r.score }));
}

/** Reciprocal Rank Fusion: combines rankers with very differently-shaped
 * score distributions (BM25-ish lexical vs. cosine similarity) without
 * needing to normalize or threshold either one. */
export function fuseRRF(rankLists: ScoredResult[][], k = 60): ScoredResult[] {
  const fused = new Map<string, number>();
  for (const list of rankLists) {
    list.forEach((item, rank) => {
      fused.set(item.id, (fused.get(item.id) ?? 0) + 1 / (k + rank + 1));
    });
  }
  return Array.from(fused.entries())
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}
