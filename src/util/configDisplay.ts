/**
 * Compact, punctuation-free rendering of provider config values, shared by the web UI
 * (inline config line in each results column header) and run titles.
 */

/** Short names for long config keys. Keys not listed here are shown as-is. */
export const CONFIG_KEY_ABBREVIATIONS: Record<string, string> = {
  num_predict: 'num_pred',
  showThinking: 'show_think',
  temperature: 'temp',
};

export function abbreviateConfigKey(key: string): string {
  return CONFIG_KEY_ABBREVIATIONS[key] ?? key;
}

function formatConfigValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map(formatConfigValue).join(' ');
  }
  if (value !== null && typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

/**
 * Flatten config into [key, value] pairs, sorted by key. Nested objects become
 * `parent.child` keys; each path segment is abbreviated.
 */
export function flattenConfig(config: Record<string, unknown>, prefix = ''): [string, string][] {
  const pairs: [string, string][] = [];
  for (const key of Object.keys(config).sort()) {
    const value = config[key];
    const name = prefix + abbreviateConfigKey(key);
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      pairs.push(...flattenConfig(value as Record<string, unknown>, `${name}.`));
    } else if (value !== undefined && typeof value !== 'function') {
      pairs.push([name, formatConfigValue(value)]);
    }
  }
  return pairs;
}

/**
 * Render config as one line of `key value` pairs separated by single spaces, e.g.
 * `num_pred 2048 seed 512 temp 0.1 think low top_p 0.9`.
 */
export function formatConfigInline(config: Record<string, unknown> | undefined): string {
  if (!config) {
    return '';
  }
  return flattenConfig(config)
    .map(([key, value]) => `${key} ${value}`)
    .join(' ');
}

/** Second provider id segment that names an API type rather than part of the model name. */
const API_TYPES = new Set([
  'chat',
  'completion',
  'completions',
  'embedding',
  'embeddings',
  'responses',
]);

/** 'ollama:chat:qwen3-vl:4b' -> 'qwen3-vl:4b'; 'openai:gpt-4o' -> 'gpt-4o'. */
export function modelName(providerId: string): string {
  const parts = providerId.split(':');
  if (parts.length >= 3 && API_TYPES.has(parts[1])) {
    return parts.slice(2).join(':');
  }
  if (parts.length >= 2) {
    return parts.slice(1).join(':');
  }
  return providerId;
}

/** Local time as one number with no punctuation: 20261014143900. */
export function formatCompactTimestamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return (
    String(date.getFullYear()) +
    pad(date.getMonth() + 1) +
    pad(date.getDate()) +
    pad(date.getHours()) +
    pad(date.getMinutes()) +
    pad(date.getSeconds())
  );
}

export interface TitleProvider {
  id: string;
  config?: Record<string, unknown>;
}

/**
 * Run title: compact local timestamp, then each value that differs across providers as
 * `key v1 v2 …`, e.g. `20261014143900 model qwen3-vl:4b qwen3.5:4b temp 0.2 0.8`.
 * When nothing differs (e.g. a single provider), every value is listed instead.
 */
export function buildAutoTitle(providers: TitleProvider[], createdAt: Date): string {
  const perProvider = providers.map(
    (p) => new Map<string, string>([['model', modelName(p.id)], ...flattenConfig(p.config ?? {})]),
  );
  const keys: string[] = [];
  for (const entries of perProvider) {
    for (const key of entries.keys()) {
      if (!keys.includes(key)) {
        keys.push(key);
      }
    }
  }

  const valuesByKey = keys.map((key) => {
    const values = perProvider.map((entries) => entries.get(key));
    const differs = new Set(values).size > 1;
    const unique = [...new Set(values.filter((v): v is string => v !== undefined))];
    return { key, unique, differs };
  });
  const varying = valuesByKey.filter((k) => k.differs);
  const shown = varying.length > 0 ? varying : valuesByKey;

  return [formatCompactTimestamp(createdAt), ...shown.map((k) => `${k.key} ${k.unique.join(' ')}`)]
    .join(' ')
    .trim();
}
