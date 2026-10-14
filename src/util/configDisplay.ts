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
    } else if (value !== undefined) {
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
