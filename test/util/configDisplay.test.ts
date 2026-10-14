import { describe, expect, it } from 'vitest';
import {
  abbreviateConfigKey,
  buildAutoTitle,
  formatCompactTimestamp,
  formatConfigInline,
  modelName,
} from '../../src/util/configDisplay';

describe('abbreviateConfigKey', () => {
  it('shortens known keys and leaves others as-is', () => {
    expect(abbreviateConfigKey('temperature')).toBe('temp');
    expect(abbreviateConfigKey('num_predict')).toBe('num_pred');
    expect(abbreviateConfigKey('showThinking')).toBe('show_think');
    expect(abbreviateConfigKey('top_p')).toBe('top_p');
  });
});

describe('formatConfigInline', () => {
  it('sorts keys and separates everything with single spaces', () => {
    expect(
      formatConfigInline({
        top_p: 0.9,
        temperature: 0.1,
        think: 'low',
        seed: 512,
        num_predict: 2048,
      }),
    ).toBe('num_pred 2048 seed 512 temp 0.1 think low top_p 0.9');
  });

  it('keeps false and 0 values', () => {
    expect(formatConfigInline({ think: false, temperature: 0 })).toBe('temp 0 think false');
  });

  it('flattens nested objects and joins arrays', () => {
    expect(formatConfigInline({ options: { temperature: 0.5 }, stop: ['a', 'b'] })).toBe(
      'options.temp 0.5 stop a b',
    );
  });

  it('returns an empty string for missing or empty config', () => {
    expect(formatConfigInline(undefined)).toBe('');
    expect(formatConfigInline({})).toBe('');
  });
});

describe('modelName', () => {
  it('drops the provider prefix and API type', () => {
    expect(modelName('ollama:chat:qwen3-vl:4b')).toBe('qwen3-vl:4b');
    expect(modelName('openai:gpt-4o')).toBe('gpt-4o');
    expect(modelName('echo')).toBe('echo');
  });
});

describe('formatCompactTimestamp', () => {
  it('formats local time as YYYYMMDDhhmmss', () => {
    expect(formatCompactTimestamp(new Date(2026, 9, 14, 14, 39, 0))).toBe('20261014143900');
    expect(formatCompactTimestamp(new Date(2026, 0, 2, 3, 4, 5))).toBe('20260102030405');
  });
});

describe('buildAutoTitle', () => {
  const at = new Date(2026, 9, 14, 14, 39, 0);
  const base = { num_predict: 2048, seed: 512, think: false, showThinking: false };

  it('lists only the values that differ across providers, model first', () => {
    const providers = [
      { id: 'ollama:chat:qwen3-vl:4b', config: { ...base, temperature: 0.2 } },
      { id: 'ollama:chat:qwen3-vl:4b', config: { ...base, temperature: 0.8 } },
      { id: 'ollama:chat:qwen3.5:4b', config: { ...base, temperature: 0.2 } },
      { id: 'ollama:chat:qwen3.5:4b', config: { ...base, temperature: 0.8 } },
    ];
    expect(buildAutoTitle(providers, at)).toBe(
      '20261014143900 model qwen3-vl:4b qwen3.5:4b temp 0.2 0.8',
    );
  });

  it('shows booleans and think levels', () => {
    const providers = [
      { id: 'ollama:chat:m', config: { think: false } },
      { id: 'ollama:chat:m', config: { think: 'low' } },
      { id: 'ollama:chat:m', config: { think: true } },
    ];
    expect(buildAutoTitle(providers, at)).toBe('20261014143900 think false low true');
  });

  it('lists everything when nothing differs', () => {
    const providers = [{ id: 'ollama:chat:m', config: { temperature: 0.1, seed: 1 } }];
    expect(buildAutoTitle(providers, at)).toBe('20261014143900 model m seed 1 temp 0.1');
  });

  it('counts a key missing on some providers as differing', () => {
    const providers = [
      { id: 'openai:gpt-4o', config: { temperature: 0.5 } },
      { id: 'openai:gpt-4o' },
    ];
    expect(buildAutoTitle(providers, at)).toBe('20261014143900 temp 0.5');
  });

  it('returns just the timestamp when there are no providers', () => {
    expect(buildAutoTitle([], at)).toBe('20261014143900');
  });
});
