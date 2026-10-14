import { describe, expect, it } from 'vitest';
import { abbreviateConfigKey, formatConfigInline } from '../../src/util/configDisplay';

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
