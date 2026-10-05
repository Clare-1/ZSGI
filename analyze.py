import json, statistics

with open('validation_30.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

def has_repetition(text, threshold=5):
    if len(text) < 100:
        return False
    tail = text[-80:]
    if len(tail) < 25:
        return False
    snippet = tail[:25]
    count = text.count(snippet)
    return count >= threshold

def max_instability_position(r):
    instabs = [s['instability'] for s in r['step_signals']]
    if not instabs:
        return 0, 0
    max_idx = instabs.index(max(instabs))
    return max_idx / len(instabs), max(instabs)

correct = [r for r in results if r['correct']]
wrong = [r for r in results if not r['correct']]
wrong_nonrep = [r for r in wrong if not has_repetition(r['output_text'])]
wrong_rep = [r for r in wrong if has_repetition(r['output_text'])]

print('Correct: %d' % len(correct))
print('Wrong (no repetition): %d' % len(wrong_nonrep))
print('Wrong (repetition): %d' % len(wrong_rep))

def mean_signal(samples, key):
    vals = []
    for r in samples:
        for s in r['step_signals']:
            if key == 'instability':
                vals.append(s['instability'])
            else:
                vals.append(s['signals'].get(key, 0))
    return statistics.mean(vals) if vals else 0

print()
print('=== Signal Comparison (excluding repetition samples) ===')
print('Signal              Correct    Wrong(non-rep)  Ratio')
print('-' * 60)
for key in ['top1_flip', 'kl_div', 'entropy', 'instability']:
    c = mean_signal(correct, key)
    w = mean_signal(wrong_nonrep, key)
    ratio = w / c if c > 0 else 0
    print('%-20s %.4f     %.4f          %.2fx' % (key, c, w, ratio))

print()
print('=== Max Instability Position Analysis ===')
print('Correct max instability pos ratio:', [round(max_instability_position(r)[0], 2) for r in correct])
print('Wrong(non-rep) max instability pos ratio:', [round(max_instability_position(r)[0], 2) for r in wrong_nonrep[:10]])
print()
print('Correct max instability values:', [round(max_instability_position(r)[1], 2) for r in correct])
print('Wrong(non-rep) max instability values:', [round(max_instability_position(r)[1], 2) for r in wrong_nonrep[:10]])

# Last step instability comparison
print()
print('=== Last Step Instability ===')
c_last = [r['step_signals'][-1]['instability'] for r in correct if r['step_signals']]
w_last = [r['step_signals'][-1]['instability'] for r in wrong_nonrep if r['step_signals']]
print('Correct last step instability: mean=%.3f' % (statistics.mean(c_last) if c_last else 0))
print('Wrong(non-rep) last step instability: mean=%.3f' % (statistics.mean(w_last) if w_last else 0))
