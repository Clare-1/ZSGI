import json, statistics

with open('validation_50_th05.json', 'r', encoding='utf-8') as f:
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

correct = [r for r in results if r['correct']]
wrong = [r for r in results if not r['correct']]
wrong_nonrep = [r for r in wrong if not has_repetition(r['output_text'])]

print('Total: %d, Correct: %d, Wrong: %d (non-rep: %d, rep: %d)' % (
    len(results), len(correct), len(wrong), len(wrong_nonrep), len(wrong) - len(wrong_nonrep)))

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
print('=== Step-level Signal Comparison ===')
print('Signal              Correct    Wrong(non-rep)  Ratio(W/C)')
print('-' * 60)
for key in ['top1_flip', 'top1_prob_change', 'topk_jaccard', 'kl_div', 'confidence_gap_change', 'entropy_change', 'entropy', 'instability']:
    c = mean_signal(correct, key)
    w = mean_signal(wrong_nonrep, key)
    ratio = w / c if c > 0 else 0
    print('%-20s %.4f     %.4f          %.2fx' % (key, c, w, ratio))

# Last step analysis
print()
print('=== Last Step Instability ===')
c_last = [r['step_signals'][-1]['instability'] for r in correct if r['step_signals']]
w_last = [r['step_signals'][-1]['instability'] for r in wrong_nonrep if r['step_signals']]
print('Correct last step: mean=%.3f (n=%d)' % (statistics.mean(c_last) if c_last else 0, len(c_last)))
print('Wrong last step:   mean=%.3f (n=%d)' % (statistics.mean(w_last) if w_last else 0, len(w_last)))

# Max instability position
def max_inst_info(r):
    instabs = [s['instability'] for s in r['step_signals']]
    if not instabs:
        return 0, 0
    max_idx = instabs.index(max(instabs))
    return max_idx / len(instabs), max(instabs)

print()
print('=== Max Instability Position (ratio in step sequence) ===')
c_pos = [max_inst_info(r)[0] for r in correct]
w_pos = [max_inst_info(r)[0] for r in wrong_nonrep]
print('Correct: mean=%.2f' % (statistics.mean(c_pos) if c_pos else 0))
print('Wrong(non-rep): mean=%.2f' % (statistics.mean(w_pos) if w_pos else 0))

# Instability spike detection: does instability spike at the end?
print()
print('=== Instability Spike at End (last step > mean of all steps) ===')
c_spike = sum(1 for r in correct if r['step_signals'] and r['step_signals'][-1]['instability'] > statistics.mean([s['instability'] for s in r['step_signals']]))
w_spike = sum(1 for r in wrong_nonrep if r['step_signals'] and r['step_signals'][-1]['instability'] > statistics.mean([s['instability'] for s in r['step_signals']]))
print('Correct with end spike: %d/%d (%.0f%%)' % (c_spike, len(correct), 100*c_spike/len(correct) if correct else 0))
print('Wrong with end spike:   %d/%d (%.0f%%)' % (w_spike, len(wrong_nonrep), 100*w_spike/len(wrong_nonrep) if wrong_nonrep else 0))
