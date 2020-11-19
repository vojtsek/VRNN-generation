import json
import sys

data = []
with open(sys.argv[1], 'rt') as in_fd:
    dial = []
    last = None
    for line in in_fd:
        line = line.strip()
        for utterance in line.split('__eou__'):
            if last is None:
                last = utterance
            else:
                dial.append({'usr': last, 'system': utterance})
                last = None
    data.append(dial)

with open(sys.argv[2], 'wt') as out_fd:
    json.dump(data, out_fd)
