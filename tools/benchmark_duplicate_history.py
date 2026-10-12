"""Measure exact duplicate checks after a long cloud/noise result history."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_processing import _SavedEventIndex, _candidate_already_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=20000)
    parser.add_argument('--queries', type=int, default=200)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.events < 1 or args.queries < 1:
        parser.error('Events and queries must be positive')
    indexed = _SavedEventIndex(25)
    for index in range(args.events):
        indexed.append(dict(center=(50, 60), label='not_meteor',
                            detection_start_frame=index * 25,
                            detection_end_frame=index * 25 + 2))
    saved = list(indexed)
    bounds = [((args.events - 1 - i % min(args.events, 100)) * 25,
               (args.events - 1 - i % min(args.events, 100)) * 25 + 2)
              for i in range(args.queries)]
    results, timings = [], []
    for history in (saved, indexed):
        started = time.perf_counter()
        results.append([_candidate_already_saved(((600, 550), (600, 650)), window,
                                                history, 25) for window in bounds])
        timings.append(time.perf_counter() - started)
    result = dict(events=args.events, queries=args.queries, baseline_seconds=timings[0],
                  indexed_seconds=timings[1], exact_decisions=results[0] == results[1])
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result['exact_decisions'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
