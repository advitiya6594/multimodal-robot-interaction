"""Publish deterministic post-hoc examples of improvement and regression."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_interaction.data import ROOT, load_rows


def main():
    records = json.loads((ROOT/'reports/test_predictions.json').read_text())
    rows = load_rows('test').set_index('clip_id')
    filters = {
        'audio_helps': lambda r: r['fusion'] == r['label'] and r['text'] != r['label'],
        'audio_hurts': lambda r: r['text'] == r['label'] and r['fusion'] != r['label'],
        'both_wrong': lambda r: r['text'] != r['label'] and r['fusion'] != r['label'],
    }
    result = {'selection': 'Post-hoc illustrative cases. First matching test turn in original CSV order per category. Selected after final scoring. No model tuning from these cases.', 'cases': {}}
    for category, predicate in filters.items():
        match = next((r for r in records if predicate(r)), None)
        if match:
            match = {**match, 'transcript': str(rows.loc[match['clip_id'],'Utterance']),
                     'replay': f"python -m robot_interaction.demo --split test --dialogue {match['dialogue_id']}"}
        result['cases'][category] = match
    (ROOT/'reports/case_studies.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
