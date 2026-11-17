"""Regression checks for signal extraction, preprocessing parity and streaming recovery."""
import unittest

import numpy as np
from fastapi.testclient import TestClient

from robot_interaction.audio import features
from robot_interaction.data import LABELS, ROOT, load_rows, media_index
from robot_interaction.engine import Engine
from robot_interaction.server import app
from robot_interaction.train import read_split, fuse, parameter_count


class AudioFeaturesTests(unittest.TestCase):
    def test_silence_and_short_audio_are_finite(self):
        self.assertEqual(features(np.zeros(160)).shape, (42,))
        self.assertTrue(np.isfinite(features(np.zeros(16000))).all())

    def test_invalid_audio_rejected(self):
        for audio in (np.zeros(10), np.array([np.nan]*400), np.zeros((400,2))):
            with self.assertRaises(ValueError):
                features(audio)


@unittest.skipUnless((ROOT/'artifacts/model.joblib').exists(), 'Train or fetch the supplied model first')
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.index = media_index()['dev']
        cls.rows = load_rows('dev')
        cls.row = next(row for row in cls.rows.itertuples() if row.clip_id in cls.index)

    def test_missing_audio_never_silently_downgrades(self):
        with self.assertRaises(ValueError):
            self.engine.predict('Hello', mode='fusion')
        with self.assertRaises(ValueError):
            self.engine.predict(' ', mode='text')

    def test_online_and_cached_preprocessing_agree(self):
        rows, acoustic = read_split('dev')
        bundle = self.engine.bundle
        for i in range(5):
            row = rows.iloc[i]
            words = bundle['vectorizer'].transform([row.Utterance])
            signal = bundle['scaler'].transform(acoustic[i:i+1])
            expected = bundle['fusion'].predict_proba(fuse(words,signal,bundle['audio_weight']))[0]
            result = self.engine.predict(row.Utterance,self.index[row.clip_id])
            actual = np.array([result['scores'][str(c)] for c in bundle['fusion'].classes_])
            np.testing.assert_allclose(actual,expected,atol=0.00006)
            self.assertIn(result['emotion'],LABELS)
            self.assertIn(' '.join(row.Utterance.split())[:90],result['response'])

    def test_parameter_budget(self):
        self.assertLess(parameter_count(self.engine.bundle)['total_required_for_demo'],6_000_000_000)

    def test_websocket_error_recovery_and_order(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/turns?split=invalid').status_code,400)
            self.assertEqual(client.get('/media/dev/unknown').status_code,404)
            with client.websocket_connect('/ws') as socket:
                socket.send_json({'split':'invalid','clip_id':self.row.clip_id})
                self.assertEqual(socket.receive_json()['type'],'error')
                for mode in ('text','fusion'):
                    socket.send_json({'split':'dev','clip_id':self.row.clip_id,'mode':mode})
                    state = socket.receive_json()
                    self.assertEqual(state['type'],'state')
                    self.assertEqual(state['turn_id'],self.row.clip_id)
                    chunks=[]
                    while True:
                        event=socket.receive_json()
                        if event['type']=='done':
                            self.assertEqual(''.join(chunks).strip(),event['response'])
                            break
                        self.assertEqual(event['type'],'response_chunk')
                        chunks.append(event['text'])


if __name__=='__main__':
    unittest.main()
