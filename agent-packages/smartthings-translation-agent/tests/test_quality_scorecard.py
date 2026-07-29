import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from quality_scorecard import evaluate
class ScoreTests(unittest.TestCase):
 def test_metrics(self):
  g=[{"id":"a","before":"A","expected_after":"B"},{"id":"b","before":"C","expected_after":"C"}]
  r=[{"id":"a","before":"A","proposed_after":"B"},{"id":"b","before":"C","proposed_after":"D"}]
  x=evaluate(g,r);self.assertEqual(x["exact_matches"],1);self.assertEqual(x["false_positive_changes"],1)
 def test_id_mismatch(self):
  with self.assertRaises(ValueError):evaluate([{"id":"a"}],[])
if __name__=="__main__":unittest.main()
