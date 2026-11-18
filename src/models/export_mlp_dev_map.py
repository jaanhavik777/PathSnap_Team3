"""Export a student-test development map using the saved, real MLP."""
import argparse
from pathlib import Path
import joblib
from src.data.person4_candidates import read_trip,candidate_features
from src.data.person4_maps import build_map

def main():
 p=argparse.ArgumentParser();p.add_argument('--trip-folder',required=True,help='One permitted train/student_test trip folder');p.add_argument('--model',default='artifacts/models/mlp/mlp_pipeline.joblib');p.add_argument('--output',default='artifacts/maps/development_map.html');p.add_argument('--include-reference',action='store_true',help='Permitted only for train/student_test report figures')
 a=p.parse_args(); folder=Path(a.trip_folder)
 if folder.parent.name.lower() not in {'train','student_test'}:raise SystemExit('Refusing reference map outside train/student_test partition')
 trip=read_trip(folder,include_route=a.include_reference);bundle=joblib.load(a.model);features=candidate_features(trip,candidate_count=bundle.get('candidate_count',4))
 features['_score']=bundle['model'].predict_proba(features[bundle['feature_columns']])[:,1]
 pred=[]
 for _,group in features.groupby('observation_id',sort=True):pred.append(int(group.loc[group._score.idxmax(),'candidate_arc_id']))
 m=build_map(trip,predicted_arc_ids=pred,reference_arc_ids=trip.route if a.include_reference else (),include_reference=a.include_reference)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);m.save(str(out));print(f'Wrote {out} for {trip.trip_id}; {len(trip.gps)} GPS points and {len(pred)} predicted candidate arcs')
if __name__=='__main__':main()
