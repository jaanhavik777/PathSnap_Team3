import argparse
from src.data.visualize import make_trip_map
p=argparse.ArgumentParser();p.add_argument('trip_folder');p.add_argument('--out',default='artifacts/maps/example.html');a=p.parse_args()
make_trip_map(a.trip_folder,a.out);print('Saved',a.out)
