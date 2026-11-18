import argparse
from pathlib import Path
from src.data.inventory import inventory
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='data/raw');p.add_argument('--out',default='artifacts/results/dataset_inventory.csv');a=p.parse_args()
    df=inventory(a.data_root);Path(a.out).parent.mkdir(parents=True,exist_ok=True);df.to_csv(a.out,index=False)
    print(df.groupby('partition')[['gps_points','nodes','arcs','route_arcs']].agg(['count','sum','median']).to_string())
    print('Saved',a.out)
if __name__=='__main__':main()
