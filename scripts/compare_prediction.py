import sys
import deepchopper

def compare(p1,p2):
    assert len(p1) == len(p2)
    for key, value in p1.items():
        if p2[key].prediction != value.prediction:
            print(f"{key} has diff prediction")
    print("all same")

def main():
    p1 = sys.argv[1]
    p2 = sys.argv[2]

    p1p = deepchopper.load_predicts_from_batch_pts(p1)
    p2p = deepchopper.load_predicts_from_batch_pts(p2)
    compare(p1p, p2p)

if __name__ == "__main__":
    main()
