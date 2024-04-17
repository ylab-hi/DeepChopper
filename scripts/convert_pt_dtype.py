import sys
from rich.progress import track
from pathlib import Path
from rich import print
import torch


def main():
    pt_path = Path(sys.argv[1])
    all_files = list(pt_path.rglob("*.pt"))

    for file in track(all_files, description=f"Converting {len(all_files)} files..."):
        try:
            data = torch.load(file)
            data['id'] = data['id'].to(torch.int64)
            data['target'] = data['target'].to(torch.int64)
        except Exception as e:
            print(f"[red]Error in {file} {e}[/red]")
        else:
            torch.save(data, file)



if __name__ == "__main__":
    main()
