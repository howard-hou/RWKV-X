import os
import json
import subprocess
from tqdm import tqdm
from datasets import load_dataset

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_and_gzip(input_path, subset, lines_per_chunk, output_dir):
    ensure_dir(output_dir)
    chunk_id = 0
    chunk = []
    gzip_procs = []
    if subset:
        infile = load_dataset(input_path, subset, split="train", num_proc=4)
    else:
        infile = load_dataset(input_path, split="train", num_proc=4)

    with tqdm(desc="📝 Splitting + Gzipping (approx)", unit="chunk") as pbar:
        for line in infile:
            chunk.append(json.dumps(line, ensure_ascii=False) + "\n")

            if len(chunk) >= lines_per_chunk:
                chunk_path = os.path.join(output_dir, f"chunk_{chunk_id:05d}.jsonl")
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.writelines(chunk)
                proc = subprocess.Popen(["gzip", chunk_path])
                gzip_procs.append(proc)

                chunk_id += 1
                chunk = []
                pbar.update(1)

        # Write remaining lines
        if chunk:
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_id:05d}.jsonl")
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.writelines(chunk)
            proc = subprocess.Popen(["gzip", chunk_path])
            gzip_procs.append(proc)
            pbar.update(1)

    print(f"⌛ Waiting for {len(gzip_procs)} gzip processes to finish...")
    for proc in gzip_procs:
        proc.wait()

    print(f"✅ Done. All .jsonl.gz files saved to: {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast split + gzip .jsonl file using subprocess")
    parser.add_argument("input", help="Path to Huggingface Dataset, local or remote")
    parser.add_argument("--subset", default="", help="Subset of the dataset to process")
    parser.add_argument("--lines", type=int, default=1_000_000, help="Lines per chunk")
    parser.add_argument("--output_dir", default="split_chunks_gz", help="Directory to store .jsonl.gz files")
    args = parser.parse_args()

    print(f"📂 Input: {args.input}")
    if args.subset:
        print(f"📂 Subset: {args.subset}")
    else:
        print("📂 No subset specified, processing the entire dataset.")
    print(f"📦 Output dir: {args.output_dir}")
    print(f"📏 Chunk size: {args.lines} lines")

    split_and_gzip(args.input, args.subset, args.lines, args.output_dir)

if __name__ == "__main__":
    main()

