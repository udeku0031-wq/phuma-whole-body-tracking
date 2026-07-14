import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch convert LAFAN1 csv motions to WandB motion artifacts.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing input csv files.")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for csv files.")
    parser.add_argument("--input_fps", type=int, default=30, help="Input motion fps.")
    parser.add_argument("--output_fps", type=int, default=50, help="Output motion fps.")
    parser.add_argument("--progress_interval", type=int, default=1000, help="Print conversion progress every N frames.")
    parser.add_argument("--start", type=int, default=0, help="Start index after sorting matched files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to convert.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--device", type=str, default=None, help="Isaac Lab device argument, e.g. cuda:0 or cpu.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for each motion if Isaac Sim crashes.")
    parser.add_argument("--retry_delay", type=float, default=15.0, help="Seconds to wait before retrying a failed motion.")
    parser.add_argument("--sleep_between", type=float, default=5.0, help="Seconds to wait between successful motions.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue with the next motion after retries fail.")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if args.limit is not None:
        files = files[args.start : args.start + args.limit]
    else:
        files = files[args.start :]

    if not files:
        raise SystemExit(f"No files matched {args.input_dir / args.pattern}")

    script = Path(__file__).with_name("csv_to_npz.py")
    failed = []
    for index, csv_file in enumerate(files, start=1):
        output_name = csv_file.stem
        cmd = [
            sys.executable,
            str(script),
            "--input_file",
            str(csv_file),
            "--input_fps",
            str(args.input_fps),
            "--output_fps",
            str(args.output_fps),
            "--output_name",
            output_name,
            "--progress_interval",
            str(args.progress_interval),
            "--exit_after_save",
        ]
        if args.headless:
            cmd.append("--headless")
        if args.device is not None:
            cmd.extend(["--device", args.device])

        print(f"[{index}/{len(files)}] converting {csv_file} -> {output_name}", flush=True)
        for attempt in range(args.retries + 1):
            try:
                subprocess.run(cmd, check=True)
                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)
                break
            except subprocess.CalledProcessError as exc:
                if attempt >= args.retries:
                    failed.append(str(csv_file))
                    print(f"[ERROR] failed after {attempt + 1} attempt(s): {csv_file}", flush=True)
                    if not args.continue_on_error:
                        raise
                    break
                print(
                    f"[WARN] conversion failed for {csv_file} with exit code {exc.returncode}; "
                    f"retrying in {args.retry_delay:g}s ({attempt + 1}/{args.retries})",
                    flush=True,
                )
                time.sleep(args.retry_delay)

    if failed:
        print("[WARN] Failed motions:", flush=True)
        for csv_file in failed:
            print(f"  {csv_file}", flush=True)


if __name__ == "__main__":
    main()
