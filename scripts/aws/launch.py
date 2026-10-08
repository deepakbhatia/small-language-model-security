#!/usr/bin/env python3
"""Launch an EC2 GPU instance that runs SEI train (smoke/stage1/stage2) then shuts down.

Examples:
  # Smoke
  python scripts/aws/launch.py --mode smoke --wait ...

  # Continue stage-1 on next 50k GUIDE rows from existing adapter
  python scripts/aws/launch.py --mode stage1 --wait \\
    --guide-s3 s3://bucket/GUIDE_Train.csv \\
    --resume-adapter-s3 s3://bucket/sei-adapter/ \\
    --adapter-s3 s3://bucket/sei-adapter/ \\
    --guide-offset 20000 --guide-limit 50000 ...

  # Stage-2 from stage-1 checkpoint (techniques unlocked)
  python scripts/aws/launch.py --mode stage2 --wait \\
    --guide-s3 s3://bucket/GUIDE_Train.csv \\
    --resume-adapter-s3 s3://bucket/sei-adapter/ \\
    --adapter-s3 s3://bucket/sei-adapter-stage2/ \\
    --guide-limit 50000 ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def build_user_data(args: argparse.Namespace, entrypoint: str) -> str:
    exports = {
        "SEI_MODE": args.mode,
        "SEI_REPO_URL": args.repo_url,
        "SEI_S3_BUCKET": args.s3_bucket or "",
        "SEI_GUIDE_S3": args.guide_s3 or "",
        "SEI_ADAPTER_S3": args.adapter_s3 or "",
        "SEI_RESUME_ADAPTER_S3": args.resume_adapter_s3 or "",
        "SEI_MODEL_S3": args.model_s3 or "",
        "SEI_GUIDE_LIMIT": str(args.guide_limit),
        "SEI_GUIDE_OFFSET": str(args.guide_offset),
        "SEI_GUIDE_STAGE": str(args.guide_stage),
        "SEI_SHARD_SIZE": str(args.shard_size),
        "SEI_KEEP_ALIVE": "1" if args.keep_alive else "0",
        "HF_TOKEN": args.hf_token or "",
    }
    export_lines = "\n".join(f'export {k}="{v}"' for k, v in exports.items())
    return f"""#!/bin/bash
set -euo pipefail
{export_lines}
cat >/tmp/sei-entrypoint.sh <<'SEI_EOF'
{entrypoint}
SEI_EOF
chmod +x /tmp/sei-entrypoint.sh
# Give network/IAM a moment on fresh instances
sleep 5
/tmp/sei-entrypoint.sh
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="SEI AWS GPU launch (smoke|full|stage1|stage2)")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "stage1", "stage2"],
        default="smoke",
        help="full=legacy stage1; stage1=resume GUIDE continue; stage2=techniques from stage1 adapter",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print user-data / plan only")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="g5.xlarge")
    parser.add_argument("--ami-id", default="", help="Deep Learning AMI (PyTorch GPU) id")
    parser.add_argument("--subnet-id", default="")
    parser.add_argument("--security-group-ids", default="", help="Comma-separated sg ids")
    parser.add_argument("--iam-instance-profile", default="", help="Instance profile name")
    parser.add_argument("--key-name", default="", help="Optional SSH key")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--keep-alive", action="store_true", help="Do not auto-shutdown (debug)")
    parser.add_argument("--wait", action="store_true", help="Wait until instance stopped/terminated")
    parser.add_argument("--repo-url", default="https://github.com/deepakbhatia/small-language-model-security.git")
    parser.add_argument("--s3-bucket", default="", help="Bucket for logs / OK markers")
    parser.add_argument("--guide-s3", default="", help="s3://.../GUIDE_Train.csv")
    parser.add_argument("--adapter-s3", default="", help="s3://.../ write adapter prefix")
    parser.add_argument(
        "--resume-adapter-s3",
        default="",
        help="s3://.../ warm-start adapter (stage1 continue or stage2 from stage1)",
    )
    parser.add_argument("--model-s3", default="", help="Optional HF cache prefix on S3")
    parser.add_argument("--guide-limit", type=int, default=None, help="Max GUIDE rows after offset")
    parser.add_argument("--guide-offset", type=int, default=None, help="Skip first N GUIDE rows")
    parser.add_argument("--guide-stage", type=int, default=None, help="GUIDE curriculum stage (1 or 2)")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--hf-token", default="", help="Optional HF token")
    parser.add_argument("--volume-gb", type=int, default=150)
    args = parser.parse_args()

    # Mode-specific defaults: stage1 continues +50k GUIDE from row 20k;
    # stage2 trains techniques on up to 50k rows from the start at stage=2.
    if args.guide_limit is None:
        args.guide_limit = 50000 if args.mode in {"stage1", "stage2"} else 20000
    if args.guide_offset is None:
        args.guide_offset = 20000 if args.mode == "stage1" else 0
    if args.guide_stage is None:
        args.guide_stage = 2 if args.mode == "stage2" else 1

    if args.mode in {"full", "stage1", "stage2"} and not args.guide_s3 and not args.dry_run:
        parser.error("--guide-s3 is required for --mode full|stage1|stage2")
    if args.mode == "stage2" and not args.resume_adapter_s3 and not args.dry_run:
        parser.error("--resume-adapter-s3 (stage-1 adapter) is required for --mode stage2")
    if args.mode == "stage1" and not args.resume_adapter_s3 and not args.dry_run:
        print(
            "WARNING: --resume-adapter-s3 unset; stage1 will train a fresh adapter "
            "unless weights already exist on the instance.",
            file=sys.stderr,
        )

    entrypoint_path = Path(__file__).resolve().parent / "entrypoint.sh"
    entrypoint = entrypoint_path.read_text()
    user_data = build_user_data(args, entrypoint)

    print("=== plan ===")
    print(f"mode={args.mode} instance={args.instance_type} region={args.region} spot={args.spot}")
    print(f"repo={args.repo_url}")
    print(f"guide_s3={args.guide_s3 or '-'}")
    print(
        f"guide_stage={args.guide_stage} offset={args.guide_offset} limit={args.guide_limit} "
        f"shard_size={args.shard_size}"
    )
    print(f"resume_adapter_s3={args.resume_adapter_s3 or '-'}")
    print(f"adapter_s3={args.adapter_s3 or '-'}")
    print(f"keep_alive={args.keep_alive}")
    if args.dry_run:
        print("\n=== user-data (first 40 lines) ===")
        print("\n".join(user_data.splitlines()[:40]))
        print("... [truncated] ...")
        print("\nDry-run only — no instance launched.")
        return

    missing = [
        name
        for name, val in [
            ("--ami-id", args.ami_id),
            ("--subnet-id", args.subnet_id),
            ("--security-group-ids", args.security_group_ids),
            ("--iam-instance-profile", args.iam_instance_profile),
            ("--s3-bucket", args.s3_bucket),
        ]
        if not val
    ]
    if missing:
        parser.error("Missing required args for launch: " + ", ".join(missing))

    try:
        import boto3
    except ImportError as e:
        raise SystemExit("Install boto3: pip install boto3") from e

    ec2 = boto3.client("ec2", region_name=args.region)
    sg_ids = [s.strip() for s in args.security_group_ids.split(",") if s.strip()]

    block = [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": args.volume_gb,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }
    ]

    run_kwargs = {
        "ImageId": args.ami_id,
        "InstanceType": args.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": user_data,
        "SubnetId": args.subnet_id,
        "SecurityGroupIds": sg_ids,
        "IamInstanceProfile": {"Name": args.iam_instance_profile},
        "BlockDeviceMappings": block,
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"sei-{args.mode}"},
                    {"Key": "Project", "Value": "sei-slm"},
                    {"Key": "SEIMode", "Value": args.mode},
                ],
            }
        ],
        "InstanceInitiatedShutdownBehavior": "terminate" if not args.keep_alive else "stop",
    }
    if args.key_name:
        run_kwargs["KeyName"] = args.key_name
    if args.spot:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }

    print("Launching...")
    resp = ec2.run_instances(**run_kwargs)
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"instance_id={instance_id}")
    print(f"Console: aws ec2 get-console-output --instance-id {instance_id} --region {args.region}")
    print(f"Logs (after run): s3://{args.s3_bucket}/sei-logs/")

    if not args.wait:
        print("Not waiting (--wait not set). Remember to verify auto-shutdown.")
        return

    print("Waiting for running...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    print("Instance running. Waiting until stopped/terminated (training + shutdown)...")

    while True:
        desc = ec2.describe_instances(InstanceIds=[instance_id])
        state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"  state={state}")
        if state in {"stopped", "terminated", "shutting-down"}:
            break
        time.sleep(30)

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
    print(f"Done. final_state={state}")

    if args.s3_bucket:
        s3 = boto3.client("s3", region_name=args.region)
        markers = {
            "smoke": "sei-logs/SMOKE_OK",
            "full": "sei-logs/STAGE1_OK",
            "stage1": "sei-logs/STAGE1_OK",
            "stage2": "sei-logs/STAGE2_OK",
        }
        key = markers.get(args.mode)
        if key:
            try:
                s3.head_object(Bucket=args.s3_bucket, Key=key)
                print(f"{key} found in S3 — {args.mode} passed.")
            except Exception:
                print(
                    f"WARNING: {key} not in S3 yet. Check s3://{args.s3_bucket}/sei-logs/ "
                    "and console output.",
                    file=sys.stderr,
                )
                raise SystemExit(1)


if __name__ == "__main__":
    main()
