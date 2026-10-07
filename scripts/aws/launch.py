#!/usr/bin/env python3
"""Launch an EC2 GPU instance that runs SEI train (smoke or full) then shuts down.

Examples:
  # 1) Print user-data only (no spend)
  python scripts/aws/launch.py --mode smoke --dry-run \\
    --s3-bucket my-sei-bucket

  # 2) Smoke: start → 2 train steps → sync → auto-stop
  python scripts/aws/launch.py --mode smoke \\
    --ami-id ami-xxxxxxxx \\
    --subnet-id subnet-xxxxxxxx \\
    --security-group-ids sg-xxxxxxxx \\
    --iam-instance-profile sei-train-profile \\
    --s3-bucket my-sei-bucket \\
    --adapter-s3 s3://my-sei-bucket/sei-adapter/ \\
    --wait

  # 3) Full training after smoke works
  python scripts/aws/launch.py --mode full \\
    --ami-id ami-xxxxxxxx \\
    --subnet-id subnet-xxxxxxxx \\
    --security-group-ids sg-xxxxxxxx \\
    --iam-instance-profile sei-train-profile \\
    --s3-bucket my-sei-bucket \\
    --guide-s3 s3://my-sei-bucket/GUIDE_Train.csv \\
    --adapter-s3 s3://my-sei-bucket/sei-adapter/ \\
    --guide-limit 20000 \\
    --spot --wait
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path


def build_user_data(args: argparse.Namespace, entrypoint: str) -> str:
    # Embed entrypoint and export env for the remote bash script.
    exports = {
        "SEI_MODE": args.mode,
        "SEI_REPO_URL": args.repo_url,
        "SEI_S3_BUCKET": args.s3_bucket or "",
        "SEI_GUIDE_S3": args.guide_s3 or "",
        "SEI_ADAPTER_S3": args.adapter_s3 or "",
        "SEI_MODEL_S3": args.model_s3 or "",
        "SEI_GUIDE_LIMIT": str(args.guide_limit),
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
    parser = argparse.ArgumentParser(description="SEI AWS GPU launch (smoke|full)")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
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
    parser.add_argument("--s3-bucket", default="", help="Bucket for logs / SMOKE_OK")
    parser.add_argument("--guide-s3", default="", help="s3://.../GUIDE_Train.csv (full mode)")
    parser.add_argument("--adapter-s3", default="", help="s3://.../sei-adapter/")
    parser.add_argument("--model-s3", default="", help="Optional HF cache prefix on S3")
    parser.add_argument("--guide-limit", type=int, default=20000)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--hf-token", default="", help="Optional HF token (prefer Secrets Manager later)")
    parser.add_argument("--volume-gb", type=int, default=150)
    args = parser.parse_args()

    if args.mode == "full" and not args.guide_s3 and not args.dry_run:
        parser.error("--guide-s3 is required for --mode full")

    entrypoint_path = Path(__file__).resolve().parent / "entrypoint.sh"
    entrypoint = entrypoint_path.read_text()
    user_data = build_user_data(args, entrypoint)

    print("=== plan ===")
    print(f"mode={args.mode} instance={args.instance_type} region={args.region} spot={args.spot}")
    print(f"repo={args.repo_url}")
    print(f"guide_s3={args.guide_s3 or '-'}")
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

    # Poll until not running/pending
    while True:
        desc = ec2.describe_instances(InstanceIds=[instance_id])
        state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"  state={state}")
        if state in {"stopped", "terminated", "shutting-down"}:
            break
        time.sleep(30)

    # Final state
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
    print(f"Done. final_state={state}")
    if args.mode == "smoke" and args.s3_bucket:
        s3 = boto3.client("s3", region_name=args.region)
        try:
            s3.head_object(Bucket=args.s3_bucket, Key="sei-logs/SMOKE_OK")
            print("SMOKE_OK found in S3 — smoke passed.")
        except Exception:
            print(
                "WARNING: SMOKE_OK not in S3 yet. Check s3://{}/sei-logs/ and console output.".format(
                    args.s3_bucket
                ),
                file=sys.stderr,
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
