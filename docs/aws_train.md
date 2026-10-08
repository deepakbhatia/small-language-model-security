# AWS GPU training (smoke → full)

Programmatic EC2 flow: **launch → train → sync adapter to S3 → auto-shutdown**.

## Prerequisites

1. AWS CLI / credentials with EC2 + S3 permissions  
2. S3 bucket (e.g. `my-sei-bucket`)  
3. IAM **instance profile** that can read/write that bucket  
4. VPC subnet + security group (egress HTTPS for HF/git/S3)  
5. Deep Learning AMI (PyTorch GPU), e.g. search EC2 console for “Deep Learning OSS Nvidia Driver AMI GPU PyTorch”

Upload GUIDE once:

```bash
aws s3 cp data/raw/guide/GUIDE_Train.csv s3://my-sei-bucket/GUIDE_Train.csv
```

Optional (speeds every run): cache the base model on S3 after first download.

## Quick iteration (smoke)

Does **not** need GUIDE. Uses seed JSONL, **2 train steps**, writes `s3://bucket/sei-logs/SMOKE_OK`, then **terminates**.

```bash
# No spend — inspect user-data only
python scripts/aws/launch.py --mode smoke --dry-run --s3-bucket my-sei-bucket

# Real smoke (~$1–3 depending on AMI pull + HF download)
pip install boto3
python scripts/aws/launch.py --mode smoke --wait --spot \
  --region us-east-1 \
  --instance-type g5.xlarge \
  --ami-id ami-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --security-group-ids sg-XXXXXXXX \
  --iam-instance-profile sei-train-profile \
  --s3-bucket my-sei-bucket \
  --adapter-s3 s3://my-sei-bucket/sei-adapter/
```

Check:

```bash
aws s3 ls s3://my-sei-bucket/sei-logs/
aws s3 ls s3://my-sei-bucket/sei-adapter/
```

If `SMOKE_OK` is present and the instance is terminated, infra + train path works.

Debug without shutdown:

```bash
python scripts/aws/launch.py --mode smoke --keep-alive ...
# SSH in, read /var/log/sei-train.log, then stop/terminate manually
```

## Full training (after smoke)

```bash
# Initial stage-1 (first 20k GUIDE) — already done if sei-adapter/ exists
python scripts/aws/launch.py --mode full --wait \
  --region us-east-1 \
  --instance-type g5.xlarge \
  --ami-id ami-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --security-group-ids sg-XXXXXXXX \
  --iam-instance-profile sei-train-profile \
  --s3-bucket my-sei-bucket \
  --guide-s3 s3://my-sei-bucket/GUIDE_Train.csv \
  --adapter-s3 s3://my-sei-bucket/sei-adapter/ \
  --guide-limit 20000 \
  --shard-size 2000
```

## Stage-1 continue (+50k GUIDE from row 20k, resume adapter)

Defaults: `--guide-offset 20000 --guide-limit 50000`.

```bash
python scripts/aws/launch.py --mode stage1 --wait \
  --region us-east-1 \
  --instance-type g5.xlarge \
  --ami-id ami-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --security-group-ids sg-XXXXXXXX \
  --iam-instance-profile sei-train-profile \
  --s3-bucket my-sei-bucket \
  --guide-s3 s3://my-sei-bucket/GUIDE_Train.csv \
  --resume-adapter-s3 s3://my-sei-bucket/sei-adapter/ \
  --adapter-s3 s3://my-sei-bucket/sei-adapter/
```

Success marker: `s3://…/sei-logs/STAGE1_OK`

## Stage-2 (techniques; warm-start from stage-1)

```bash
python scripts/aws/launch.py --mode stage2 --wait \
  --region us-east-1 \
  --instance-type g5.xlarge \
  --ami-id ami-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --security-group-ids sg-XXXXXXXX \
  --iam-instance-profile sei-train-profile \
  --s3-bucket my-sei-bucket \
  --guide-s3 s3://my-sei-bucket/GUIDE_Train.csv \
  --resume-adapter-s3 s3://my-sei-bucket/sei-adapter/ \
  --adapter-s3 s3://my-sei-bucket/sei-adapter-stage2/ \
  --guide-limit 50000
```

Success marker: `s3://…/sei-logs/STAGE2_OK`. Pull with:
`aws s3 sync s3://my-sei-bucket/sei-adapter-stage2/ checkpoints/sei-sft-stage2/adapter/`

**Push `main` before launching** — the instance clones the repo for `compose_all` / `train_shards` (user-data only embeds `entrypoint.sh`).

## Local smoke (no AWS)

```bash
export PYTHONPATH=src:.
python scripts/compose_synthetic.py --out data/processed/seed
mkdir -p data/processed/v1/shards
head -n 8 data/processed/seed/train.jsonl > data/processed/v1/shards/train_shard00.jsonl
cp data/processed/seed/val.jsonl data/processed/v1/val.jsonl
python scripts/aws/train_shards.py \
  --shards-dir data/processed/v1/shards \
  --val-file data/processed/v1/val.jsonl \
  --max-shards 1 --max-steps 2 --max-length 256 \
  --batch-size 1 --grad-accum 1 --eval-strategy no
```

## Safety

- Prefer **Spot** `g5.xlarge`  
- Set a **Budget alert**  
- Default shutdown behavior terminates the instance when the job finishes  
- Do not pass long-lived secrets in user-data for production (use SSM/Secrets Manager)
