#!/usr/bin/env bash
# Sequential EMR runs (5 then 10 core nodes). Waits for each cluster to die,
# then copies metrics + logs into unique Desktop folders so nothing overwrites.
#
# Run from the project root, and keep the Mac awake:
#   caffeinate -i bash scripts/aws_overnight.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUCKET="${AWS_BUCKET:-cs6240-demo-bucket-kr1}"
JAR_NAME="yelp-stars.jar"
JOB="yelp.StarRating"
EMR_RELEASE="emr-6.10.0"
INSTANCE="${AWS_INSTANCE_TYPE:-m5.xlarge}"
INPUT_PREFIX="input_1m"
VOCAB=8192
TRAIN_FRAC=0.8
MAX_DEPTH=8
NUM_TREES=50
SEED=42

DEST="${DEST_DIR:-$HOME/Desktop/cs6240-yelp-aws}"
RUNLOG="$DEST/overnight_run.log"
mkdir -p "$DEST"

log() {
  # Must go to stderr so $(launch) only captures the cluster id.
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUNLOG" >&2
}

wait_terminated() {
  local id="$1"
  id="${id##*$'\n'}"
  id="$(echo "$id" | tr -d '\r' | awk '/^j-/{print; exit}')"
  local max_hours="${2:-4}"
  local deadline=$(( $(date +%s) + max_hours * 3600 ))
  while true; do
    local state
    state="$(aws emr describe-cluster --cluster-id "$id" --query 'Cluster.Status.State' --output text)"
    log "cluster $id state=$state"
    case "$state" in
      TERMINATED|TERMINATED_WITH_ERRORS)
        return 0
        ;;
    esac
    if (( $(date +%s) > deadline )); then
      log "ERROR: timed out waiting for $id (still $state)"
      return 1
    fi
    sleep 60
  done
}

launch() {
  local cores="$1"
  local out_prefix="$2"
  local log_prefix="$3"
  local name="Yelp-1m-${cores}core"

  log "deleting prior S3 output/log for this run only: $out_prefix $log_prefix"
  aws s3 rm "s3://$BUCKET/" --recursive --exclude "*" --include "${out_prefix}*" || true
  aws s3 rm "s3://$BUCKET/" --recursive --exclude "*" --include "${log_prefix}*" || true

  log "launching $name"
  local cid
  cid="$(aws emr create-cluster \
    --name "$name" \
    --release-label "$EMR_RELEASE" \
    --instance-groups "[{\"InstanceCount\":${cores},\"InstanceGroupType\":\"CORE\",\"InstanceType\":\"${INSTANCE}\"},{\"InstanceCount\":1,\"InstanceGroupType\":\"MASTER\",\"InstanceType\":\"${INSTANCE}\"}]" \
    --applications Name=Hadoop Name=Spark \
    --steps "Type=CUSTOM_JAR,Name=${name},Jar=command-runner.jar,ActionOnFailure=TERMINATE_CLUSTER,Args=[spark-submit,--deploy-mode,cluster,--driver-memory,4g,--executor-memory,4g,--class,${JOB},s3://${BUCKET}/${JAR_NAME},s3://${BUCKET}/${INPUT_PREFIX},s3://${BUCKET}/${out_prefix},${VOCAB},${TRAIN_FRAC},${MAX_DEPTH},${NUM_TREES},${SEED}]" \
    --log-uri "s3://${BUCKET}/${log_prefix}" \
    --configurations '[{"Classification":"hadoop-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}},{"Classification":"spark-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}}]' \
    --use-default-roles \
    --enable-debugging \
    --auto-terminate \
    --query 'ClusterId' \
    --output text)"
  echo "$cid" > "$DEST/${out_prefix}.cluster_id"
  log "started $name cluster_id=$cid"
  echo "$cid"
}

download_run() {
  local out_prefix="$1"
  local log_prefix="$2"
  local local_dir="$3"
  mkdir -p "$local_dir/metrics_and_output" "$local_dir/emr_logs"
  log "syncing s3://$BUCKET/$out_prefix -> $local_dir/metrics_and_output"
  aws s3 sync "s3://$BUCKET/$out_prefix" "$local_dir/metrics_and_output"
  log "syncing s3://$BUCKET/$log_prefix -> $local_dir/emr_logs"
  aws s3 sync "s3://$BUCKET/$log_prefix" "$local_dir/emr_logs"
  if [[ -f "$local_dir/metrics_and_output/metrics/part-00000" ]]; then
    cp "$local_dir/metrics_and_output/metrics/part-00000" "$local_dir/metrics.txt"
    log "copied metrics.txt"
  else
    log "WARNING: no metrics/part-00000 (step may have failed); check emr_logs"
  fi
}

log "==== overnight AWS start ===="
log "desktop dest=$DEST bucket=$BUCKET instance=$INSTANCE vocab=$VOCAB depth=$MAX_DEPTH trees=$NUM_TREES"

if [[ ! -f "$JAR_NAME" ]]; then
  log "building jar"
  make jar
fi
log "uploading jar"
aws s3 cp "$JAR_NAME" "s3://$BUCKET/$JAR_NAME"

if ! aws s3 ls "s3://$BUCKET/$INPUT_PREFIX/" | grep -q .; then
  log "uploading 1m reviews to s3://$BUCKET/$INPUT_PREFIX/"
  tmp="$(mktemp -d)"
  cp "$ROOT/input/reviews_1m.json" "$tmp/"
  aws s3 sync "$tmp" "s3://$BUCKET/$INPUT_PREFIX/"
  rm -rf "$tmp"
else
  log "S3 input already present: s3://$BUCKET/$INPUT_PREFIX/"
fi

# Sequential so only one cluster bills at a time.
# Resume 5-node wait: EXISTING_N5_CLUSTER=j-XXXX bash scripts/aws_overnight.sh
# Extra size only:    ONLY_CORES=7 bash scripts/aws_overnight.sh
# 10-node only:       ONLY_N10=1 bash scripts/aws_overnight.sh
if [[ -n "${ONLY_CORES:-}" ]]; then
  log "ONLY_CORES=$ONLY_CORES: skipping 5-node launch/wait/download"
  CIDX="$(launch "$ONLY_CORES" "output_1m_n${ONLY_CORES}" "log_1m_n${ONLY_CORES}")"
  wait_terminated "$CIDX" 4
  download_run "output_1m_n${ONLY_CORES}" "log_1m_n${ONLY_CORES}" "$DEST/1m_${ONLY_CORES}workers"
elif [[ -n "${ONLY_N10:-}" ]]; then
  log "ONLY_N10=1: skipping 5-node launch/wait/download"
  CID10="$(launch 10 output_1m_n10 log_1m_n10)"
  wait_terminated "$CID10" 4
  download_run output_1m_n10 log_1m_n10 "$DEST/1m_10workers"
else
  if [[ -n "${EXISTING_N5_CLUSTER:-}" ]]; then
    CID5="$EXISTING_N5_CLUSTER"
    echo "$CID5" > "$DEST/output_1m_n5.cluster_id"
    log "resuming wait on existing 5-node cluster $CID5 (will not launch another)"
  else
    CID5="$(launch 5 output_1m_n5 log_1m_n5)"
  fi
  wait_terminated "$CID5" 4
  download_run output_1m_n5 log_1m_n5 "$DEST/1m_5workers"

  CID10="$(launch 10 output_1m_n10 log_1m_n10)"
  wait_terminated "$CID10" 4
  download_run output_1m_n10 log_1m_n10 "$DEST/1m_10workers"
fi

log "==== overnight AWS done ===="
log "5-node:  $DEST/1m_5workers"
log "10-node: $DEST/1m_10workers"
log "this log: $RUNLOG"
