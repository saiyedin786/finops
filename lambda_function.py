# lambda_function.py
import boto3, json, os, time, uuid, logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from botocore.config import Config

# -------- Settings --------
REGION = os.environ.get("AWS_REGION", "us-east-2")
MODEL_ID = os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-3-haiku-20240307-v1:0")

TABLE_NAME = os.environ["TABLE"]  # fail early if missing

# Optional knobs
CPU_LOOKBACK_MIN = int(os.environ.get("CPU_LOOKBACK_MIN", "60"))      # last 60 mins
CPU_PERIOD_SEC   = int(os.environ.get("CPU_PERIOD_SEC", "300"))       # 5 min bins
COST_LOOKBACK_DAYS = int(os.environ.get("COST_LOOKBACK_DAYS", "7"))
REQUIRE_API_KEY = None
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "600"))

# boto config: small retries
boto_cfg = Config(retries={"max_attempts": 4, "mode": "standard"})

log = logging.getLogger()
log.setLevel(logging.INFO)

cloudwatch = boto3.client("cloudwatch", region_name=REGION, config=boto_cfg)
ec2 = boto3.client("ec2", region_name=REGION, config=boto_cfg)
ddb = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
ce = boto3.client("ce", config=boto_cfg)
bedrock = boto3.client("bedrock-runtime", region_name=REGION, config=boto_cfg)

def _utcnow():
    return datetime.now(timezone.utc)

def get_cost_summary():
    end = _utcnow().strftime("%Y-%m-%d")
    start = (_utcnow() - timedelta(days=COST_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    token = None
    lines = []
    while True:
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            NextPageToken=token
        ) if token else ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}]
        )
        for day in resp.get("ResultsByTime", []):
            for g in day.get("Groups", []):
                service = g["Keys"][0]
                amount = g["Metrics"]["UnblendedCost"]["Amount"]
                try:
                    lines.append(f"{service}: ${float(amount):.2f}")
                except Exception:
                    # handle "-" or missing values
                    lines.append(f"{service}: ${amount}")
        token = resp.get("NextPageToken")
        if not token:
            break

    if not lines:
        return "No cost data returned for the selected window."
    # Keep it terse: last day plus a few others can be overwhelming;
    # we’ll just join everything; the LLM can handle summarizing.
    return "\n".join(lines)

def to_ddb(value):
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: to_ddb(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_ddb(v) for v in value]
    return value

def get_cpu_average(instance_id):
    end = _utcnow()
    start = end - timedelta(minutes=CPU_LOOKBACK_MIN)
    stats = cloudwatch.get_metric_statistics(
        Namespace="AWS/EC2",
        MetricName="CPUUtilization",
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        StartTime=start,
        EndTime=end,
        Period=CPU_PERIOD_SEC,
        Statistics=["Average"]
    )
    dps = stats.get("Datapoints", [])
    if not dps:
        return 0.0
    return round(sum(d["Average"] for d in dps) / len(dps), 2)

def summarize_with_llm(context_text):
    prompt = (
        "You are an expert AWS DevOps & FinOps assistant. "
        "From the cost and utilization context, produce a short executive summary, "
        "then 3-6 prioritized, low-risk savings or hygiene actions. "
        "Call out anomalies, specify instance IDs where relevant, and note verification steps.\n\n"
        f"{context_text}"
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }
    resp = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    result = json.loads(resp["body"].read())
    return result.get("content", [{}])[0].get("text", json.dumps(result))

def _paginate_instances():
    token = None
    while True:
        kwargs = {"NextToken": token} if token else {}
        page = ec2.describe_instances(**kwargs)
        for r in page.get("Reservations", []):
            for i in r.get("Instances", []):
                yield i
        token = page.get("NextToken")
        if not token:
            break

def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Api-Key"
    }

def lambda_handler(event, context):
    try:
        if event.get("httpMethod") == "OPTIONS":
            return {"statusCode": 200, "headers": _cors_headers(), "body": ""}

        # Optional shared-secret check for Function URL
        if REQUIRE_API_KEY:
            hdr = (event.get("headers") or {}).get("x-api-key") or (event.get("headers") or {}).get("X-Api-Key")
            if hdr != REQUIRE_API_KEY:
                return {
                    "statusCode": 401,
                    "headers": _cors_headers(),
                    "body": json.dumps({"error": "unauthorized"})
                }

        raw_body = event.get("body", "") or ""
        body = json.loads(raw_body) if raw_body and raw_body.strip().startswith("{") else {}
        question = body.get("question", "Analyze EC2 utilization trends for the last hour.")
        if not isinstance(question, str):
            question = str(question)
        if len(question) > 2000:
            question = question[:2000]

        log.info(json.dumps({"msg": "request_received", "len_question": len(question)}))

        # Build utilization report
        report_lines = []
        for inst in _paginate_instances():
            iid = inst["InstanceId"]
            name = next((t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), iid)
            state = inst["State"]["Name"]
            cpu = get_cpu_average(iid)
            report_lines.append(f"{name} ({iid}) - state={state}, avgCPU={cpu}%")

        context_text = get_cost_summary() + "\n\n" + "\n".join(report_lines) + f"\n\nQuestion: {question}"

        start = time.time()
        answer = summarize_with_llm(context_text)
        latency = int((time.time() - start) * 1000)

        item = to_ddb({
            "pk": str(uuid.uuid4()),
            "ts": int(time.time()),
            "query": question,
            "analysis": answer,
            "latencyMs": latency,
            "model": MODEL_ID
        })
        ddb.put_item(Item=item)

        log.info(json.dumps({"msg": "request_complete", "latencyMs": latency, "items_written": 1}))

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", **_cors_headers()},
            "body": json.dumps({"answer": answer, "latencyMs": latency, "model": MODEL_ID})
        }

    except Exception as e:
        log.exception("handler_error")
        return {
            "statusCode": 502,
            "headers": {"Content-Type": "application/json", **_cors_headers()},
            "body": json.dumps({"error": "bedrock_failure", "detail": str(e)[:300]})
        }
