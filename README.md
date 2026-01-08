Problem Statement:
AcmeRetail’s monthly AWS bill is up 28% while EC2 CPU sits mostly in single digits. Engineers open CloudWatch dashboards ad-hoc and finance relies on Cost Explorer once a week, so no one gets daily, prioritized actions. We will build a small serverless service that:
•	Pulls the last 7 days of UnblendedCost (grouped by service) from Cost Explorer.
•	Reads per-instance CPU for the last hour from CloudWatch and EC2 metadata.
•	Summarizes risks and concrete savings recommendations using an LLM on Amazon Bedrock.
•	Persists every analysis with latency and model meta into DynamoDB.
•	Exposes a simple HTTPS endpoint (Lambda Function URL) so anyone—or ChatGPT—can ask questions like “Where can we rightsize today?”
Success criteria
•	One POST call returns a human-readable summary plus latency in milliseconds.
•	A DynamoDB item is written for each call.
•	Team can run it with 512 MB memory, 5-minute timeout, and (for the quickstart) admin permissions on the execution role, then tighten later.
2.2 Prerequisites
•	AWS account with permissions to create IAM roles/policies, Lambda, DynamoDB, and enable Bedrock model access.
•	Cost Explorer enabled for the account.
•	Bedrock access to the chosen model ID (e.g., us.anthropic.claude-3-haiku-20240307-v1:0) in your selected region.
•	Basic familiarity with the AWS Console or AWS CLI.
•	Your Lambda code (provided in the prompt).
Note on regions
•	Keep all services in the same region for simplicity. If your chosen Bedrock model is not available in that region, use a supported region for Bedrock and set the Lambda env vars accordingly. Your code currently defaults to us-east-2; set AWS_REGION to match your plan.
2.3 Approach to solve it
1.	Create a DynamoDB table to store every analysis (pk + ts as keys).
2.	Create a Lambda function (Python 3.x) using the provided code.
3.	Configure environment variables: TABLE, BEDROCK_MODEL, and AWS_REGION (or your preferred).
4.	Grant the function permissions for Cost Explorer, EC2, CloudWatch, DynamoDB, Bedrock, and logs.
5.	Increase Lambda timeout to 5 minutes and memory to 512 MB to accommodate LLM latency.
6.	Create a Function URL to invoke the service over HTTPS.
7.	Test with curl/postman, verify DynamoDB writes, and confirm a readable LLM summary is returned.

Project Solution
2.4 Complete setup guide
Step 0 — Enable Cost Explorer and Bedrock model access (only if its not enabled)
•	In Billing > Cost Explorer, enable if it’s not already on.
•	In Bedrock console, enable access to your target model (e.g., Claude 3 Haiku) in your region.


<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/68e6a121-c447-4c7d-8094-7983ce7ff044" />

Amazon Bedrock model acces is enabled by default
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/f372d0b8-4cd3-4936-b11b-67a46d571271" />

Step 1 — Create DynamoDB table
•	Name: llm_observability (or your choice—match the env var).
•	Partition key: pk (String).
•	Sort key: ts (Number).
•	Billing mode: On-demand is fine for this project.
•	No GSIs needed for now.
Created llm_observablility table in dynamodb
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/7121baee-9b50-4834-97eb-3a81ca9779be" />

Step 2 — Create the Lambda function
•	Runtime: Python 3.12 (or 3.11).
•	Architecture: x86_64 is fine.
•	Handler: lambda_function.handler (or your filename).
•	Upload your Python file as lambda_function.py (or zip if using layers/deps; this code uses only the built-in boto3).

Created lambda function named lambdaFunctionForFinops as per above
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/0d9c59d1-ee90-48be-918f-bc992be9e74b" />

Code:
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

Creted an lambda_function.py with above code
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/b006e244-e101-4784-87c7-6ded71bed599" />

Made a zip file and uploaded on lambda function:
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/142a4ff8-1ef2-4dc9-bb1d-b8a3b4c78ccf" />

<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/f9f006e9-176a-49ca-bf77-f5efba5cbb86" />

Step 3 — Set environment variables
•	TABLE = llm_observability
•	BEDROCK_MODEL = us.anthropic.claude-3-haiku-20240307-v1:0
•	AWS_REGION = us-east-2 (or the region you picked)
•	Optional (if you prefer): also set BEDROCK_REGION for clarity, but your code reads AWS_REGION.
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/4cfbf872-2ff9-42c1-971e-1eaa892f2277" />

Step 4 — Configure memory and timeout
•	Memory: 512 MB
•	Timeout: 5 minutes (300 seconds)
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/187572d1-db23-42da-ab2d-3261382e852f" />

Step 5 — Execution role and permissions
Quickstart (what you requested):
•	Attach AdministratorAccess to the Lambda execution role to avoid permission blockers during the first run.
Best practice (apply after initial validation):
Replace admin with a least-privilege inline policy that covers exactly what the function uses. Example policy actions (tighten resources later):
IAM policy actions (allow on Resource: "*", then scope down):
•	Logs
•	logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
•	Cost Explorer
•	ce:GetCostAndUsage
•	EC2 and CloudWatch metrics
•	ec2:DescribeInstances
•	cloudwatch:GetMetricStatistics
•	DynamoDB
•	dynamodb:PutItem, dynamodb:DescribeTable
•	Bedrock
•	bedrock:InvokeModel, bedrock:InvokeModelWithResponseStream
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/3207fbb1-c16a-4c85-870a-664c64249129" />


Provided administrator access
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/178aef61-feeb-44ab-b48b-bfb4cdab95da" />

Step 6 — Create a Function URL
•	In Lambda > your function > Function URL > Create.
•	Auth type: choose NONE for simplest testing or AWS_IAM if you want signed requests.
•	If you choose NONE, consider adding an API key header check in code later.
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/b88ab054-805d-4d68-aabd-f3eb989a8e0f" />

Testing the lambda function:
Created testEvent
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/e75d32c9-58e8-4c09-9c96-c2341d91313b" />

Executing testEvent:
Output:
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/fab4552a-69c4-4109-a232-d5bcb6dbd555" />
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/4a4d5de8-721a-46ad-89f3-cd23a835a1f8" />


Create MyGPT - AWS GPT
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/754d0ecf-b6d0-416b-ab3d-6d5165230278" />
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/62fa4588-8f0a-4aa2-8722-a33724694f89" />

Saved it :
<img width="979" height="552" alt="image" src="https://github.com/user-attachments/assets/71feb3f3-5a55-4877-8ca5-05b3e6f28d75" />

















