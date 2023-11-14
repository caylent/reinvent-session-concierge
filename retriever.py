from datetime import datetime
from functools import wraps
import json
import logging
import re
import os
import time

import boto3

from sqlalchemy import create_engine, text

log_level = logging.getLevelName(os.getenv("LOG_LEVEL", "DEBUG").upper())
logger = logging.getLogger("question-service")
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def cache_with_timeout(timeout_sec):
    cache = {}

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            now = time.time()
            key = args + tuple(sorted(kwargs.items()))
            if key not in cache or now - cache[key][1] > timeout_sec:
                cache[key] = (func(*args, **kwargs), now)
            return cache[key][0]

        return wrapped

    return decorator


@cache_with_timeout(300)
def fetch_value(prompt_table, pk, sk):
    return prompt_table.get_item(Key={"PK": pk, "SK": sk})["Item"]["prompt"]


ddb_r = boto3.resource("dynamodb")
bedrock_r = boto3.client("bedrock-runtime")
prompt_table = ddb_r.Table("session-concierge")
comprehend = boto3.client("comprehend")

engine = create_engine(DB_CONNECTION_STRING)

TABLE_SCHEMA = fetch_value(prompt_table, "PROMPT#TABLE_SCHEMA", "PROMPT")
SQL_QUERY_GEN_PROMPT = fetch_value(prompt_table, "PROMPT#SQL_QUERY_GEN", "PROMPT")
RESULT_GEN_PROMPT = fetch_value(prompt_table, "PROMPT#RESULT_GEN", "PROMPT")
CORRECT_SQL_QUERY_PROMPT = fetch_value(
    prompt_table, "PROMPT#CORRECT_SQL_QUERY", "PROMPT"
)


def get_vector_for_query(text_for_vectorization):
    return json.loads(
        bedrock_r.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=json.dumps({"inputText": text_for_vectorization}),
        )["body"].read()
    )["embedding"]


def generate_query(query_str):
    query = json.loads(
        bedrock_r.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(
                {
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "max_tokens_to_sample": 1024 * 50,
                    "prompt": SQL_QUERY_GEN_PROMPT.format(
                        query_str=query_str, schema=TABLE_SCHEMA
                    ),
                }
            ),
        )["body"].read()
    )["completion"].strip()

    if "'[query_vector]'" in query:
        query = query.replace(
            "'[query_vector]'", f"'{get_vector_for_query(query_str)}'"
        )

    return re.search(r"<SQL>\s*(.*?)\s*</SQL>", query, re.DOTALL).group(1)


def correct_sql_query(original_question, query, error):
    resp = json.loads(
        bedrock_r.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(
                {
                    "temperature": 0.9,
                    "top_p": 1.0,
                    "max_tokens_to_sample": 1024 * 50,
                    "prompt": CORRECT_SQL_QUERY_PROMPT.format(
                        original_question=original_question,
                        original_query=query,
                        error=error,
                        schema=TABLE_SCHEMA,
                    ),
                }
            ),
        )["body"].read()
    )["completion"].strip()
    return re.search(r"<NEW_QUERY>\s*(.*?)\s*</NEW_QUERY>", resp, re.DOTALL).group(1)


def generate_response(original_question, results):
    return json.loads(
        bedrock_r.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(
                {
                    "temperature": 0.9,
                    "top_p": 1.0,
                    "max_tokens_to_sample": 1024 * 10,
                    "prompt": RESULT_GEN_PROMPT.format(
                        original_question=original_question, results=results
                    ),
                },
                cls=DateTimeEncoder,
            ),
        )["body"].read()
    )["completion"].strip()


def generate_response_stream(original_question, results):
    return bedrock_r.invoke_model_with_response_stream(
        modelId="anthropic.claude-v2",
        body=json.dumps(
            {
                "temperature": 0.9,
                "top_p": 1.0,
                "max_tokens_to_sample": 1024 * 50,
                "prompt": RESULT_GEN_PROMPT.format(
                    original_question=original_question, results=results
                ),
            }
        ),
    ).get("body")


def question(original_question, query=None, retry_on_exception=False):
    if query is None:
        query = generate_query(original_question)
    with engine.connect() as conn:
        try:
            rs = conn.execute(text(query))
        except Exception as ex:
            if retry_on_exception:
                new_query = correct_sql_query(original_question, query, ex)
                return question(original_question, new_query, retry_on_exception=False)
            else:
                raise
        results = rs.fetchall()
        response = generate_response(original_question, results)
    return response


def question_stream(
    original_question, query=None, retry_on_exception=False, retry_on_no_results=False
):
    yield {"status": "Comprehend Classifying Prompt"}
    prompt_classes = comprehend.classify_document(
        EndpointArn=f"arn:aws:comprehend:{comprehend.meta.region_name}:aws:document-classifier-endpoint/prompt-intent",
        Text=original_question,
    )["Classes"]
    for cls in prompt_classes:
        if cls["Name"] == "UNDESIRED_PROMPT":
            if cls["Score"] >= 0.8:
                yield {"status": "Prompt is unsafe, please try another prompt"}
                return
    yield {"status": "Generating Query"}
    if query is None:
        query = generate_query(original_question)
    logger.info(query)
    yield {"query": query}
    with engine.connect() as conn:
        try:
            yield {"status": "Executing Query"}
            rs = conn.execute(text(query))
            yield {"status": "Successful Query"}
        except Exception as ex:
            if retry_on_exception:
                yield {"status": "Error encountered, retrying query"}
                logger.debug(ex)
                new_query = correct_sql_query(original_question, query, ex)
                return question_stream(
                    original_question, new_query, retry_on_exception=False
                )
            else:
                raise
        columns = rs.keys()
        results = [
            {column: row[idx] for idx, column in enumerate(columns)}
            for row in rs.fetchall()
        ]
        logger.debug(results)
        yield {"results": results}
        if not len(results) and retry_on_no_results:
            logger.debug("Query had no results")
            yield {"status": "Query had no results, trying again"}
            new_query = correct_sql_query(original_question, query, "No results")
            return question_stream(
                original_question,
                new_query,
                retry_on_exception=False,
                retry_on_no_results=False,
            )
    logger.debug(results)
    if stream := generate_response_stream(original_question, results):
        yield {"status": "Generating response"}
        complete_response = []
        for event in stream:
            if chunk := event.get("chunk"):
                data = json.loads(chunk.get("bytes").decode())
                complete_response.append(data["completion"])
                yield data
        yield {
            "status": "Successful response!",
            "output": "".join(complete_response).strip(),
        }
