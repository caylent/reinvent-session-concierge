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
                    "max_tokens_to_sample": 1024 * 10,
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
                    "max_tokens_to_sample": 1024 * 10,
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
                }
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
                "max_tokens_to_sample": 1024 * 10,
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


def question_stream(original_question, query=None, retry_on_exception=False):
    yield json.dumps({"status": "generating query"})
    if query is None:
        query = generate_query(original_question)
    logger.info(query)
    yield json.dumps({"query": query})
    with engine.connect() as conn:
        try:
            yield json.dumps({"status": "executing query"})
            rs = conn.execute(text(query))
            yield json.dumps({"status": "successful query"})
        except Exception as ex:
            if retry_on_exception:
                yield json.dumps({"status": "retrying query"})
                logger.debug(ex)
                new_query = correct_sql_query(original_question, query, ex)
                return question_stream(
                    original_question, new_query, retry_on_exception=False
                )
            else:
                raise
        results = rs.fetchall()
        if not results:
            yield json.dumps({"status": "query had no results, trying again"})
            new_query = correct_sql_query(original_question, query, "No results")
            return question_stream(
                original_question, new_query, retry_on_exception=False
            )
        logger.debug(results)
        if stream := generate_response_stream(original_question, results):
            yield json.dumps({"status": "generating response"})
            complete_response = []
            for event in stream:
                if chunk := event.get("chunk"):
                    data = json.loads(chunk.get("bytes").decode())
                    complete_response.append(data["completion"])
                    yield json.dumps(data)
            yield json.dumps(
                {
                    "status": "successful response",
                    "output": "".join(complete_response).strip(),
                }
            )
