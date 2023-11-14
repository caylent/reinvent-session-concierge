import json
import re
import os
import boto3
import concurrent.futures

session = boto3.Session(profile_name="caylent-sso")
bedrock_r = session.client("bedrock-runtime")

KEYS_TO_KEEP = [
    "title",
    "description",
    "venueName",
    "sessionType",
    "startDateTime",
    "endDateTime",
    "thirdPartyID",
    "trackName",
    "floorplanName",
    "locationName",
    "tags",
    "speakers",
]


def title_to_snake(name):
    # Convert title or camel case to snake case
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def process_item(item):
    filtered_item = {
        title_to_snake(key) if key != "thirdPartyID" else "session_id": item[key]
        for key in KEYS_TO_KEEP
        if key in item
    }

    # cleanup speakers dictionary
    filtered_item["speakers"] = [
        f"{speaker['firstName']} {speaker['lastName']}"
        for speaker in filtered_item["speakers"]
    ]

    # cleanup tags dictionaryColumn 'tag_venue' has at least one list with more than one value.
    restructured_tags_map = {
        "Additional Services": "tag_additional_services",
        "Area of Interest": "tag_area_of_interest",
        "Day": "tag_day",
        "Industry": "tag_industry",
        "Level": "tag_level",
        "Role": "tag_role",
        "Services": "tag_services",
        "Topic": "tag_topic",
        "Venue": "tag_venue",
    }
    restructured_tags = {}

    for tag in filtered_item["tags"]:
        parent_tag_name = tag.get("parentTagName")
        if parent_tag_name:
            restructured_tags.setdefault(parent_tag_name, []).append(tag["tagName"])
    for tag_name, tag_value in restructured_tags.items():
        new_key = restructured_tags_map.get(tag_name)
        if new_key:
            filtered_item[new_key] = tag_value
    for tag_name in ["tag_additonal_services", "tag_day", "tag_level"]:
        if tag_name in filtered_item:
            filtered_item[tag_name] = filtered_item[tag_name][0]
            if tag_name == "tag_level":
                filtered_item[tag_name] = int(filtered_item[tag_name][:3])

    # Remove the original 'tags' key if it's no longer needed.
    filtered_item.pop("tags", None)

    # create the joined text for the embedding - separated by newlines
    text_for_vectorization = str(
        [
            filtered_item[key]
            for key in [
                "session_id",
                "title",
                "description",
                "session_type",
                "venue_name",
                "location_name",
                "track_name",
                "speakers",
            ]
            + [k for k in filtered_item.keys() if k.startswith("tag_")]
        ]
    )
    response = bedrock_r.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps({"inputText": text_for_vectorization}),
    )
    embedding = json.loads(response["body"].read())["embedding"]
    filtered_item["embedding"] = embedding

    return filtered_item


with open("session3.json", "r") as file:
    parsed_json = json.load(file)

total_items = len(parsed_json["data"])
print(f"Total items to process: {total_items}")

filtered_data = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(process_item, parsed_json["data"])
    filtered_data.extend(results)

print("Processing complete.")

# Save the filtered data to a new JSON file
with open("session3_filtered_with_embeddings.json", "w") as file:
    json.dump({"data": filtered_data}, file)


from datetime import datetime
import json

from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY


def preprocess_session_data(session_data):
    integer_fields = ["start_date_time", "end_date_time", "tag_level"]
    for field in integer_fields:
        if session_data.get(field) == "":
            session_data[field] = None
    for field in ["start_date_time", "end_date_time"]:
        if session_data.get(field):
            session_data[field] = datetime.fromtimestamp(session_data[field])
    return session_data


DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
engine = create_engine(DB_CONNECTION_STRING)
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

Base = declarative_base()


class Session(Base):
    __tablename__ = "session"
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    title = Column(String)
    description = Column(String)
    venue_name = Column(String)
    session_type = Column(String)
    start_date_time = Column(DateTime)
    end_date_time = Column(DateTime)
    track_name = Column(String)
    floorplan_name = Column(String)
    location_name = Column(String)
    speakers = Column(ARRAY(String))
    tag_role = Column(ARRAY(String))
    tag_area_of_interest = Column(ARRAY(String))
    tag_additional_services = Column(ARRAY(String))
    tag_services = Column(ARRAY(String))
    tag_industry = Column(ARRAY(String))
    tag_day = Column(String)
    tag_venue = Column(ARRAY(String))
    tag_topic = Column(ARRAY(String))
    tag_level = Column(Integer)
    embedding = Column(Vector(1536))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()
file_path = "session3_filtered_with_embeddings.json"
with open(file_path, "r") as f:
    sessions = json.load(f)["data"]

preprocessed_sessions = [preprocess_session_data(session) for session in sessions]
session_to_insert = [Session(**session) for session in preprocessed_sessions]
try:
    db.bulk_save_objects(session_to_insert)
    db.commit()
except Exception as e:
    db.rollback()
    print(str(e))
finally:
    db.close()
