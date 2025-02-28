from io import BytesIO
from audiorecorder import audiorecorder
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
from hashlib import md5
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

#
# MAIN

#
# CONSTANCES
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
EMBEDDING_DIM = 3072
EMBEDDING_MODEL = "text-embedding-3-large"
QDRANT_COLLECTION_NAME = "notes"

env = dotenv_values(".env")


#
# FUNCTIONS
def get_client():
    return OpenAI(api_key=st.session_state["OpenAI_key"])

def openai_whisper(audio_buffer):
    openai_client = get_client()
    audio_buffer.name="audio.mp3"
    audio_buffer.seek(0)
    transcription = openai_client.audio.transcriptions.create(
        file=audio_buffer,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )
    return transcription.text

#
# db
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url="https://8b4853b0-68ad-47bc-9b46-296189bb689c.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="wFHCy66r3CAUtk3yk_KrRGEUR-GqfjbxaQLnVcmgIcFMkvVXCoDHSQ",
)

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()

    if not qdrant_client.collection_exists(
        collection_name=QDRANT_COLLECTION_NAME    
    ):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config= VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
    else:
        print(f"Kolekcja {QDRANT_COLLECTION_NAME} już istnieje")

def get_embedding(text):
    openai_client = get_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding

def add_note_to_db(text):
    qdrant_client = get_qdrant_client()

    all_points = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=all_points.count + 1,
                vector=get_embedding(text),
                payload={
                    "text":text,
                },
            )
        ]
    )

def list_notes_from_db(query=None):
    qdrant_client = get_qdrant_client()

    if not query:
        notes = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=10,
        )[0]

        result = []
        for note in notes:
            result.append(
                {
                    "text": note.payload["text"],
                    "score": None,
                }
            )
        return result
    else:
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(query),
            limit=5,
        )
        result = []
        for note in notes:
            result.append(
                {
                    "text": note.payload["text"],
                    "score": note.score,
                }
            )
        return result
    

    
#
if not st.session_state.get("OpenAI_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["OpenAI_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Wpisz swój klucz AI, żeby korzystać z aplikacji: ")
        st.session_state["OpenAI_key"] = st.text_input("Klucz AI: ", type="password")
        if st.session_state.get("OpenAI_key"):
            st.rerun()

if not st.session_state.get("OpenAI_key"):
    st.stop()


#
if not "note_audio_bytes" in st.session_state:
    st.session_state["note_audio_bytes"] = None

if not "note_audio_bytes_md5" in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if not "note_txt_bytes" in st.session_state:
    st.session_state["note_txt_bytes"] = ""

st.set_page_config(page_title="Audio Notatki", layout="centered")

st.title("Audio Notatnik")

tran_txt, search_txt = st.tabs(["Transkrybuj notatkę", "Wyszukaj notatkę"])

with tran_txt:
    assure_db_collection_exists()
    note_audio = audiorecorder(
        start_prompt="Nagraj notatkę",
        stop_prompt="Zatrzymaj nagrywanie",
    )

    if note_audio:
        audio_buffer = BytesIO() # Create a audio buffer (object file-like) 
        note_audio.export(audio_buffer, format='mp3') # Write the note_audio to the audio buffer (object file-like) in mp3 format 
        st.session_state["note_audio_bytes"] = audio_buffer.getvalue() # Get the raw bytes from audio buffer (object file-like) and write to st.session_state["note_audio_bytes"]
        st.audio(st.session_state["note_audio_bytes"], format='audio/mp3') # Use them (raw bytes) from st.session_state["note_audio_bytes"] to display at screen as audio button
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest() # Hold the current a raw bytes (st.session_state["note_audio_bytes"]) from audio buffer as a md5().hexdigest()

        if current_md5 != st.session_state["note_audio_bytes_md5"]:
            st.session_state["note_audio_bytes_md5"] = current_md5
            st.session_state["note_txt_bytes"] = "" # Now, the text into text area is disappear 

        if st.button("Transkrybuj tekst"):
            st.session_state["note_txt_bytes"] = openai_whisper(audio_buffer) # Launch the transcription a voice to a text
            #st.write(type(st.session_state["note_txt_bytes"]))

        if st.session_state["note_txt_bytes"]:
            st.text_area(
                "Edytuj notatkę",
                value=st.session_state["note_txt_bytes"],
                disabled=False,
            )

    if st.session_state["note_txt_bytes"] and st.button("Zapisz notatkę", disabled=not st.session_state["note_txt_bytes"]):
        add_note_to_db(st.session_state["note_txt_bytes"])
        st.toast("Notatka zapisana", icon="🎉")

with search_txt:
    query = st.text_input("Wyszukaj notatkę:")
    if st.button("Szukaj"):
        notes = list_notes_from_db(query)
        for note in notes:
            st.markdown(note["text"])
            if note["score"]:
                st.markdown(f':violet[{note["score"]}]')



    #st.write(list_notes_from_db("test"))