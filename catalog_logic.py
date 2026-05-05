import os
import re
import time
import json
import requests
import xml.etree.ElementTree as ET
from lxml import html
from google import genai
from google.genai import types
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

PRODUCTS = []
SESSIONS = {}

# Embedding index
EMBEDDING_MATRIX = None   # np.ndarray shape (N, D)
EMBEDDING_CACHE = {}      # query_text -> np.ndarray

# TF-IDF fallback (lazy-loaded only if embedding API fails at query time)
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
Jesteś wyszukiwarką i doradcą produktowym sklepu DABSTORY.

Zasady:
- Odpowiadasz wyłącznie po polsku.
- Używasz tylko danych przekazanych w sekcji Dane katalogowe.
- Nie wymyślasz parametrów, których nie ma w danych.
- Jeśli czegoś nie ma w danych, napisz dokładnie: "Brak potwierdzenia w danych katalogowych".
- Nie wyciągaj wniosków z samego zdjęcia produktu.
- Jeśli produkt ma pole "wariant", uwzględnij je w odpowiedzi.
- Odróżniaj klasyczną szczotkę WC od zestawu 2w1 ze stojakiem lub uchwytem na papier.
- Jeśli produkt jest wariantem ze stojakiem, zestawem 2w1 albo ma zintegrowany uchwyt na papier, zaznacz to wyraźnie w odpowiedzi.
- Nie traktuj części zamiennej jako pełnego produktu, jeśli dane wskazują, że to tylko element wymienny.
- Nie pisz marketingowo.
- Nie używaj zwrotów typu "z przyjemnością", "mam nadzieję", "chętnie pomogę".

Zasady interpretacji:
- "in stock" oznacza "dostępny od ręki".
- "on demand" oznacza "na zamówienie".
- Jeśli pytanie jest kontynuacją wcześniejszego wątku, uwzględnij kontekst rozmowy.
- Jeśli użytkownik rozpoczyna nowy temat, nie przenoś niepotrzebnie wcześniejszych filtrów.

Format odpowiedzi:
- Jeśli użytkownik pyta o listę produktów, zwróć krótką listę punktowaną.
- Dla każdego produktu zawsze wypisz osobno:
  1. nazwę,
  2. wariant produktu,
  3. cenę,
  4. dostępność,
  5. kolor,
  6. materiał,
  7. typ montażu,
  8. wymiary,
  9. link.
- Nie pomijaj pola "wariant", jeśli występuje w danych katalogowych.
- Jeśli użytkownik pyta o porównanie, porównaj produkty w punktach.
- Jeśli wyników jest dużo, pokaż najtrafniejsze.
- Zachowuj odpowiedzi zwięzłe i konkretne.
- Jeśli dwa produkty różnią się głównie kolorem, napisz to wprost.
- Jeśli w wynikach są zarówno klasyczne szczotki WC, jak i zestawy 2w1, zaznacz tę różnicę przy każdym produkcie.
""".strip()

FALLBACK_MODELS = [
    MODEL_NAME,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

FOLLOWUP_PHRASES = [
    "a które", "a ktore", "które z nich", "ktore z nich", "z nich",
    "te", "ten", "ta", "tamte", "pierwsze", "drugie",
    "tańsze", "tansze", "droższe", "drozsze",
    "dostępne", "dostepne", "od ręki", "od reki"
]

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_polish_text(text):
    text = str(text).lower().strip()
    for k, v in {"ą":"a","ć":"c","ę":"e","ł":"l","ń":"n","ó":"o","ś":"s","ż":"z","ź":"z"}.items():
        text = text.replace(k, v)
    return text


def is_followup_question(question):
    q = normalize_polish_text(question)
    return any(phrase in q for phrase in FOLLOWUP_PHRASES)


# ---------------------------------------------------------------------------
# Product domain helpers
# ---------------------------------------------------------------------------

def product_variant_label(p):
    pt = p.get("product_type", "")
    title = normalize_polish_text(p.get("title", ""))
    if pt == "toilet_brush_stand":
        return "zestaw 2w1: szczotka WC + stojak/uchwyt na papier"
    if pt == "replacement_brush":
        return "część zamienna: wymienna szczotka"
    if pt == "toilet_brush":
        return "klasyczna szczotka WC"
    if "stojak" in title and "szczotka" in title:
        return "zestaw 2w1: szczotka WC + stojak"
    return ""


def infer_product_type(title):
    t = normalize_polish_text(title)
    if "wymienna szczotka" in t:
        return "replacement_brush"
    if "stojak" in t and "szczotka" in t:
        return "toilet_brush_stand"
    if "szczotka" in t and "wc" in t:
        return "toilet_brush"
    return "other"


def infer_color(title):
    t = normalize_polish_text(title)
    if "czarn" in t: return "czarny"
    if "bial" in t:  return "biały"
    if "bezow" in t or "bezowa" in t: return "beżowy"
    return ""


# ---------------------------------------------------------------------------
# Product text builders
# ---------------------------------------------------------------------------

def build_embedding_text(p):
    parts = [
        p.get("title", ""),
        p.get("brand", ""),
        p.get("category", ""),
        product_variant_label(p),
        p.get("color", ""),
        p.get("mount_type", ""),
        p.get("material", ""),
        p.get("finish", ""),
        p.get("description", "")[:300],
    ]
    return " ".join(x for x in parts if x)


def build_product_json(p):
    return {
        "id": p.get("id"),
        "nazwa": p.get("title", ""),
        "marka": p.get("brand", ""),
        "mpn": p.get("mpn", ""),
        "gtin": p.get("gtin", ""),
        "dostepnosc": p.get("availability", ""),
        "cena": p.get("price", ""),
        "kategoria": p.get("category", ""),
        "typ_produktu": p.get("product_type", ""),
        "wariant": product_variant_label(p),
        "kolor": p.get("color", ""),
        "typ_montazu": p.get("mount_type", ""),
        "material": p.get("material", ""),
        "wykonczenie": p.get("finish", ""),
        "wymiary": p.get("dimensions", ""),
        "link": p.get("link", ""),
        "zdjecie": p.get("image", ""),
        "opis": p.get("description", ""),
    }


# ---------------------------------------------------------------------------
# XML loading
# ---------------------------------------------------------------------------

def load_products_from_xml(xml_path="products.xml"):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    NS = "{http://base.google.com/ns/1.0}"
    products = []
    for i, item in enumerate(root.findall(".//item"), start=1):
        title = item.findtext("title", default="").strip()
        p = {
            "id": i,
            "title": title,
            "brand": item.findtext(f"{NS}brand", default="").strip(),
            "mpn": item.findtext(f"{NS}mpn", default="").strip(),
            "gtin": item.findtext(f"{NS}gtin", default="").strip(),
            "availability": item.findtext(f"{NS}availability", default="").strip(),
            "price": item.findtext(f"{NS}price", default="").strip(),
            "category": item.findtext(f"{NS}google_product_category", default="").strip(),
            "color": infer_color(title),
            "mount_type": "", "material": "", "finish": "", "dimensions": "",
            "link": item.findtext("link", default="").strip(),
            "image": item.findtext(f"{NS}image_link", default="").strip(),
            "description": item.findtext("description", default="").strip(),
            "product_type": infer_product_type(title),
        }
        products.append(p)
    return products


# ---------------------------------------------------------------------------
# Page scraping — runs ONCE at startup
# ---------------------------------------------------------------------------

def fetch_product_page_fields(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        tree = html.fromstring(r.text)
        text = re.sub(r"\s+", " ", " ".join(tree.xpath("//body//text()"))).strip()
        t = normalize_polish_text(text)

        material = []
        if "stal nierdzewna" in t: material.append("stal nierdzewna")
        if "tworzywo sztuczne" in t: material.append("tworzywo sztuczne")

        has_wall  = any(x in t for x in ["wiszace", "wiszacy", "wiszaca"])
        has_floor = any(x in t for x in ["wolnostojace", "wolnostojacy", "wolnostojaca"])
        if has_wall and has_floor: mount_type = "wiszący / wolnostojący"
        elif has_wall:             mount_type = "wiszący"
        elif has_floor:            mount_type = "wolnostojący"
        else:                      mount_type = ""

        finish = []
        if "malowana proszkowo" in t: finish.append("powłoka malowana proszkowo")
        if "polmat" in t:             finish.append("półmat")
        elif any(x in t for x in ["matowa", "matowe", "matowy"]): finish.append("mat")

        return {
            "material": ", ".join(dict.fromkeys(material)),
            "mount_type": mount_type,
            "finish": ", ".join(dict.fromkeys(finish)),
        }
    except Exception:
        return {"material": "", "mount_type": "", "finish": ""}


def enrich_all_products(products):
    print(f"[catalog] Enriching {len(products)} products from product pages...")
    for idx, p in enumerate(products, start=1):
        if not p.get("link"):
            continue
        if not p.get("material") or not p.get("mount_type") or not p.get("finish"):
            extra = fetch_product_page_fields(p["link"])
            p["material"]   = p["material"]   or extra["material"]
            p["mount_type"] = p["mount_type"] or extra["mount_type"]
            p["finish"]     = p["finish"]     or extra["finish"]
        if idx % 10 == 0:
            print(f"[catalog] Enriched {idx}/{len(products)}")
        time.sleep(0.15)
    print("[catalog] Enrichment complete.")
    return products


# ---------------------------------------------------------------------------
# Embedding index — with 429 retry + exponential backoff
# ---------------------------------------------------------------------------

def _embed_batch_with_retry(texts, max_retries=6):
    """Embed a single batch, retrying on 429 with exponential backoff."""
    delay = 15  # seconds — free tier window is 60s/100req
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
            )
            return np.array([e.values for e in result.embeddings], dtype=np.float32)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = delay * (2 ** attempt)  # 15, 30, 60, 120 ...
                print(f"[catalog] 429 rate limit, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Embedding failed after {max_retries} retries (persistent 429).")


def build_embedding_index(products):
    global EMBEDDING_MATRIX
    print(f"[catalog] Building embedding index for {len(products)} products...")
    texts = [build_embedding_text(p) for p in products]

    # Batch size 50 — well under the 100 req/min free tier limit
    # Sleep 65s between batches to reset the per-minute quota window
    batch_size = 50
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"[catalog] Embedding batch {i//batch_size + 1}/{-(-len(texts)//batch_size)} ({len(batch)} items)...")
        batches.append(_embed_batch_with_retry(batch))
        if i + batch_size < len(texts):  # no sleep after last batch
            time.sleep(65)  # wait out the 1-minute quota window

    EMBEDDING_MATRIX = np.vstack(batches)
    norms = np.linalg.norm(EMBEDDING_MATRIX, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    EMBEDDING_MATRIX = EMBEDDING_MATRIX / norms
    print(f"[catalog] Embedding index ready. Shape: {EMBEDDING_MATRIX.shape}")


def _build_tfidf_fallback():
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX
    if _TFIDF_VECTORIZER is not None:
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("[catalog] Building TF-IDF fallback index...")
    corpus = [build_embedding_text(p) for p in PRODUCTS]
    _TFIDF_VECTORIZER = TfidfVectorizer()
    _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(corpus)


def search_products(query, top_k=8):
    if EMBEDDING_MATRIX is not None:
        try:
            query_vec = EMBEDDING_CACHE.get(query)
            if query_vec is None:
                query_vec = _embed_batch_with_retry([query])[0]
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm
                EMBEDDING_CACHE[query] = query_vec
            scores = EMBEDDING_MATRIX @ query_vec
            ranked_idx = scores.argsort()[::-1][:top_k]
            return [PRODUCTS[i] for i in ranked_idx]
        except Exception as e:
            print(f"[catalog] Embedding search failed, falling back to TF-IDF: {e}")

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    _build_tfidf_fallback()
    qvec = _TFIDF_VECTORIZER.transform([query])
    sims = cos_sim(qvec, _TFIDF_MATRIX).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [PRODUCTS[i] for i in ranked_idx]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def reset_session(session_id):
    SESSIONS[session_id] = {"chat_memory": []}


def get_session(session_id):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"chat_memory": []}
    return SESSIONS[session_id]


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

def chat_with_session(session_id, question, top_k=8, retries=3, memory_turns=6):
    session = get_session(session_id)
    chat_memory = session["chat_memory"]

    history_text = ""
    if chat_memory:
        recent = chat_memory[-memory_turns:]
        history_text = "\n\nHistoria rozmowy:\n" + "\n".join(
            [f"{m['role'].upper()}: {m['text']}" for m in recent]
        )

    retrieval_query = question
    if chat_memory and is_followup_question(question):
        last_user_questions = [m["text"] for m in chat_memory if m["role"] == "user"]
        if last_user_questions:
            retrieval_query = f"{last_user_questions[-1]} {question}"

    matches = search_products(retrieval_query, top_k=top_k)
    context_json = json.dumps([build_product_json(p) for p in matches], ensure_ascii=False, indent=2)

    user_message = f"""
{history_text}

Aktualne pytanie użytkownika:
{question}

Dane katalogowe (JSON):
{context_json}
""".strip()

    last_error = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                    )
                )
                answer = response.text
                chat_memory.append({"role": "user", "text": question})
                chat_memory.append({"role": "assistant", "text": answer})
                session["chat_memory"] = chat_memory[-(memory_turns * 2):]
                return answer
            except Exception as e:
                last_error = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                break

    return f"Wystąpił błąd po stronie modelu: {last_error}"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_catalog():
    global PRODUCTS
    PRODUCTS = load_products_from_xml("products.xml")
    PRODUCTS = enrich_all_products(PRODUCTS)
    build_embedding_index(PRODUCTS)
