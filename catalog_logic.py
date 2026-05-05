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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()

EMBEDDINGS_CACHE_FILE = "embeddings.npy"
PRODUCTS_CACHE_FILE = "products_cache.json"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PRODUCTS = []
SESSIONS = {}

EMBEDDING_MATRIX = None
EMBEDDING_CACHE = {}

_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
Jesteś wyszukiwarką i doradcą produktowym sklepu DABSTORY.

Zasady ogólne:
- Odpowiadasz wyłącznie po polsku.
- Używasz tylko danych przekazanych w sekcji "Dane katalogowe".
- Nie wymyślasz wartości, których nie ma w danych.
- Jeśli pole JSON ma wartość null lub nie istnieje, napisz "brak danych" dla tego pola.
- Nie pisz marketingowo. Nie używaj zwrotów: "z przyjemnością", "mam nadzieję", "chętnie pomogę".

Zasady odczytu danych:
- Pole "nazwa" zawiera pełną nazwę produktu, w tym kolor i wariant montażu — wyciągnij je stamtąd, jeśli dedykowane pola "kolor" lub "typ_montazu" są null.
  Przykład: "Szczotka WC EQULA - Czarna Wersja Wiszące" → kolor: czarny, typ montażu: wiszące.
- Pole "wariant" opisuje typ produktu (klasyczna szczotka, zestaw 2w1, część zamienna) — zawsze je wypisz.
- "in stock" = dostępny od ręki, "on demand" = na zamówienie.
- Odróżniaj klasyczną szczotkę WC od zestawu 2w1 ze stojakiem. Zaznacz tę różnicę przy każdym produkcie.
- Nie traktuj części zamiennej jako pełnego produktu.

Format odpowiedzi:
- Lista produktów: krótka lista punktowana.
- Dla każdego produktu wypisz:
  1. nazwa
  2. wariant (typ produktu)
  3. cena
  4. dostępność
  5. kolor (z pola "kolor"; jeśli null — wyciągnij z nazwy)
  6. materiał
  7. typ montażu (z pola "typ_montazu"; jeśli null — wyciągnij z nazwy)
  8. wymiary
  9. link
- Jeśli pole ma wartość null i nie można go wyciągnąć z nazwy → napisz "brak danych".
- Porównanie: punkty różnic.
- Jeśli wyników jest dużo, pokaż najtrafniejsze.
- Zachowuj odpowiedzi zwięzłe i konkretne.
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
# Query filter extraction
# ---------------------------------------------------------------------------

COLOR_KEYWORDS = {
    "czarn": "czarny",
    "biał": "biały",
    "bial": "biały",
    "bezow": "beżowy",
    "bezó": "beżowy",
}

AVAILABILITY_KEYWORDS = {
    "od ręki": "in stock",
    "od reki": "in stock",
    "dostępn": "in stock",
    "dostepn": "in stock",
    "na zamówienie": "on demand",
    "na zamowienie": "on demand",
}


def extract_filters(query):
    q = normalize_polish_text(query)
    filters = {}
    for kw, color in COLOR_KEYWORDS.items():
        if kw in q:
            filters["color"] = color
            break
    for kw, avail in AVAILABILITY_KEYWORDS.items():
        if kw in q:
            filters["availability"] = avail
            break
    return filters


def apply_filters(products, filters):
    result = products
    if "color" in filters:
        result = [p for p in result if p.get("color") == filters["color"]]
    if "availability" in filters:
        result = [p for p in result if p.get("availability") == filters["availability"]]
    return result


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
    return None


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
    return None


def _none_if_empty(value):
    if value is None:
        return None
    v = str(value).strip()
    return v if v else None


# ---------------------------------------------------------------------------
# Product text builders
# ---------------------------------------------------------------------------

def build_embedding_text(p):
    parts = [
        p.get("title", ""),
        p.get("brand", ""),
        p.get("category", ""),
        product_variant_label(p) or "",
        p.get("color", "") or "",
        p.get("mount_type", "") or "",
        p.get("material", "") or "",
        p.get("finish", "") or "",
        (p.get("description", "") or "")[:300],
    ]
    return " ".join(x for x in parts if x)


def build_product_json(p):
    return {
        "id": p.get("id"),
        "nazwa": _none_if_empty(p.get("title")),
        "marka": _none_if_empty(p.get("brand")),
        "mpn": _none_if_empty(p.get("mpn")),
        "gtin": _none_if_empty(p.get("gtin")),
        "dostepnosc": _none_if_empty(p.get("availability")),
        "cena": _none_if_empty(p.get("price")),
        "kategoria": _none_if_empty(p.get("category")),
        "typ_produktu": _none_if_empty(p.get("product_type")),
        "wariant": product_variant_label(p),
        "kolor": _none_if_empty(p.get("color")),
        "typ_montazu": _none_if_empty(p.get("mount_type")),
        "material": _none_if_empty(p.get("material")),
        "wykonczenie": _none_if_empty(p.get("finish")),
        "wymiary": _none_if_empty(p.get("dimensions")),
        "link": _none_if_empty(p.get("link")),
        "zdjecie": _none_if_empty(p.get("image")),
        "opis": _none_if_empty(p.get("description")),
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
# Page scraping
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
# Embedding cache persistence
# ---------------------------------------------------------------------------

def _save_cache(products, matrix):
    try:
        with open(PRODUCTS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False)
        np.save(EMBEDDINGS_CACHE_FILE, matrix)
        print(f"[catalog] Cache saved.")
    except Exception as e:
        print(f"[catalog] Warning: could not save cache: {e}")


def _load_cache():
    if not os.path.isfile(PRODUCTS_CACHE_FILE) or not os.path.isfile(EMBEDDINGS_CACHE_FILE):
        return None, None
    try:
        xml_mtime = os.path.getmtime("products.xml")
        cache_mtime = os.path.getmtime(PRODUCTS_CACHE_FILE)
        if xml_mtime > cache_mtime:
            print("[catalog] products.xml newer than cache — rebuilding.")
            return None, None
        with open(PRODUCTS_CACHE_FILE, "r", encoding="utf-8") as f:
            products = json.load(f)
        matrix = np.load(EMBEDDINGS_CACHE_FILE)
        print(f"[catalog] Loaded from cache: {len(products)} products, matrix {matrix.shape}")
        return products, matrix
    except Exception as e:
        print(f"[catalog] Cache load failed ({e}) — rebuilding.")
        return None, None


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

def _embed_batch_with_retry(texts, max_retries=8):
    delay = 10
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
                wait = delay * (2 ** attempt)
                print(f"[catalog] 429 rate limit, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Embedding failed after {max_retries} retries.")


def build_embedding_index(products):
    global EMBEDDING_MATRIX
    if not EMBEDDING_MODEL:
        print("[catalog] EMBEDDING_MODEL not set — skipping embedding index (TF-IDF will be used).")
        return
    print(f"[catalog] Building embedding index for {len(products)} products...")
    texts = [build_embedding_text(p) for p in products]
    batch_size = 20
    batches = []
    total_batches = -(-len(texts) // batch_size)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"[catalog] Embedding batch {i//batch_size+1}/{total_batches} ({len(batch)} items)...")
        batches.append(_embed_batch_with_retry(batch))
        if i + batch_size < len(texts):
            time.sleep(5)
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
    print("[catalog] Building TF-IDF index...")
    corpus = [build_embedding_text(p) for p in PRODUCTS]
    _TFIDF_VECTORIZER = TfidfVectorizer()
    _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(corpus)
    print("[catalog] TF-IDF index ready.")


def search_products(query, top_k=8, filters=None):
    candidate_k = min(len(PRODUCTS), max(top_k * 5, 40))

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
            ranked_idx = scores.argsort()[::-1][:candidate_k]
            candidates = [PRODUCTS[i] for i in ranked_idx]
        except Exception as e:
            print(f"[catalog] Embedding search failed, falling back to TF-IDF: {e}")
            candidates = None
    else:
        candidates = None

    if candidates is None:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        _build_tfidf_fallback()
        qvec = _TFIDF_VECTORIZER.transform([query])
        sims = cos_sim(qvec, _TFIDF_MATRIX).flatten()
        ranked_idx = sims.argsort()[::-1][:candidate_k]
        candidates = [PRODUCTS[i] for i in ranked_idx]

    if filters:
        filtered = apply_filters(candidates, filters)
        if filtered:
            return filtered[:top_k]
        filtered = apply_filters(PRODUCTS, filters)
        if filtered:
            if EMBEDDING_MATRIX is not None:
                query_vec = EMBEDDING_CACHE.get(query)
                if query_vec is not None:
                    idxs = [p["id"] - 1 for p in filtered]
                    sub_scores = (EMBEDDING_MATRIX[idxs] @ query_vec)
                    order = sub_scores.argsort()[::-1]
                    filtered = [filtered[i] for i in order]
            return filtered[:top_k]

    return candidates[:top_k]


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

    filters = extract_filters(retrieval_query)
    matches = search_products(retrieval_query, top_k=top_k, filters=filters or None)
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
    global PRODUCTS, EMBEDDING_MATRIX

    # Try loading from cache only if embeddings are enabled
    if EMBEDDING_MODEL:
        cached_products, cached_matrix = _load_cache()
        if cached_products is not None and cached_matrix is not None:
            PRODUCTS = cached_products
            EMBEDDING_MATRIX = cached_matrix
            return

    # Full rebuild
    PRODUCTS = load_products_from_xml("products.xml")
    PRODUCTS = enrich_all_products(PRODUCTS)

    if EMBEDDING_MODEL:
        build_embedding_index(PRODUCTS)
        if EMBEDDING_MATRIX is not None:
            _save_cache(PRODUCTS, EMBEDDING_MATRIX)
    else:
        print("[catalog] Embeddings disabled — using TF-IDF search.")
        _build_tfidf_fallback()
