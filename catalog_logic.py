import os
import re
import time
import requests
from lxml import etree
from lxml import html
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

PRODUCTS = []
SESSIONS = {}
VECTORIZER = None
DOC_MATRIX = None

SYSTEM_INSTRUCTION = """
Jesteś wyszukiwarką i doradcą produktowym sklepu DABSTORY.

Zasady:
- Odpowiadasz wyłącznie po polsku.
- Używasz tylko danych przekazanych w sekcji Dane katalogowe.
- Nie wymyślasz parametrów, których nie ma w danych.
- Jeśli czegoś nie ma w danych, napisz dokładnie: "Brak potwierdzenia w danych katalogowych".
- Nie wyciągaj wniosków z samego zdjęcia produktu.
- Jeśli produkt ma pole "Wariant produktu", uwzględnij je w odpowiedzi.
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
- Nie pomijaj pola "Wariant produktu", jeśli występuje w danych katalogowych.
- Jeśli użytkownik pyta o porównanie, porównaj produkty w punktach.
- Jeśli wyników jest dużo, pokaż najtrafniejsze.
- Zachowuj odpowiedzi zwięzłe i konkretne.
- Jeśli dwa produkty różnią się głównie kolorem, napisz to wprost.
- Jeśli w wynikach są zarówno klasyczne szczotki WC, jak i zestawy 2w1, zaznacz tę różnicę przy każdym produkcie.
""".strip()

FOLLOWUP_PHRASES = [
    "a które", "a ktore", "które z nich", "ktore z nich", "z nich",
    "te", "ten", "ta", "tamte", "pierwsze", "drugie",
    "tańsze", "tansze", "droższe", "drozsze",
    "dostępne", "dostepne", "od ręki", "od reki"
]

def normalize_polish_text(text):
    text = str(text).lower().strip()
    replacements = {
        "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
        "ó": "o", "ś": "s", "ż": "z", "ź": "z"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def is_followup_question(question):
    q = normalize_polish_text(question)
    return any(phrase in q for phrase in FOLLOWUP_PHRASES)

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

def build_product_text(p):
    return f"""
ID: {p.get('id','')}
Nazwa: {p.get('title','')}
Marka: {p.get('brand','')}
MPN: {p.get('mpn','')}
GTIN: {p.get('gtin','')}
Dostępność: {p.get('availability','')}
Cena: {p.get('price','')}
Kategoria: {p.get('category','')}
Typ produktu: {p.get('product_type','')}
Wariant produktu: {product_variant_label(p)}
Kolor: {p.get('color','')}
Typ montażu / wariant: {p.get('mount_type','')}
Materiał: {p.get('material','')}
Wykończenie: {p.get('finish','')}
Wymiary: {p.get('dimensions','')}
URL: {p.get('link','')}
Zdjęcie: {p.get('image','')}
Opis: {p.get('description','')}
""".strip()

def infer_product_type(title):
    t = normalize_polish_text(title)
    if "wymienna szczotka" in t:
        return "replacement_brush"
    if "stojak" in t and "szczotka" in t:
        return "toilet_brush_stand"
    if "szczotka" in t and "wc" in t:
        return "toilet_brush"
    return "other"

def _lxml_text(element, tag, ns=None):
    if ns:
        tag = f"{{{ns}}}{tag}"
    found = element.find(tag)
    if found is not None and found.text:
        return found.text.strip()
    return ""

G = "http://base.google.com/ns/1.0"

def load_products_from_xml(xml_path="products.xml"):
    parser = etree.XMLParser(recover=True, strip_cdata=False)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()
    products = []

    for i, item in enumerate(root.iter("item"), start=1):
        title        = _lxml_text(item, "title", G)
        link         = _lxml_text(item, "link", G)
        description  = _lxml_text(item, "description", G)
        brand        = _lxml_text(item, "brand", G)
        mpn          = _lxml_text(item, "mpn", G)
        gtin         = _lxml_text(item, "gtin", G)
        availability = _lxml_text(item, "availability", G)
        price        = _lxml_text(item, "price", G)
        image        = _lxml_text(item, "image_link", G)
        category     = _lxml_text(item, "google_product_category", G)

        if not link:
            link = _lxml_text(item, "link")
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = atom_link.get("href", "")

        color = ""
        title_norm = normalize_polish_text(title)
        if "czarn" in title_norm:
            color = "czarny"
        elif "bial" in title_norm:
            color = "biały"
        elif "bezow" in title_norm or "bezowa" in title_norm:
            color = "beżowy"

        p = {
            "id": i,
            "title": title,
            "brand": brand,
            "mpn": mpn,
            "gtin": gtin,
            "availability": availability,
            "price": price,
            "category": category,
            "color": color,
            "mount_type": "",
            "material": "",
            "finish": "",
            "dimensions": "",
            "link": link,
            "image": image,
            "description": description,
            "product_type": infer_product_type(title),
        }
        p["text"] = build_product_text(p)
        products.append(p)

    first_title = products[0]['title'] if products else 'brak'
    print(f"[catalog] Załadowano {len(products)} produktów. Przykład: '{first_title}'")
    return products

def build_search_index():
    global VECTORIZER, DOC_MATRIX
    corpus = [p["text"] for p in PRODUCTS]
    VECTORIZER = TfidfVectorizer()
    DOC_MATRIX = VECTORIZER.fit_transform(corpus)

def search_products(query, top_k=3):
    query_vec = VECTORIZER.transform([query])
    sims = cosine_similarity(query_vec, DOC_MATRIX).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]
    return [PRODUCTS[i] for i in ranked_idx]

def fetch_product_page_fields(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        tree = html.fromstring(r.text)
        text = " ".join(tree.xpath("//body//text()"))
        text = re.sub(r"\s+", " ", text).strip()
        text_norm = normalize_polish_text(text)

        material = []
        if "stal nierdzewna" in text_norm:
            material.append("stal nierdzewna")
        if "tworzywo sztuczne" in text_norm:
            material.append("tworzywo sztuczne")

        has_wall  = any(x in text_norm for x in ["wiszace", "wiszacy", "wiszaca"])
        has_floor = any(x in text_norm for x in ["wolnostojace", "wolnostojacy", "wolnostojaca"])

        mount_type = ""
        if has_wall and has_floor:
            mount_type = "wiszący / wolnostojący"
        elif has_wall:
            mount_type = "wiszący"
        elif has_floor:
            mount_type = "wolnostojący"

        finish = []
        if "malowana proszkowo" in text_norm:
            finish.append("powłoka malowana proszkowo")
        if "polmat" in text_norm:
            finish.append("półmat")
        elif any(x in text_norm for x in ["matowa", "matowe", "matowy"]):
            finish.append("mat")

        return {
            "material":   ", ".join(dict.fromkeys(material)),
            "mount_type": mount_type,
            "finish":     ", ".join(dict.fromkeys(finish)),
            "dimensions": ""
        }
    except Exception:
        return {"material": "", "mount_type": "", "finish": "", "dimensions": ""}

def enrich_one_product_from_page(p):
    if not p.get("link"):
        p["text"] = build_product_text(p)
        return p

    needs_more = not p.get("material") or not p.get("mount_type") or not p.get("finish")
    if needs_more:
        extra = fetch_product_page_fields(p["link"])
        if extra.get("material")   and not p.get("material"):   p["material"]   = extra["material"]
        if extra.get("mount_type") and not p.get("mount_type"): p["mount_type"] = extra["mount_type"]
        if extra.get("finish")     and not p.get("finish"):     p["finish"]     = extra["finish"]

    p["text"] = build_product_text(p)
    return p

def enrich_search_results(results):
    enriched = []
    for p in results:
        enriched.append(enrich_one_product_from_page(p))
        time.sleep(0.2)
    return enriched

def reset_session(session_id):
    SESSIONS[session_id] = {"chat_memory": []}

def get_session(session_id):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"chat_memory": []}
    return SESSIONS[session_id]

def chat_with_session(session_id, question, top_k=3, retries=3, memory_turns=2):
    session = get_session(session_id)
    chat_memory = session["chat_memory"]

    history_text = ""
    if chat_memory:
        recent = chat_memory[-(memory_turns * 2):]
        history_text = "\n\nHistoria rozmowy:\n" + "\n".join(
            [f"{m['role'].upper()}: {m['text']}" for m in recent]
        )

    retrieval_query = question
    if chat_memory and is_followup_question(question):
        last_user_questions = [m["text"] for m in chat_memory if m["role"] == "user"]
        if last_user_questions:
            retrieval_query = f"{last_user_questions[-1]} {question}"

    matches = search_products(retrieval_query, top_k=top_k)
    matches = enrich_search_results(matches)
    context = "\n\n".join([p["text"] for p in matches]) if matches else "Brak pasujących produktów."

    user_message = f"""
{history_text}

Aktualne pytanie użytkownika:
{question}

Dane katalogowe:
{context}
""".strip()

    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user",   "content": user_message}
                ],
                temperature=0.2,
                max_tokens=400,
            )
            answer = response.choices[0].message.content
            chat_memory.append({"role": "user",      "text": question})
            chat_memory.append({"role": "assistant", "text": answer})
            session["chat_memory"] = chat_memory[-(memory_turns * 2):]
            return answer
        except Exception as e:
            last_error = e
            msg = str(e)
            if "503" in msg or "rate_limit" in msg.lower() or "429" in msg:
                time.sleep(2 * (attempt + 1))
                continue
            break

    return f"Wystąpił błąd po stronie modelu: {last_error}"

def init_catalog():
    global PRODUCTS
    PRODUCTS = load_products_from_xml("products.xml")
    build_search_index()
