import os
import re
import datetime as dt
import streamlit as st
from openai import OpenAI
import psycopg2

# ---------------------------
# Config / Secrets
# ---------------------------
st.set_page_config(page_title="Base de Conocimiento IA", layout="wide")

def get_secret(name: str) -> str:
    v = st.secrets.get(name) or os.getenv(name)
    if not v:
        st.error(f"Falta el secreto: {name}. Ve a Settings -> Secrets en Streamlit y añádelo.")
        st.stop()
    return v

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
DB_HOST = get_secret("DB_HOST")
DB_PORT = int(st.secrets.get("DB_PORT", 5432))
DB_NAME = get_secret("DB_NAME")
DB_USER = get_secret("DB_USER")
DB_PASSWORD = get_secret("DB_PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Helpers
# ---------------------------
def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
    )

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, max_chars: int = 2000):
    """
    Troceo simple por longitud (para principiantes).
    2000 chars suele dar chunks manejables.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars
    return chunks

def embed_text(text: str):
    """
    Modelo de embeddings con dimensión 1536 (compatible con vector(1536)).
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def insert_chunks(titulo, tipo, fecha_documento, fuente, chunks):
    with connect_db() as conn:
        with conn.cursor() as cur:
            for ch in chunks:
                emb = embed_text(ch)
                cur.execute(
                    """
                    insert into documentos (titulo, tipo, fecha_documento, fuente, contenido, embedding)
                    values (%s, %s, %s, %s, %s, %s)
                    """,
                    (titulo, tipo, fecha_documento, fuente, ch, emb)
                )
        conn.commit()

def semantic_search(query: str, top_k: int = 6, filtro_tipo: str = "", desde: dt.date | None = None):
    q_emb = embed_text(query)

    where = []
    params = []

    if filtro_tipo:
        where.append("tipo = %s")
        params.append(filtro_tipo)

    if desde:
        where.append("fecha_documento >= %s")
        params.append(desde)

    where_sql = ("where " + " and ".join(where)) if where else ""

    sql = f"""
        select id, titulo, tipo, fecha_documento, fuente, contenido,
               (embedding <-> %s::vector) as distancia
        from documentos
        {where_sql}
        order by embedding <-> %s::vector
        limit {top_k}
    """

    # embedding se pasa dos veces por el order by y por el select
    params_final = [q_emb] + params + [q_emb]

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params_final)
            rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "titulo": r[1],
            "tipo": r[2],
            "fecha_documento": r[3],
            "fuente": r[4],
            "contenido": r[5],
            "distancia": r[6]
        })
    return results

def answer_with_citations(question: str, contexts):
    """
    Construye respuesta usando SOLO el contexto recuperado, con citas.
    """
    if not contexts:
        return "No he encontrado información suficiente en tu base para contestar a esa pregunta."

    # Creamos un "dossier" numerado para citar
    dossier_lines = []
    for i, c in enumerate(contexts, start=1):
        meta = f"[{i}] {c['titulo']} | {c['tipo']} | {c['fecha_documento']} | {c['fuente']}"
        dossier_lines.append(meta)
        dossier_lines.append(c["contenido"])
        dossier_lines.append("")

    dossier = "\n".join(dossier_lines)

    system = (
        "Eres un asistente que responde SOLO con la información del dossier aportado. "
        "Si el dossier no contiene evidencia suficiente, dilo claramente. "
        "Incluye citas entre corchetes usando los números [1], [2], etc. "
        "No inventes datos ni añadas conocimiento externo."
    )

    user = f"""PREGUNTA:
{question}

DOSSIER (fuentes):
{dossier}

INSTRUCCIONES:
- Responde en español.
- Estructura la respuesta en viñetas o apartados si ayuda.
- Cita siempre que afirmes algo (por ejemplo: ... [1]).
- Si hay contradicciones entre fuentes, indícalo con citas.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content

# ---------------------------
# UI
# ---------------------------
st.title("Base de Conocimiento IA (RAG)")

tab1, tab2 = st.tabs(["1) Añadir contenido", "2) Consultar"])

with tab1:
    st.subheader("Añadir documentos / noticias (pegando texto)")

    col1, col2 = st.columns(2)
    with col1:
        titulo = st.text_input("Título", placeholder="Ej.: Informe C-UAS 2025 / Noticia sobre drones en estadio...")
        tipo = st.selectbox("Tipo", ["noticia", "normativa", "informe", "guia", "jurisprudencia", "otro"])
    with col2:
        fecha_documento = st.date_input("Fecha del documento", value=dt.date.today())
        fuente = st.text_input("Fuente / URL (opcional)", placeholder="https://...")

    texto = st.text_area("Pega aquí el texto completo", height=250)

    if st.button("Guardar en la base"):
        if not titulo or not texto.strip():
            st.warning("Falta el título o el texto.")
        else:
            chunks = chunk_text(texto, max_chars=2000)
            insert_chunks(titulo, tipo, fecha_documento, fuente, chunks)
            st.success(f"Guardado: {len(chunks)} fragmentos indexados.")

    st.caption("Consejo: para noticias web, pega el texto y guarda también la URL en 'Fuente'.")

with tab2:
    st.subheader("Consulta con IA (responde con citas)")

    pregunta = st.text_input("Tu pregunta", placeholder="Ej.: ¿Qué tendencias se describen sobre drones en eventos deportivos en Europa?")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        filtro_tipo = st.selectbox("Filtrar por tipo (opcional)", ["", "noticia", "normativa", "informe", "guia", "jurisprudencia", "otro"])
    with colf2:
        usar_fecha = st.checkbox("Filtrar desde fecha", value=False)
    with colf3:
        desde = st.date_input("Desde", value=dt.date.today() - dt.timedelta(days=365), disabled=not usar_fecha)

    if st.button("Buscar y responder"):
        if not pregunta.strip():
            st.warning("Escribe una pregunta.")
        else:
            results = semantic_search(
                query=pregunta,
                top_k=6,
                filtro_tipo=filtro_tipo,
                desde=(desde if usar_fecha else None)
            )
            respuesta = answer_with_citations(pregunta, results)

            st.markdown("### Respuesta")
            st.write(respuesta)

            with st.expander("Ver fuentes recuperadas (fragmentos)"):
                for i, r in enumerate(results, start=1):
                    st.markdown(f"**[{i}] {r['titulo']}** — {r['tipo']} — {r['fecha_documento']}  \nFuente: {r['fuente']}")
                    st.write(r["contenido"])
                    st.divider()
