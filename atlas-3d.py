#!/usr/bin/env python3
"""ATLAS Geometry Chat — natural language in, geometry-grounded answer out.

Install:
    pip install streamlit numpy pandas plotly scikit-learn sentence-transformers nltk

Optional local response model (recommended):
    Install Ollama, then: ollama pull qwen2.5:7b

Run:
    streamlit run atlas_geometry_chat.py

The user's message is preserved as an utterance. ATLAS automatically derives
sentences, content tokens, and salient phrases; users never comma-separate it.
The response engine receives a compact representation of the current geometry
plus recent conversation history before producing its answer.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.manifold import MDS


APP_TITLE = "ATLAS Geometry Chat"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_AXES = {
    "AFFECT_VALENCE": {"positive": ["love", "joy", "benefit", "kindness", "hope"], "negative": ["hate", "grief", "harm", "cruelty", "despair"]},
    "EPI_CERTAINTY": {"positive": ["certain", "known", "verified", "evidence", "confidence"], "negative": ["uncertain", "unknown", "doubt", "ambiguous", "possibly"]},
    "ABSTRACTION": {"positive": ["concept", "principle", "meaning", "theory", "abstraction"], "negative": ["stone", "table", "hammer", "body", "object"]},
    "AGENCY": {"positive": ["act", "choose", "control", "cause", "decide"], "negative": ["passive", "receive", "undergo", "constraint", "inert"]},
    "SELF_OTHER": {"positive": ["self", "myself", "identity", "internal", "own"], "negative": ["other", "you", "external", "stranger", "they"]},
    "TEMPORAL_DIRECTION": {"positive": ["future", "after", "later", "prediction", "becoming"], "negative": ["past", "before", "earlier", "memory", "history"]},
    "SOCIAL_ORIENTATION": {"positive": ["collective", "community", "cooperate", "shared", "together"], "negative": ["individual", "solitary", "private", "independent", "alone"]},
    "CAUSAL_DIRECTION": {"positive": ["cause", "condition", "trigger", "source", "produce"], "negative": ["effect", "outcome", "consequence", "result", "response"]},
    "LOG_POLARITY": {"positive": ["true", "valid", "consistent", "affirm", "yes"], "negative": ["false", "invalid", "contradiction", "negate", "no"]},
    "UNCERTAINTY": {"positive": ["uncertain", "ambiguous", "variable", "unresolved", "unknown"], "negative": ["stable", "precise", "resolved", "definite", "fixed"]},
}

STOPWORDS = set("""a an and are as at be been being but by can could did do does doing for from had has have having he her hers him his how i if in into is it its may me might mine must my no nor not of on or our ours she should so than that the their theirs them then there these they this those through to too under until up us very was we were what when where which who why will with would you your yours""".split())
SYSTEM_PROMPT = """You are the response model inside ATLAS, a geometry-grounded conversational analyst.
Answer the user's latest message naturally and directly. Use the supplied typed
geometry as evidence, not as unquestionable truth. Distinguish observed text,
derived coordinates, lexical evidence, and inference. Mention geometry only
when useful. Preserve ambiguity and uncertainty. Never claim that projection
coordinates prove meaning, intent, truth, emotion, or causation."""

REASONING_PROMPT = """Analyze the user's utterance against the ATLAS reasoning
packet. Return concise JSON with keys: interpretations (at least 1), geometry_evidence,
counterevidence, ambiguities, projection_risks, warranted_conclusions,
unwarranted_conclusions, and answer_plan. Compare native typed distances with
3D projection distances. Treat coordinates as derived observations, not facts
about intent or truth. Do not answer the user yet."""

CRITIC_PROMPT = """Audit the proposed ATLAS analysis against the reasoning packet.
Identify unsupported leaps, contradictions, ignored uncertainty, misuse of a
lossy projection, and missed alternative interpretations. Return concise JSON
with keys valid, corrections, preserved_claims, and final_constraints."""


@dataclass
class LexicalEdge:
    source: str
    target: str
    relationship: str
    evidence: str
    provenance: str = "WORDNET"
    typed_distance: float | None = None
    projected_distance: float | None = None
    projection_residual: float | None = None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed_texts(texts: list[str], model_name: str, offline: bool, dims: int):
    if not offline:
        try:
            model = load_sentence_transformer(model_name)
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return normalize_rows(np.asarray(vectors, dtype=float)), model_name, "LEARNED_EMBEDDING"
        except Exception as exc:
            st.warning(f"Embedding model unavailable; using hashing proxy: {exc}")
    vectorizer = HashingVectorizer(n_features=dims, analyzer="char_wb", ngram_range=(2, 5), alternate_sign=False, norm="l2")
    return normalize_rows(vectorizer.transform(texts).toarray()), f"offline-hashed-{dims}d", "DERIVED_HASH_PROXY"


def parse_axes(raw: str) -> dict[str, dict[str, list[str]]]:
    value = json.loads(raw)
    clean = {}
    for name, poles in value.items():
        positive = [str(x).strip() for x in poles.get("positive", []) if str(x).strip()]
        negative = [str(x).strip() for x in poles.get("negative", []) if str(x).strip()]
        if positive and negative:
            clean[str(name)] = {"positive": positive, "negative": negative}
    if len(clean) < 3:
        raise ValueError("At least three complete axes are required.")
    return clean


def derive_carriers(utterance: str, max_carriers: int) -> tuple[list[str], dict[str, str]]:
    """Preserve the utterance and automatically derive analysis carriers."""
    clean = re.sub(r"\s+", " ", utterance).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|[\n]+", clean) if s.strip()]
    tokens = re.findall(r"[^\W\d_][\w'’-]*", clean.casefold(), flags=re.UNICODE)
    content = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    frequencies = {t: content.count(t) for t in set(content)}
    first_pos = {t: content.index(t) for t in frequencies}
    keywords = sorted(frequencies, key=lambda t: (-frequencies[t], first_pos[t]))
    phrases = []
    for n in (3, 2):
        for i in range(len(tokens) - n + 1):
            part = tokens[i:i+n]
            if sum(t not in STOPWORDS for t in part) >= 2:
                phrases.append(" ".join(part))
    carriers, roles = [clean], {clean: "UTTERANCE"}
    for item, role in [(x, "SENTENCE") for x in sentences if x != clean] + [(x, "PHRASE") for x in phrases] + [(x, "CONTENT_TOKEN") for x in keywords]:
        if item and item not in roles and len(carriers) < max_carriers:
            carriers.append(item); roles[item] = role
    # A tiny input still needs enough points for a meaningful projection.
    if len(carriers) == 1:
        for token in dict.fromkeys(tokens):
            if token not in roles:
                carriers.append(token); roles[token] = "TOKEN"
    return carriers, roles


@st.cache_data(show_spinner=False)
def wordnet_edges(seed_words: tuple[str, ...], limit: int):
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("word")
    except Exception:
        return [], "WordNet unavailable. Run: python -m nltk.downloader wordnet omw-1.4"
    rows, seen = [], set()
    for source in seed_words:
        if " " in source:
            continue
        for syn in wn.synsets(source)[:6]:
            candidates = [(x.name(), "SYNONYM", syn.definition()) for x in syn.lemmas()[:5]]
            candidates += [(x.name(), "ANTONYM", "WordNet antonym") for l in syn.lemmas()[:4] for x in l.antonyms()[:2]]
            candidates += [(x, "HYPERNYM", h.definition()) for h in syn.hypernyms()[:3] for x in h.lemma_names()[:2]]
            candidates += [(x, "HYPONYM", h.definition()) for h in syn.hyponyms()[:3] for x in h.lemma_names()[:1]]
            for target, relation, evidence in candidates:
                target = target.replace("_", " ")
                key = (source.casefold(), target.casefold(), relation)
                if target.casefold() != source.casefold() and key not in seen:
                    seen.add(key); rows.append(asdict(LexicalEdge(source, target, relation, evidence)))
        source_rows = [r for r in rows if r["source"] == source][:limit]
        rows = [r for r in rows if r["source"] != source] + source_rows
    return rows, ""


def build_axis_coordinates(carriers, axes, model, offline, dims):
    anchors = [x for p in axes.values() for side in ("positive", "negative") for x in p[side]]
    texts = list(dict.fromkeys(carriers + anchors))
    vectors, used_model, status = embed_texts(texts, model, offline, dims)
    by_text = {t: vectors[i] for i, t in enumerate(texts)}
    pole_vectors = {}
    for name, poles in axes.items():
        pole_vectors[name] = (
            normalize_rows(np.mean([by_text[x] for x in poles["positive"]], axis=0)[None])[0],
            normalize_rows(np.mean([by_text[x] for x in poles["negative"]], axis=0)[None])[0],
        )
    coordinates = np.asarray([[float(np.dot(by_text[c], p) - np.dot(by_text[c], n)) for p, n in pole_vectors.values()] for c in carriers])
    return coordinates, used_model, status


def scale_coordinates(x):
    std = np.std(x, axis=0); std[std < 1e-9] = 1.0
    return (x - np.mean(x, axis=0)) / std


def distance_matrix(x, metric, weights):
    z = x * np.sqrt(weights)[None, :]
    if metric == "Cosine":
        q = normalize_rows(z); return np.clip(1.0 - q @ q.T, 0.0, 2.0)
    n = len(z); out = np.zeros((n, n))
    inverse = None
    if metric == "Mahalanobis":
        inverse = np.linalg.pinv(np.atleast_2d(np.cov(z, rowvar=False)) + np.eye(z.shape[1]) * 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            delta = z[i] - z[j]
            value = math.sqrt(max(0.0, float(delta @ inverse @ delta))) if inverse is not None else float(np.linalg.norm(delta))
            out[i, j] = out[j, i] = value
    return out


def project_3d(x, distances, method, selected_axes, seed):
    n = len(x)
    if method == "Explicit typed axes": return x[:, selected_axes[:3]], "DIRECT_AXIS_VIEW"
    if n < 4:
        out = np.zeros((n, 3)); out[:, :min(3, x.shape[1])] = x[:, :min(3, x.shape[1])]
        return out, "PADDED_DIRECT_VIEW"
    if method == "PCA": return PCA(n_components=3, random_state=seed).fit_transform(x), "PCA"
    if method == "UMAP":
        try:
            import umap
            return umap.UMAP(n_components=3, metric="precomputed", random_state=seed).fit_transform(distances), "UMAP_PRECOMPUTED"
        except Exception as exc: st.warning(f"UMAP unavailable; using MDS: {exc}")
    return MDS(n_components=3, dissimilarity="precomputed", random_state=seed, n_init=4, max_iter=500).fit_transform(distances), "METRIC_MDS"


def projection_diagnostics(points, native):
    raw = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    mask = np.triu(np.ones_like(raw, dtype=bool), 1)
    a, b = raw[mask], native[mask]
    scale = float((a @ b) / max(a @ a, 1e-12)) if a.size else 1.0
    aligned = raw * scale; residual = np.abs(aligned - native)
    stress = math.sqrt(float(np.sum((aligned-native)**2)) / max(float(np.sum(native**2)), 1e-12))
    return aligned, residual, scale, stress


def differential_geometry(x, labels, axis_names, k=6):
    n, d = x.shape; k = max(1, min(k, max(1, n-1)))
    eu = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2); np.fill_diagonal(eu, np.inf)
    nbrs = np.argsort(eu, axis=1)[:, :k]; finite = eu[np.isfinite(eu)]
    sigma = max(float(np.median(finite)) if finite.size else 1.0, 1e-9); W = np.zeros((n, n))
    tangents, metrics = [], []
    for i, ids in enumerate(nbrs):
        delta = x[ids] - x[i]
        for j in ids:
            W[i, j] = W[j, i] = max(W[i, j], math.exp(-(float(np.linalg.norm(x[i]-x[j]))**2)/(2*sigma*sigma)))
        _, sv, vt = np.linalg.svd(delta, full_matrices=False)
        rank = int(np.sum(sv > 1e-8)); basis = vt[:min(rank, 3)]
        g = np.linalg.pinv(delta.T @ delta / max(len(ids), 1) + np.eye(d)*1e-6)
        tangents.append({"carrier": labels[i], "neighbors": [labels[j] for j in ids], "local_rank": rank, "tangent_basis": basis.tolist()})
        metrics.append({"carrier": labels[i], "metric_tensor": g.tolist()})
    L = np.diag(W.sum(axis=1)) - W
    G = np.full((n, n), np.inf); np.fill_diagonal(G, 0)
    for i in range(n):
        for j in np.where(W[i] > 0)[0]: G[i, j] = float(np.linalg.norm(x[i]-x[j]))
    for m in range(n): G = np.minimum(G, G[:, m, None] + G[None, m, :])
    displacements = [{"source": labels[i], "target": labels[j], "distance": float(np.linalg.norm(x[j]-x[i])), "delta": {axis_names[a]: float(x[j,a]-x[i,a]) for a in range(d)}} for i in range(n) for j in range(i+1,n)]
    return {"knn_k": k, "kernel_sigma": sigma, "adjacency": W.tolist(), "graph_laplacian": L.tolist(), "geodesic_distance_matrix": G.tolist(), "tangent_spaces": tangents, "local_metric_tensors": metrics, "pairwise_displacements": displacements}


def construct_atlas(utterance, axes, settings, axis_weights):
    seeds, roles = derive_carriers(utterance, settings["max_carriers"])
    edge_rows, resource_note = wordnet_edges(tuple(x for x in seeds if roles[x] == "CONTENT_TOKEN"), settings["neighbor_limit"]) if settings["expand"] else ([], "")
    carriers = list(dict.fromkeys(seeds + [e["target"] for e in edge_rows]))
    for item in carriers:
        roles.setdefault(item, "LEXICAL_NEIGHBOR")
    raw, used_model, status = build_axis_coordinates(carriers, axes, settings["model"], settings["offline"], settings["hash_dims"])
    working = scale_coordinates(raw) if settings["standardize"] else raw.copy()
    distances = distance_matrix(working, settings["metric"], np.asarray(axis_weights))
    points, projection_used = project_3d(working, distances, settings["projection"], settings["selected_axes"], settings["seed"])
    projected, residuals, scale, stress = projection_diagnostics(points, distances)
    index = {x:i for i,x in enumerate(carriers)}
    for e in edge_rows:
        i,j=index[e["source"]],index[e["target"]]
        e["typed_distance"]=round(float(distances[i,j]),6); e["projected_distance"]=round(float(projected[i,j]),6); e["projection_residual"]=round(float(residuals[i,j]),6)
    rows=[]
    for i,c in enumerate(carriers):
        row={"carrier":c,"role":roles[c], **{a:round(float(raw[i,j]),6) for j,a in enumerate(axes)}}
        row.update({"projection_x":float(points[i,0]),"projection_y":float(points[i,1]),"projection_z":float(points[i,2])}); rows.append(row)
    return {"schema":"ATLAS_UTTERANCE_GEOMETRY_CHAT_v2_REASONING","utterance":utterance,"carriers":carriers,"derived_seeds":seeds,"roles":roles,"axes":axes,"axis_weights":dict(zip(axes,axis_weights)),"metric":settings["metric"],"geometry_coordinates":working.tolist(),"coordinates_used":"standardized" if settings["standardize"] else "raw","projection_used":projection_used,"projection_scale":scale,"normalized_stress":stress,"embedding_model":used_model,"observation_status":status,"coordinates":rows,"relations":edge_rows,"resource_note":resource_note,"epistemic_contract":{"utterance":"OBSERVED","carriers":"DERIVED","axis_coordinates":status,"lexical_relations":"WORDNET_EVIDENCE","projection":"LOSSY_VIEW_NOT_INTRINSIC_MEANING"},"differential_geometry":differential_geometry(working,carriers,list(axes),min(6,max(1,len(carriers)-1)))}


def compact_context(result):
    df=pd.DataFrame(result["coordinates"]); axes=list(result["axes"])
    utterance_row=df[df.role=="UTTERANCE"].iloc[0]
    strongest=sorted(axes,key=lambda a:abs(float(utterance_row[a])),reverse=True)[:5]
    profile={a:round(float(utterance_row[a]),4) for a in strongest}
    nearby=[]
    if len(df)>1:
        p=df[["projection_x","projection_y","projection_z"]].to_numpy(); d=np.linalg.norm(p-p[0],axis=1)
        nearby=[{"carrier":df.iloc[i].carrier,"role":df.iloc[i].role,"projected_distance":round(float(d[i]),4)} for i in np.argsort(d)[1:7]]
    return {"utterance":result["utterance"],"utterance_axis_profile":profile,"nearby_derived_carriers":nearby,"projection":result["projection_used"],"normalized_stress":round(result["normalized_stress"],4),"observation_status":result["observation_status"],"epistemic_contract":result["epistemic_contract"]}


def atlas_reasoning_packet(result, detail="Balanced"):
    """Create a bounded but geometry-rich packet for an Ollama context window."""
    df = pd.DataFrame(result["coordinates"])
    axes = list(result["axes"])
    native = np.asarray(result.get("geometry_coordinates", df[axes].to_numpy(dtype=float)), dtype=float)
    projection = df[["projection_x", "projection_y", "projection_z"]].to_numpy(dtype=float)
    weights = np.asarray([result["axis_weights"][a] for a in axes], dtype=float)
    native_dist = distance_matrix(native, result["metric"], weights)
    projected_dist = np.linalg.norm(projection[:, None, :] - projection[None, :, :], axis=2)
    pair_rows = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            pair_rows.append({
                "source": df.iloc[i].carrier, "target": df.iloc[j].carrier,
                "native_distance": round(float(native_dist[i, j]), 5),
                "projected_distance_raw": round(float(projected_dist[i, j]), 5),
                "projection_error_indicator": round(abs(float(native_dist[i, j] - projected_dist[i, j])), 5),
            })
    pair_rows.sort(key=lambda x: x["native_distance"])
    max_nodes = {"Compact": 10, "Balanced": 20, "Full bounded": 40}[detail]
    max_pairs = {"Compact": 15, "Balanced": 45, "Full bounded": 120}[detail]
    coordinate_rows = []
    for _, row in df.head(max_nodes).iterrows():
        ordered = sorted(axes, key=lambda a: abs(float(row[a])), reverse=True)
        coordinate_rows.append({
            "carrier": row.carrier, "role": row.role,
            "axis_coordinates": {a: round(float(row[a]), 5) for a in ordered},
            "projection_3d": [round(float(row.projection_x), 5), round(float(row.projection_y), 5), round(float(row.projection_z), 5)],
        })
    dg = result["differential_geometry"]
    tangent_summary = [{"carrier": x["carrier"], "neighbors": x["neighbors"], "local_rank": x["local_rank"]} for x in dg["tangent_spaces"][:max_nodes]]
    displacement_rows = sorted(dg["pairwise_displacements"], key=lambda x: x["distance"])[:max_pairs]
    lexical = result["relations"][:max_pairs]
    return {
        "schema": "ATLAS_REASONING_PACKET_v2",
        "status_contract": result["epistemic_contract"],
        "observed_utterance": result["utterance"],
        "geometry_definition": {"axes": axes, "axis_weights": result["axis_weights"], "metric": result["metric"], "coordinates_used": result.get("coordinates_used", "raw"), "embedding_model": result["embedding_model"], "observation_status": result["observation_status"]},
        "projection_diagnostics": {"method": result["projection_used"], "normalized_stress": round(result["normalized_stress"], 6), "warning": "3D is a lossy view; reason primarily from typed coordinates and native distances."},
        "carrier_states": coordinate_rows,
        "nearest_native_pairs": pair_rows[:max_pairs],
        "local_tangent_summary": tangent_summary,
        "nearest_displacements": displacement_rows,
        "lexical_evidence": lexical,
        "resource_note": result["resource_note"],
    }


def ollama_chat(model, endpoint, messages, temperature, json_mode=False):
    payload={"model":model,"stream":False,"messages":messages,"options":{"temperature":temperature}}
    if json_mode: payload["format"] = "json"
    req=urllib.request.Request(endpoint.rstrip("/")+"/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=180) as response:
        return json.loads(response.read().decode())["message"]["content"].strip()


def geometry_grounded_reply(model, endpoint, history, result, temperature, detail, critic_enabled):
    """Two/three-pass inference: analyze geometry, audit it, then answer."""
    packet = atlas_reasoning_packet(result, detail)
    packet_text = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    latest = result["utterance"]
    analysis = ollama_chat(model, endpoint, [
        {"role": "system", "content": REASONING_PROMPT},
        {"role": "user", "content": "ATLAS PACKET:\n" + packet_text + "\n\nUSER UTTERANCE:\n" + latest},
    ], 0.1, json_mode=True)
    critique = "{\"valid\":true,\"corrections\":[],\"final_constraints\":[]}"
    if critic_enabled:
        critique = ollama_chat(model, endpoint, [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": "ATLAS PACKET:\n" + packet_text + "\n\nPROPOSED ANALYSIS:\n" + analysis},
        ], 0.0, json_mode=True)
    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "ATLAS REASONING PACKET:\n" + packet_text},
        {"role": "system", "content": "INTERNAL GEOMETRY ANALYSIS:\n" + analysis},
        {"role": "system", "content": "EVIDENCE AUDIT:\n" + critique},
    ] + history
    answer = ollama_chat(model, endpoint, final_messages, temperature)
    trace = {"packet": packet, "analysis": analysis, "critique": critique, "pipeline": "ATLAS_PACKET -> QWEN_ANALYSIS -> QWEN_CRITIC -> QWEN_RESPONSE"}
    return answer, trace


def atlas_fallback(result):
    ctx=compact_context(result); profile=ctx["utterance_axis_profile"]
    high=", ".join(f"{k}={v:+.3f}" for k,v in profile.items())
    carriers=", ".join(x["carrier"] for x in ctx["nearby_derived_carriers"][:4]) or "no additional carriers"
    return (f"I read this as: {result['utterance']}\n\nATLAS derives its strongest local axis signals as {high}. "
            f"The nearest derived carriers in the 3D view are {carriers}. This is a structural reading, not proof of the speaker’s intent or the statement’s truth. "
            "Connect Ollama in the sidebar for a fully generative, geometry-grounded conversational reply.")


def geometry_figure(result):
    df=pd.DataFrame(result["coordinates"]); points=df[["projection_x","projection_y","projection_z"]].to_numpy()
    colors={"UTTERANCE":"#ff5c4d","SENTENCE":"#ffb347","PHRASE":"#a78bfa","CONTENT_TOKEN":"#29dfc1","TOKEN":"#29dfc1","LEXICAL_NEIGHBOR":"#60a5fa"}
    fig=go.Figure(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode="markers+text",text=df.carrier,textposition="top center",customdata=df.role,hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",marker=dict(size=[11 if r=="UTTERANCE" else 7 for r in df.role],color=[colors.get(r,"#fff") for r in df.role],line=dict(width=.5,color="#d8e6ff"))))
    for edge in result["relations"]:
        a=df.index[df.carrier==edge["source"]]; b=df.index[df.carrier==edge["target"]]
        if len(a) and len(b):
            p,q=points[a[0]],points[b[0]]; fig.add_trace(go.Scatter3d(x=[p[0],q[0]],y=[p[1],q[1]],z=[p[2],q[2]],mode="lines",line=dict(width=2,color="rgba(99,102,241,.3)"),hoverinfo="skip",showlegend=False))
    fig.update_layout(template="plotly_dark",height=700,title="Current utterance geometry",scene=dict(aspectmode="data"),margin=dict(l=0,r=0,t=45,b=0),showlegend=False)
    return fig


st.set_page_config(page_title=APP_TITLE,layout="wide")
st.title(APP_TITLE)
st.caption("Write normally. ATLAS derives the carriers, constructs typed geometry, reads the result, and replies.")

with st.sidebar:
    st.header("Response engine")
    response_engine=st.selectbox("Engine",["Ollama local LLM","ATLAS deterministic fallback"])
    ollama_model=st.text_input("Ollama model","qwen2.5:7b")
    ollama_endpoint=st.text_input("Ollama endpoint","http://localhost:11434")
    temperature=st.slider("Response temperature",0.0,1.5,0.35,0.05)
    reasoning_detail=st.selectbox("Geometry reasoning detail",["Compact","Balanced","Full bounded"],index=1)
    critic_enabled=st.toggle("Audit reasoning before answering",True)
    st.divider(); st.header("Carrier derivation")
    max_carriers=st.slider("Maximum automatic carriers",4,40,18)
    expand=st.toggle("Add WordNet evidence",True); neighbor_limit=st.slider("WordNet neighbors per token",1,12,4)
    st.divider(); st.header("Geometry")
    metric=st.selectbox("Native metric",["Weighted Euclidean","Cosine","Mahalanobis"])
    projection=st.selectbox("3D projection",["Metric MDS","PCA","Explicit typed axes","UMAP"])
    standardize=st.toggle("Standardize typed axes",True); seed=int(st.number_input("Random seed",0,999999,42))
    offline=st.toggle("Offline hashing proxy",False); model_name=st.text_input("Embedding model",DEFAULT_MODEL); hash_dims=st.slider("Hash dimensions",64,1024,384,64)
    history_turns=st.slider("Conversation turns sent to LLM",1,12,6)
    if st.button("Clear conversation",width="stretch"):
        st.session_state.messages=[]; st.session_state.atlas_results=[]; st.session_state.reasoning_traces=[]; st.rerun()

with st.expander("Typed-axis registry and geometry controls",expanded=False):
    axes_json=st.text_area("Axis poles (JSON)",json.dumps(DEFAULT_AXES,indent=2),height=420)
    try: axes=parse_axes(axes_json)
    except Exception as exc: st.error(f"Axis registry error: {exc}"); st.stop()
    axis_names=list(axes); cols=st.columns(min(5,len(axis_names))); axis_weights=[]
    for i,a in enumerate(axis_names):
        with cols[i%len(cols)]:
            axis_weights.append(st.slider(a,0.0,3.0,1.0,.05,key=f"weight::{a}"))
    selected_axes=[0,1,2]
    if projection=="Explicit typed axes":
        c=st.columns(3); selected_axes=[]
        for i,label in enumerate(("X axis","Y axis","Z axis")):
            with c[i]:
                selected_axes.append(axis_names.index(st.selectbox(label,axis_names,index=min(i,len(axis_names)-1),key=f"axis::{i}")))
        if len(set(selected_axes))<3: st.error("Choose three different axes."); st.stop()

st.session_state.setdefault("messages",[]); st.session_state.setdefault("atlas_results",[]); st.session_state.setdefault("reasoning_traces",[])
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt=st.chat_input("Write an utterance or ask ATLAS anything…")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    settings={"max_carriers":max_carriers,"expand":expand,"neighbor_limit":neighbor_limit,"metric":metric,"projection":projection,"standardize":standardize,"seed":seed,"offline":offline,"model":model_name,"hash_dims":hash_dims,"selected_axes":selected_axes}
    with st.chat_message("assistant"):
        with st.spinner("ATLAS is constructing and reading the current geometry…"):
            try:
                result=construct_atlas(prompt,axes,settings,axis_weights); st.session_state.atlas_results.append(result)
                if response_engine=="Ollama local LLM":
                    try:
                        reply, trace=geometry_grounded_reply(ollama_model,ollama_endpoint,st.session_state.messages[-(history_turns*2+1):],result,temperature,reasoning_detail,critic_enabled)
                        st.session_state.reasoning_traces.append(trace)
                    except (urllib.error.URLError,TimeoutError,KeyError,ValueError) as exc:
                        reply=atlas_fallback(result)+f"\n\n_Local LLM connection failed: {exc}_"
                else: reply=atlas_fallback(result)
            except Exception as exc:
                reply=f"ATLAS could not construct this turn's geometry: {exc}"
                result=None
        st.markdown(reply)
    st.session_state.messages.append({"role":"assistant","content":reply})

if st.session_state.atlas_results:
    result=st.session_state.atlas_results[-1]; df=pd.DataFrame(result["coordinates"]); edge_df=pd.DataFrame(result["relations"])
    st.divider(); st.subheader("Evidence for the latest reply")
    m=st.columns(5); m[0].metric("Carriers",len(result["carriers"])); m[1].metric("Typed axes",len(result["axes"])); m[2].metric("Relations",len(result["relations"])); m[3].metric("Projection",result["projection_used"]); m[4].metric("Stress",f"{result['normalized_stress']:.4f}")
    st.plotly_chart(geometry_figure(result),width="stretch",config={"scrollZoom":True,"displaylogo":False})
    tabs=st.tabs(["Typed coordinates","Lexical evidence","Differential geometry","Qwen reasoning trace","Export"])
    with tabs[0]:
        st.dataframe(df,hide_index=True,width="stretch",height=480)
    with tabs[1]:
        if not edge_df.empty:
            st.dataframe(edge_df,hide_index=True,width="stretch",height=420)
        else:
            st.info("No lexical evidence overlays for this turn.")
    with tabs[2]:
        st.json({"contract":"Discrete local numerical geometry; smooth-manifold status is not assumed.","knn_k":result["differential_geometry"]["knn_k"],"kernel_sigma":result["differential_geometry"]["kernel_sigma"]})
        st.dataframe(pd.DataFrame(result["differential_geometry"]["tangent_spaces"]),hide_index=True,width="stretch")
    with tabs[3]:
        if st.session_state.reasoning_traces:
            trace=st.session_state.reasoning_traces[-1]
            st.caption("This is the evidence packet and intermediate audit used for the latest Ollama response.")
            st.json(trace)
        else:
            st.json(atlas_reasoning_packet(result,reasoning_detail))
    with tabs[4]:
        st.download_button("Download current ATLAS result",json.dumps(result,indent=2,ensure_ascii=False),"atlas_chat_turn.json","application/json",width="stretch")
        st.download_button("Download current coordinates",df.to_csv(index=False),"atlas_chat_coordinates.csv","text/csv",width="stretch")
        transcript={"schema":"ATLAS_CHAT_TRANSCRIPT_v2_REASONING","messages":st.session_state.messages,"turn_results":st.session_state.atlas_results,"reasoning_traces":st.session_state.reasoning_traces}
        st.download_button("Download complete chat + geometry",json.dumps(transcript,indent=2,ensure_ascii=False),"atlas_geometry_chat.json","application/json",width="stretch")
