"""Streamlit UI: search the BioASQ corpus, optionally generate a cited answer.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="BioASQ RAG", layout="wide")

with st.spinner("Loading models and indices (first run only, ~1-2 min)..."):
    import hybrid_search
    import reranker
    import generate_response

QTYPES = ["summary", "yesno", "factoid", "list"]
POOL = 100
TOP_K = 10

st.title("BioASQ biomedical literature search")

question = st.text_input("Question", placeholder="What is the difference between Cushing syndrome and Cushing disease?")
qtype = st.selectbox("Question type", QTYPES)

if st.button("Search", type="primary", disabled=not question):
    with st.spinner("Hybrid search + cross-encoder reranking..."):
        candidates = [pmid for pmid, _ in hybrid_search.search(question, k=POOL)]
        top_pmids = reranker.rerank(question, candidates, k=TOP_K)
        docs = reranker.CORPUS.loc[[str(p) for p in top_pmids]].reset_index().to_dict("records")
    st.session_state["question"] = question
    st.session_state["qtype"] = qtype
    st.session_state["docs"] = docs
    st.session_state.pop("generated", None)

if "docs" in st.session_state and st.session_state.get("question") == question:
    docs = st.session_state["docs"]
    st.subheader(f"Top {len(docs)} reranked results")
    for rank, d in enumerate(docs, start=1):
        with st.expander(f"{rank}. {d['title']}  (PMID {d['pmid']})"):
            st.write(d["abstract"])

    if st.button("Generate answer"):
        with st.spinner("Asking Gemini..."):
            result = generate_response.answer(
                st.session_state["question"],
                qtype=st.session_state["qtype"],
                docs=docs,
            )
        st.session_state["generated"] = result

    if "generated" in st.session_state:
        result = st.session_state["generated"]
        st.subheader("Generated answer")
        if result["exact_answer"] is not None:
            st.write("**Exact answer:**", result["exact_answer"])
        st.write(result["ideal_answer"])
