import json
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
load_in_4bit=True

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectordb = Chroma(persist_directory="chromadb_reviews", embedding_function=hf_embeddings)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10})


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token="Your token here",
    temperature=0.3,
    max_new_tokens=512
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable and thoughtful beauty advisor.

A user has asked the following question:
❓ {question}

Based on the product reviews below, perform the following steps:
1. Identify relevant products based on the user's request (e.g. skin type, brand, product type).
2. Summarize what people like (**pros**) and dislike (**cons**) about each product.
3. Analyze the **sentiment** of the reviews for each product (Positive / Mixed / Negative).
4. Provide a personalized **recommendation or insight**, tailored to the user's query, in 4–6 informative sentences.

If the reviews include multiple products, handle them one by one.

📄 Product Reviews:
{context}

📌 Output format:
- Product: <Product Name>
- Skin Type: (if mentioned)
- Brand: <if known>
- Pros: <List key advantages>
- Cons: <List key complaints>
- Sentiment: Positive / Mixed / Negative
- Recommendation: <Summarize and provide tailored advice>

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt},
    
)

st.set_page_config(page_title="💄 Beauty Product Recommender", layout="centered")
st.title("💬 Ask for a Beauty Product Recommendation")
query = st.text_input("Enter your question (e.g., cleanser for oily skin)")

if query:
    with st.spinner("Analyzing reviews and generating recommendation..."):
        response = qa_chain.invoke({"query": query})
        raw_output = response["result"]

        # Strip everything before "Answer:"
        final_output = raw_output.split("Answer:")[-1].strip() if "Answer:" in raw_output else raw_output.strip()

        st.markdown("### 💡 Recommendation")
        st.markdown(final_output)

        with st.expander("🔍 View Source Chunks"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**{i+1}. {doc.metadata.get('product')}**")
                st.caption(f"⭐ {doc.metadata.get('average_rating')} | 💬 {doc.metadata.get('rating_number')}")
                st.write(doc.page_content[:300] + "...")
