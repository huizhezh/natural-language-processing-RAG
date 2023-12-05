import collections
import os
from dotenv import load_dotenv
import openai
# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

raw_pdf_elements = partition_pdf(
    filename="../learningRAG3.8/nv.pdf",
    extract_images_in_pdf=False,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=".",
)

category_counts = collections.defaultdict(int)

for element in raw_pdf_elements:
    category = str(type(element))
    category_counts[category] += 1

unique_categories =list(category_counts.keys())
print(category_counts)

class Element(BaseModel) :
    type : str
    text : Any

table_elements = []
text_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        table_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        text_elements.append(Element(type="text", text=str(element)))
print('table_elements', table_elements)
print('*********************************')
print('text_elements', text_elements)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Build up
# summarization
# chain
# with LangChain framework

prompt_text = """
  You are responsible for concisely summarizing table or text chunk:

  {element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)
summarize_chain = {"element": lambda x: x} | prompt | ChatOpenAI(temperature=0,
                                                                 model="gpt-3.5-turbo") | StrOutputParser()

# Summarize each text and table element

tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

# print(text_summaries)
# Use LangChain MultiVectorRetriever to associate summaries of tables and texts with original text chunks in parent-child relationship.

import uuid

from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma

id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings()),
    docstore=InMemoryStore(),
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

from langchain.schema.runnable import RunnablePassthrough

template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(temperature=0, model="gpt-4")
    | StrOutputParser()
)

res = chain.invoke("Give summary about the stocks? Who is the beneficial owner?")
print('GPT4:',res)
print('*************')

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    | StrOutputParser()
)
res = chain.invoke("How many stocks were disposed? Who is the beneficial owner?")
print('GPT3:',res)

