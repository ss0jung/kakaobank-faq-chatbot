import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()
client = OpenAI()

file_path = "kakaobank_faq_final.json"

# 1. JSON 파일 로드
with open(file_path, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# 2. Document 형태로 변환 (LangChain 요구 포맷)
documents = []
for item in faq_data:
    content = f"질문: {item['question']}\n답변: {item['answer']}"
    metadata = {
        "category": item["category"],
        "original_question": item["question"],
        "original_answer": item["answer"],
    }
    documents.append(Document(page_content=content, metadata=metadata))

# 3. OpenAI 임베딩 모델 지정
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=1000)

# 4. FAISS를 사용한 벡터 저장소 생성
vectorstore = FAISS.from_documents(documents, embedding_model)

# 5. 저장 (로컬에 faiss_index 저장)
vectorstore.save_local("faiss_kakaobank_faq_index")

# 6. 테스트
query = "청소년은 계좌를 개설할 수 있나요?"
retrieved_docs = vectorstore.similarity_search(query, k=3)

print(f"\n------코사인 시밀러리티 검색 테스트-----")
print(f"입력 질문:{query}")
print(f"\n가장 유사한 FAQ {len(retrieved_docs)}개:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n[유사도 순위 {i+1}]")
    print(f"매칭된 질문: {doc.metadata['original_question']}")
    print(f"매칭된 내용: {doc.page_content}")  # 전체 내용을 보려면 주석 해제
