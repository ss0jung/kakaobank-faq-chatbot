import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# --- 1. 검색기(Retriever) 준비 ---

# 모델 및 데이터 로딩 (retriever.py와 동일)
model = SentenceTransformer("jhgan/ko-sbert-nli")
df = pd.read_json("kakaobank_faq_final.json")
embeddings = np.load("faq_embeddings.npy")

# FAISS 인덱스 생성 (retriever.py와 동일)
index = faiss.IndexFlatL2(768)
index.add(embeddings)


def search_faq(user_query, k=3):
    """사용자 질문과 가장 유사한 FAQ를 k개 검색"""
    query_vector = model.encode([user_query])
    D, I = index.search(query_vector, k)
    result_df = df.iloc[I[0]]
    return result_df


# --- 2. 생성기(Generator) 준비 ---

# OpenAI 클라이언트 초기화
# API 키는 자동으로 환경 변수 'OPENAI_API_KEY'에서 읽어옵니다.
client = OpenAI()


def generate_answer(user_query, search_results):
    """검색된 내용을 바탕으로 LLM을 통해 최종 답변 생성"""

    # 검색 결과를 프롬프트에 넣기 좋은 형식으로 변환
    context = ""
    for i, row in search_results.iterrows():
        context += f"Q: {row['question']}\nA: {row['answer']}\n\n"

    # OpenAI API에 전달할 프롬프트 구성
    prompt = f"""
당신은 카카오뱅크의 친절한 FAQ 안내 챗봇입니다.
주어진 '참고 자료'를 바탕으로 사용자의 '질문'에 대해 답변해 주세요.
자료에 명확한 답변이 없는 경우, "죄송하지만 문의하신 내용에 대한 정확한 답변을 찾지 못했습니다."라고 솔직하게 답변하세요.
절대 자료에 없는 내용을 지어내서 답변하면 안 됩니다.
답변은 한국어로, 친절하고 간결하게 해주세요.

---
[참고 자료]
{context}
---

[질문]
{user_query}

[답변]
"""

    # OpenAI ChatCompletion API 호출
    response = client.chat.completions.create(
        model="gpt-4o",  # 또는 "gpt-4" 등
        messages=[
            {"role": "system", "content": "당신은 카카오뱅크의 FAQ 안내 챗봇입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # 답변의 창의성 조절
    )

    return response.choices[0].message.content


# --- 3. 챗봇 실행 ---

if __name__ == "__main__":
    # 테스트 질문
    my_question = "해외 송금 한도를 늘리고 싶어요"
    # my_question = "mini카드는 어떻게 신청하나요?"

    print(f"💬 사용자 질문: {my_question}")

    # 1단계: 유사한 FAQ 검색 (Retrieval)
    retrieved_docs = search_faq(my_question)
    print("  - 검색된 관련 문서 수:", len(retrieved_docs))

    # 2단계: 검색 결과 기반으로 답변 생성 (Generation)
    final_answer = generate_answer(my_question, retrieved_docs)

    print(f"\n🤖 챗봇 답변:\n{final_answer}")
