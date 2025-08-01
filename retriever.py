import pandas as pd
import numpy as np
import faiss
import logging
import os
from sentence_transformers import SentenceTransformer
from datetime import datetime

# logs 디렉토리 생성 (없으면 자동 생성)
if not os.path.exists("logs"):
    os.makedirs("logs")

# 오늘 날짜를 YYYYMMDD 형식으로 가져오기
today = datetime.now().strftime("%Y%m%d")
log_filename = f"logs/retriever_{today}.log"

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),  # 파일 로그
        logging.StreamHandler(),  # 콘솔 출력
    ],
)

# 1. 임베딩 모델과 FAQ 데이터 불러오기
model = SentenceTransformer("jhgan/ko-sbert-nli")
df = pd.read_json("kakaobank_faq_final.json")
embeddings = np.load("faq_embeddings.npy")

# 2. FAISS 인덱스 생성 및 벡터 추가
# 768은 ko-sbert-nli 모델의 임베딩 차원(벡터의 길이)입니다.
index = faiss.IndexFlatL2(768)
index.add(embeddings)


# 3. 검색기 함수 정의
def search_faq(user_query, k=3):
    """
    사용자의 질문(query)과 가장 유사한 FAQ를 k개 검색합니다.
    """
    # 사용자 질문을 벡터로 변환
    query_vector = model.encode([user_query])

    # FAISS 인덱스에서 가장 가까운 벡터 k개를 검색
    # D: 유사도(거리), I: 벡터의 인덱스(위치)
    D, I = index.search(query_vector, k)

    # 검색된 FAQ의 인덱스로 원본 데이터프레임에서 해당 FAQ를 가져옴
    result_df = df.iloc[I[0]]

    return result_df


# 4. 테스트 실행
if __name__ == "__main__":
    # 테스트 질문
    my_question = "해외 송금 한도를 늘리고 싶어요"

    logging.info(f"질문: {my_question}\n")
    logging.info("--- 가장 유사한 FAQ Top 3 ---")

    # 검색 실행
    search_results = search_faq(my_question, k=3)

    # 결과 출력
    for i, row in search_results.iterrows():
        logging.info(f"질문: {row['question']}")
        logging.info(f"답변: {row['answer']}\n")
