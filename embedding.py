import pandas as pd
import numpy as np
import logging
import os
from sentence_transformers import SentenceTransformer
from datetime import datetime

# logs 디렉토리 생성 (없으면 자동 생성)
if not os.path.exists("logs"):
    os.makedirs("logs")

# 오늘 날짜를 YYYYMMDD 형식으로 가져오기
today = datetime.now().strftime("%Y%m%d")
log_filename = f"logs/embedding_{today}.log"

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),  # 파일 로그
        logging.StreamHandler(),  # 콘솔 출력
    ],
)

# 1. 사전 훈련된 한국어 임베딩 모델 불러오기
# 이 모델은 문장을 768차원의 벡터로 변환해줍니다.
logging.info("임베딩 모델을 불러오는 중입니다...")
model = SentenceTransformer("jhgan/ko-sbert-nli")

# 2. 전 단계에서 수집한 FAQ 데이터 불러오기
logging.info("FAQ 데이터를 불러오는 중입니다...")
df = pd.read_json("kakaobank_faq_final.json")

# 3. 임베딩할 텍스트 데이터 준비
# '질문'과 '답변'을 합쳐서 하나의 텍스트로 만들어 의미를 더 풍부하게 합니다.
# 이렇게 하면 검색 시 질문뿐만 아니라 답변의 내용까지 고려할 수 있습니다.
logging.info("임베딩할 텍스트를 준비 중입니다...")
df["embedding_text"] = df["question"] + ". " + df["answer"]
texts_to_embed = df["embedding_text"].tolist()

# 4. 텍스트 데이터를 벡터로 변환 (임베딩 실행)
# 이 과정은 데이터의 양에 따라 몇 분 정도 소요될 수 있습니다.
logging.info(f"{len(texts_to_embed)}개의 텍스트에 대한 임베딩을 시작합니다...")
embeddings = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
logging.info("임베딩이 완료되었습니다.")

# 5. 생성된 임베딩 벡터를 파일로 저장
# .npy 형식은 파이썬에서 숫자 배열을 다룰 때 매우 효율적입니다.
logging.info("생성된 임베딩을 파일로 저장 중입니다...")
np.save("faq_embeddings.npy", embeddings)

logging.info("\n--- 작업 완료 ---")
logging.info(f"생성된 임베딩의 형태: {embeddings.shape}")
logging.info("'faq_embeddings.npy' 파일이 성공적으로 저장되었습니다.")
