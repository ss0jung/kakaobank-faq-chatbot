import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import time  # 시간 지연을 위해 추가

# .env 파일에서 환경 변수 로드
load_dotenv()

# 1. OpenAI 클라이언트 초기화
client = OpenAI()

# 2. FAQ 데이터 불러오기
print("FAQ 데이터를 불러오는 중입니다...")
df = pd.read_json("kakaobank_faq_final.json")

# 3. 임베딩할 텍스트 데이터 준비
print("임베딩할 텍스트를 준비 중입니다...")
df["embedding_text"] = df["question"] + ". " + df["answer"]
texts_to_embed = df["embedding_text"].tolist()

# 4. 배치 처리를 통해 텍스트를 벡터로 변환 (수정된 부분)
print(f"{len(texts_to_embed)}개의 텍스트에 대한 임베딩을 시작합니다...")
try:
    batch_size = 100  # 한 번에 처리할 텍스트의 수
    all_embeddings = []

    for i in range(0, len(texts_to_embed), batch_size):
        # 텍스트 목록에서 현재 처리할 배치(묶음)를 가져옵니다.
        batch = texts_to_embed[i : i + batch_size]

        print(
            f"  - 배치 {i//batch_size + 1} 처리 중 ({i+1} ~ {min(i + batch_size, len(texts_to_embed))}번 텍스트)"
        )

        # OpenAI Embedding API 호출
        response = client.embeddings.create(input=batch, model="text-embedding-3-small")

        # 현재 배치의 임베딩 결과를 전체 목록에 추가
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # API 속도 제한을 피하기 위해 잠시 대기 (선택 사항이지만 권장)
        time.sleep(0.1)

    # 모든 배치의 결과를 하나의 numpy 배열로 변환
    embeddings = np.array(all_embeddings)
    print("임베딩이 완료되었습니다.")

    # 5. 생성된 임베딩 벡터를 파일로 저장
    print("생성된 임베딩을 파일로 저장 중입니다...")
    np.save("faq_embeddings_openai.npy", embeddings)

    print("\n--- 작업 완료 ---")
    print(f"생성된 임베딩의 형태: {embeddings.shape}")
    print("'faq_embeddings_openai.npy' 파일이 성공적으로 저장되었습니다.")

except Exception as e:
    print(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
