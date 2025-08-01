import requests
import json
import time
import logging
from bs4 import BeautifulSoup

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),  # 파일 로그
        logging.StreamHandler(),  # 콘솔 출력 (기존 print와 동일)
    ],
)

# 1. 수집할 FAQ 카테고리 목록 정의
categories = [
    "예적금",
    "대출",
    "외환",
    "카드",
    "투자",
    "mini",
    "사업자",
    "ATM/CD",
    "제휴서비스",
    "보안설정",
    "알림",
    "앱서비스",
    "개인금고",
    "AI",
    "기타",
]

all_faqs = []
base_url = "https://www.kakaobank.com/api/v1/search/FAQ"

logging.info("#### 카카오뱅크 FAQ 데이터 수집을 시작합니다")

# 2. 각 카테고리를 순회
for category in categories:
    page = 1
    while True:
        logging.info(f"카테고리: '{category}', 페이지: {page} 수집 중...")
        params = {"tag": category, "page": page}

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

            # 3. JSON 데이터 구조에 맞게 파싱
            result_data = response.json().get("result", {})
            content_list = result_data.get("content", [])

            # 4. 내용이 없으면 해당 카테고리 수집 중단
            if not content_list:
                logging.info(
                    f"'{category}' 카테고리 수집 완료 (페이지 {page-1}에서 내용 없음)."
                )
                break

            # 5. 필요한 정보 추출 및 데이터 정제
            for item in content_list:
                question = item.get("title")

                # HTML 답변에서 텍스트만 추출
                html_answer = item.get("content", "")
                soup = BeautifulSoup(html_answer, "html.parser")
                answer = soup.get_text(
                    separator="\n"
                ).strip()  # 줄바꿈을 유지하며 텍스트 추출

                all_faqs.append(
                    {"category": category, "question": question, "answer": answer}
                )

            # 6. 마지막 페이지인지 확인 후 반복 중단
            if result_data.get("last", True):
                logging.info(f"'{category}' 카테고리 수집 완료 (마지막 페이지 도달).")
                break

            page += 1
            time.sleep(0.5)  # 서버 부하를 줄이기 위한 지연 시간

        except requests.exceptions.RequestException as e:
            logging.info(f"'{category}' 카테고리 수집 중 오류 발생: {e}")
            break
        except json.JSONDecodeError:
            logging.info(
                f"'{category}', 페이지 {page}에서 JSON 응답을 파싱할 수 없습니다."
            )
            break


# 7. 수집된 모든 데이터를 최종 JSON 파일로 저장
with open("kakaobank_faq_final.json", "w", encoding="utf-8") as f:
    json.dump(all_faqs, f, ensure_ascii=False, indent=2)

logging.info(
    f"\n총 {len(all_faqs)}개의 FAQ 수집 완료! kakaobank_faq_final.json 파일로 저장되었습니다."
)
