import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableMap
from langchain.prompts import PromptTemplate

# 환경변수 로딩
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 설정
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 벡터 DB + LLM 설정
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
vectorstore = FAISS.load_local(
    "faiss_kakaobank_faq_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)

# k=3일 경우, 질문과 관련된 문서가 충분히 전달되지 않아
# LLM이 답변을 못하는 경우가 있었음 → k=5로 조정하여 정확도 개선
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)

prompt = PromptTemplate.from_template(
    """
당신은 카카오뱅크의 친절한 FAQ 안내 챗봇입니다.

주어진 '참고 자료'를 바탕으로 사용자의 '질문'에 대해 명확하고 간결하게 한국어로 답변해 주세요.

- 문서에 명확한 기준(예: 나이, 조건 등)이 있다면, 이를 바탕으로 질문에 해당 여부를 논리적으로 판단하세요.
- 주관적인 추측이나 문서에 없는 해석은 절대 하지 마세요.
- 절대 자료에 없는 내용을 지어내서 답변하면 안 됩니다.
- 문서에서 답변 가능한 근거가 없다면 아래와 같이 답변하세요:

"죄송하지만 문의하신 내용에 대한 정확한 답변을 찾기 어렵습니다.  
카카오뱅크 고객센터로 문의해주시면 더 자세히 안내해 드리겠습니다."


질문: {question}

참고 자료:
{context}

답변:
"""
)

rag_chain = (
    RunnableMap(
        {
            "context": lambda x: "\n\n".join(
                [doc.page_content for doc in retriever.invoke(x["question"])]
            ),
            "question": lambda x: x["question"],
        }
    )
    | prompt
    | llm
)


# HTML 렌더링 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 질문 응답 엔드포인트
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_api(request: QuestionRequest):
    try:
        question = request.question
        print("📌 질문:", question)
        print("📌 검색된 문서들:")
        for doc in retriever.invoke(question):
            print("----")
            print(doc.page_content)

        result = rag_chain.invoke({"question": question})
        # print("답변>>>", result.content)
        return {"answer": result.content}
    except Exception as e:
        print("❌ RAG 처리 오류 발생:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
