from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tiktoken
import numpy as np
from scipy.interpolate import interp1d
from typing import List, Literal, Optional, Union
import torch
import os
import time

# ==================== 全局缓存 ====================

# 缓存 tiktoken encoding，避免每次请求都重新创建
_token_encoding = None


def get_token_encoding():
    global _token_encoding
    if _token_encoding is None:
        _token_encoding = tiktoken.get_encoding('cl100k_base')
    return _token_encoding


# ==================== 环境变量 ====================

API_KEY = os.environ.get('sk-key', 'sk-aaabbbcccdddeeefffggghhhiiijjjkkk')


# ==================== FastAPI 应用 ====================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('本次加载模型的设备为GPU: ', torch.cuda.get_device_name(0))
else:
    print('本次加载模型的设备为CPU.')

# ---------- 加载 Embedding 模型 ----------
print('正在加载 Embedding 模型: moka-ai_m3e-large ...')
model = SentenceTransformer('moka-ai/m3e-large', device=device)

# ---------- 加载 Reranker 模型 ----------
print('正在加载 Reranker 模型: BAAI/bge-reranker-v2-m3 ...')
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)

# ---------- 模型预热 ----------
print('正在进行模型预热...')
_warmup_text = "模型预热文本"
model.encode(_warmup_text)
reranker.predict([(_warmup_text, _warmup_text)])
print('模型预热完成，服务已就绪')

# ---------- 认证 ----------
security = HTTPBearer()


# ==================== Pydantic 数据模型 ====================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: dict


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str
    usage: dict


# ==================== 工具函数 ====================

def num_tokens_from_string(string: str) -> int:
    """计算文本的 token 数量（使用缓存的 encoding）"""
    encoding = get_token_encoding()
    return len(encoding.encode(string))


def interpolate_vector(vector: np.ndarray, target_length: int) -> np.ndarray:
    """
    使用线性插值将向量扩展/压缩到目标维度。
    相比 PolynomialFeatures（52万维→截断），插值法计算量小且语义保持更好。
    """
    original_indices = np.arange(len(vector))
    target_indices = np.linspace(0, len(vector) - 1, target_length)
    f = interp1d(original_indices, vector, kind='linear')
    return f(target_indices)


# ==================== API 端点 ====================

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content='你说得对，但这个是向量模型不能对话'),
        finish_reason="stop",
    )
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    http_request: Request,
    request: EmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )

    try:
        # 批量编码（利用模型内部 batch 能力，比逐条循环快得多）
        embeddings = model.encode(request.input)

        # 维度扩展：M3E-Large 是 1024维 -> 扩展到 1536维（兼容 OpenAI 格式）
        embeddings = [
            interpolate_vector(emb, 1536) if len(emb) != 1536 else emb
            for emb in embeddings
        ]

        # L2 归一化
        embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]

        # 转 Python 列表
        embeddings = [emb.tolist() for emb in embeddings]

        prompt_tokens = sum(len(text.split()) for text in request.input)
        total_tokens = sum(num_tokens_from_string(text) for text in request.input)

        response = {
            "data": [
                {
                    "embedding": embedding,
                    "index": index,
                    "object": "embedding",
                }
                for index, embedding in enumerate(embeddings)
            ],
            "model": request.model,
            "object": "list",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            },
        }
        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {str(e)}",
        )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Rerank API（兼容 Cohere Rerank 格式）

    对 query 和每个 document 计算相关性分数，按分数降序排列，返回 top_n 个结果。
    """
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )

    if not request.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="documents must not be empty",
        )

    try:
        # 构造 query-document 对
        pairs = [(request.query, doc) for doc in request.documents]

        # 计算相关性分数
        scores = reranker.predict(pairs)

        # 构建结果列表
        results = [
            {
                "index": idx,
                "relevance_score": float(score),
                "document": {"text": request.documents[idx]},
            }
            for idx, score in enumerate(scores)
        ]

        # 按分数降序排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # 截取 top_n
        top_n = request.top_n
        if top_n is not None:
            results = results[:top_n]

        # 计算 token 用量
        total_tokens = num_tokens_from_string(request.query) + sum(
            num_tokens_from_string(doc) for doc in request.documents
        )

        return RerankResponse(
            results=results,
            model=request.model,
            usage={"total_tokens": total_tokens},
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rerank failed: {str(e)}",
        )


# ==================== 启动入口 ====================

if __name__ == "__main__":
    uvicorn.run("localembedding:app", host='0.0.0.0', port=6008, workers=1)