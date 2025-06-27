import os
import json
import traceback
from fastapi import FastAPI, Request, HTTPException, Form, Depends
from fastapi.responses import Response, HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
import uvicorn
from typing import Dict, List, Optional, Union
from config import config
from pydantic import BaseModel
from datetime import datetime, timezone
import time
import logging


# 配置日志记录
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0))
templates = Jinja2Templates(directory="templates")


# --- 辅助函数 ---
def extract_tokens(usage: dict) -> tuple[int, int]:
    try:
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        return prompt_tokens, completion_tokens
    except Exception:
        return 0, 0


def del_key(d: Dict, key: str):
    if key in d:
        del d[key]


def _format_ollama_timestamp(ts: Optional[float] = None) -> str:
    """将时间戳格式化为 Ollama 兼容的特定 ISO 8601 字符串。"""
    dt = datetime.fromtimestamp(ts, timezone.utc) if ts else datetime.now(timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')


def _prepare_openai_request(request: Union["ChatRequest", "GenerateRequest"]) -> tuple[dict, dict]:
    """为 chat/generate 请求准备 openai_body 和 headers"""
    if isinstance(request, GenerateRequest):
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
    else:  # ChatRequest
        messages = [{"role": m.role, "content": m.content}
                    for m in request.messages]

    openai_body = {
        "model": config.model_mapping.get(request.model, request.model),
        "messages": messages,
        "stream": request.stream
    }

    if request.options:
        if "temperature" in request.options:
            openai_body["temperature"] = request.options["temperature"]
        if "top_p" in request.options:
            openai_body["top_p"] = request.options["top_p"]
        if "num_ctx" in request.options:
            openai_body["max_tokens"] = request.options["num_ctx"]

    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json"
    }
    return openai_body, headers


async def _call_openai_non_stream(openai_body: dict, headers: dict) -> dict:
    """调用 OpenAI 进行非流式响应"""
    try:
        response = await client.post(f"{config.openai_api_base}/v1/chat/completions", json=openai_body, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI API 错误: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"OpenAI API 错误: {e.response.text}")
    except Exception as e:
        logger.error(f"处理非流式请求时发生错误: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


async def _stream_openai_chat_completions(openai_body: dict, headers: dict):
    """调用 OpenAI 并生成解析后的 SSE 数据块"""
    try:
        async with client.stream("POST", f"{config.openai_api_base}/v1/chat/completions", json=openai_body, headers=headers) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                error_detail = error_body.decode()
                logger.error(
                    f"OpenAI API 错误: {response.status_code} - {error_detail}")
                yield {"error": f"Upstream API error: {error_detail}"}
                return

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line.removeprefix("data: ").strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    data = json.loads(chunk)
                    yield data
                except json.JSONDecodeError:
                    logger.warning(f"JSON 解析错误, chunk: {chunk}")
                    continue
    except Exception as e:
        logger.error(f"处理流式请求时发生错误: {str(e)}")
        traceback.print_exc()
        yield {"error": f"An unexpected error occurred: {str(e)}"}


async def _handle_ollama_stream_request(request: Union["ChatRequest", "GenerateRequest"], response_type: str):
    """处理 Ollama 兼容的流式请求。"""
    openai_body, headers = _prepare_openai_request(request)
    logger.info(f"向 OpenAI 发送流式请求: {json.dumps(openai_body, ensure_ascii=False)}")

    async def stream_generator():
        async for data in _stream_openai_chat_completions(openai_body, headers):
            if "error" in data:
                error_chunk = {
                    "model": request.model,
                    "created_at": _format_ollama_timestamp(),
                    "done": True,
                    "error": data["error"]
                }
                if response_type == "chat":
                    error_chunk["message"] = {"role": "assistant", "content": ""}
                else:  # generate
                    error_chunk["response"] = ""
                yield json.dumps(error_chunk) + "\n"
                break

            if not ("choices" in data and len(data["choices"]) > 0):
                continue
            
            choice = data["choices"][0]
            iso_time = _format_ollama_timestamp(data.get("created", time.time()))

            # 最终块
            if finish_reason := choice.get("finish_reason"):
                final_content = choice.get("delta", {}).get("content", "")
                chunk = {
                    "model": request.model, "created_at": iso_time, "done": True,
                    "done_reason": finish_reason, "total_duration": 0, "load_duration": 0,
                    "prompt_eval_count": 0, "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0
                }
                if response_type == "chat":
                    chunk["message"] = {"role": "assistant", "content": final_content}
                else:  # generate
                    chunk["response"] = final_content
                    chunk["context"] = []
                yield json.dumps(chunk) + "\n"
            # 中间块
            elif "delta" in choice and (content := choice["delta"].get("content")):
                chunk = {"model": request.model, "created_at": iso_time, "done": False}
                if response_type == "chat":
                    chunk["message"] = {"role": "assistant", "content": content}
                else:  # generate
                    chunk["response"] = content
                yield json.dumps(chunk) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


async def _handle_ollama_non_stream_request(request: Union["ChatRequest", "GenerateRequest"], response_type: str):
    """处理 Ollama 兼容的非流式请求。"""
    openai_body, headers = _prepare_openai_request(request)
    logger.info(f"向 OpenAI 发送非流式请求: {json.dumps(openai_body, ensure_ascii=False)}")

    data = await _call_openai_non_stream(openai_body, headers)
    if "choices" in data and len(data["choices"]) > 0:
        # 直接构建 Ollama 兼容的响应体
        choice = data["choices"][0]
        content = choice["message"]["content"]
        
        response_body = {
            "model": request.model,
            "created_at": _format_ollama_timestamp(data.get("created", time.time())),
            "done": True,
            "done_reason": choice.get("finish_reason"),
            "total_duration": 0, "load_duration": 0, "prompt_eval_count": 0,
            "prompt_eval_duration": 0, "eval_count": 0, "eval_duration": 0
        }
        if response_type == "chat":
            response_body["message"] = {"role": "assistant", "content": content}
        else:  # generate
            response_body["response"] = content
            response_body["context"] = []

        return JSONResponse(content=response_body)
    else:
        logger.error(f"OpenAI 响应缺少 choices: {data}")
        detail = "No response from model" if response_type == "generate" else "OpenAI API 返回了无效的响应格式"
        raise HTTPException(status_code=500, detail=detail)


def _clean_response_content(content: str) -> str:
    """清理响应内容中的空行"""
    return "\n".join(line for line in content.splitlines() if line.strip())


@app.middleware("http")
async def log_requests_responses(request: Request, call_next):
    # 记录请求日志
    start_time = time.perf_counter()
    body_bytes = await request.body()
    request_body_str = body_bytes.decode('utf-8')

    if request_body_str:
        logger.info(
            f"[请求开始] {request.method} {request.url.path} | 请求体: \n{request_body_str}")
    else:
        logger.info(f"[请求开始] {request.method} {request.url.path}")

    # 调用下游处理
    async def receive():
        return {"type": "http.request", "body": body_bytes}
    request = Request(request.scope, receive)
    response = await call_next(request)

    # 为所有响应添加CORS头
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,HEAD,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization,Content-Type,User-Agent,Accept,X-Requested-With,Openai-Beta,X-Stainless-Arch,X-Stainless-Async,X-Stainless-Custom-Poll-Interval,X-Stainless-Helper-Method,X-Stainless-Lang,X-Stainless-Os,X-Stainless-Package-Version,X-Stainless-Poll-Helper,X-Stainless-Retry-Count,X-Stainless-Runtime,X-Stainless-Runtime-Version,X-Stainless-Timeout"
    response.headers["Access-Control-Max-Age"] = "43200"
    
    # 删除Content-Length头以避免协议错误
    if "content-length" in response.headers:
        del response.headers["content-length"]

    # 只处理补全请求
    if request.url.path not in ["/v1/chat/completions", "/api/chat", "/api/generate"] or request.method != "POST":
        return response

    if "text/event-stream" in response.headers.get("content-type") or "application/x-ndjson" in response.headers.get("content-type"):
        # 对于流式响应，包装 body_iterator 以正确记录时间
        async def logging_stream_wrapper():
            usages = {}
            response_content = ""
            first_chunk_time = None

            async for chunk in response.body_iterator:
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()

                # 尝试解析块以收集内容和用法
                chunk_str = chunk.decode('utf-8')

                if chunk_str.startswith('data: '):
                    # OpenAI流响应
                    chunk_data = chunk_str[6:].strip()
                    if chunk_data and chunk_data != '[DONE]':
                        data = json.loads(chunk_data)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                response_content += delta['content']
                        if 'usage' in data:
                            usages = data['usage']
                elif chunk_str.startswith('{'):
                    # Ollama流响应
                    data = json.loads(chunk_str)
                    # /api/chat
                    if 'message' in data and data['message']:
                        message = data['message']
                        if 'content' in message:
                            response_content += message['content']
                    # /api/generate
                    if 'response' in data and data['response']:
                        response_content += data['response']

                yield chunk

            # 流结束后，进行日志记录
            end_time = time.perf_counter()

            # 如果流为空，first_chunk_time 将为 None
            actual_first_response_time = first_chunk_time if first_chunk_time is not None else end_time

            prompt_duration = actual_first_response_time - start_time
            completion_duration = end_time - actual_first_response_time
            total_duration = end_time - start_time

            prompt_tokens, completion_tokens = extract_tokens(usages)
            prompt_speed = prompt_tokens / prompt_duration if prompt_duration > 0 else 0
            completion_speed = completion_tokens / \
                completion_duration if completion_duration > 0 else 0

            # 清理响应内容中的空行
            cleaned_response_content = _clean_response_content(response_content)

            logger.info(f"[响应完成] {request.method} {request.url.path} {response.status_code} | "
                        f"请求Token数: {prompt_tokens} | 响应Token数: {completion_tokens} | "
                        f"请求时间: {prompt_duration:.3f}s | "
                        f"响应时间: {completion_duration:.3f}s | "
                        f"总耗时: {total_duration:.3f}s | "
                        f"请求速度: {prompt_speed:.2f} tokens/s | "
                        f"响应速度: {completion_speed:.2f} tokens/s | "
                        f"响应内容: \n{cleaned_response_content}")

        return StreamingResponse(
            logging_stream_wrapper(),
            status_code=response.status_code,
            headers=response.headers,
            media_type=response.media_type
        )
    else:
        # 处理非流式响应
        response_chunks = []
        async for chunk in response.body_iterator:
            response_chunks.append(chunk)

        response_body = b''.join(response_chunks)
        response_body_str = response_body.decode('utf-8')

        response_json = json.loads(response_body_str)
        if 'choices' in response_json: # OpenAI
            response_content = response_json['choices'][0]['message']['content']
        elif 'message' in response_json: # Ollama /api/chat
            response_content = response_json['message']['content']
        elif 'response' in response_json: # Ollama /api/generate
            response_content = response_json['response']
        else:
            print(f"Unexpected response format: {response_json}")
            return response

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        prompt_tokens, completion_tokens = extract_tokens(
            response_json.get('usage', {}))
        total_tokens_count = prompt_tokens + completion_tokens
        overall_speed = total_tokens_count / total_duration if total_duration > 0 else 0

        # 清理响应内容中的空行
        cleaned_response_content = _clean_response_content(response_content)
        if not cleaned_response_content:
            cleaned_response_content = "(No visible response content after cleaning empty lines)"

        logger.info(f"[响应完成] {request.method} {request.url.path} {response.status_code} | "
                    f"请求Token数: {prompt_tokens} | 响应Token数: {completion_tokens} | "
                    f"总耗时: {total_duration:.3f}s | "
                    f"总速度: {overall_speed:.2f} tokens/s | "
                    f"响应内容: \n{cleaned_response_content}")

        # 由于我们消耗了 body_iterator，因此重新创建响应
        return JSONResponse(
            content=response_json,
            status_code=response.status_code,
            headers=dict(response.headers)
        )


# --- Pydantic 模型 ---
class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    format: Optional[str] = None
    options: Optional[Dict] = None


class EmbeddingRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    options: Optional[Dict] = None


class ShowRequest(BaseModel):
    model: str
    verbose: Optional[bool] = False


# --- 认证与配置 UI ---
# 简单会话管理，生产环境建议使用更安全的方式
sessions = set()


def is_authenticated(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return RedirectResponse(url="/login", status_code=302)
    return True


@app.get("/", response_class=JSONResponse)
async def root_status():
    return {"status": "Ollama is running"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if password == config.admin_password:
        session_id = os.urandom(16).hex()
        sessions.add(session_id)
        response = RedirectResponse(url="/config", status_code=302)
        response.set_cookie(key="session_id", value=session_id)
        return response
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "密码错误"},
        status_code=401
    )


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request, _=Depends(is_authenticated)):
    return templates.TemplateResponse(
        "config.html",
        {
            "request": request,
            "config": config,
            "model_mapping_json": json.dumps(config.model_mapping, indent=2)
        }
    )


@app.post("/config")
async def save_config(
    request: Request,
    admin_password: str = Form(...),
    openai_api_key: str = Form(...),
    ollama_api_key: str = Form(None),
    openai_api_base: str = Form(...),
    model_mapping: str = Form(...),
    _=Depends(is_authenticated)
):
    try:
        model_mapping_dict = json.loads(model_mapping)
        config.admin_password = admin_password
        config.openai_api_key = openai_api_key
        config.ollama_api_key = ollama_api_key if ollama_api_key else None
        config.openai_api_base = openai_api_base
        config.model_mapping = model_mapping_dict
        config.save()

        return templates.TemplateResponse(
            "config.html",
            {
                "request": request,
                "config": config,
                "model_mapping_json": model_mapping,
                "success": "配置已保存"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "config.html",
            {
                "request": request,
                "config": config,
                "model_mapping_json": model_mapping,
                "error": f"保存失败: {str(e)}"
            }
        )


# --- Ollama 兼容 API ---
@app.get("/api/tags")
async def list_models():
    try:
        # 获取OpenAI可用模型列表
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }
        response = await client.get(
            f"{config.openai_api_base}/v1/models",
            headers=headers
        )
        openai_models = response.json()

        models = []
        # 修改后的 model_details 基础模板
        base_model_details = {
            "parent_model": "",
            "format": "gguf",
            "family": "openai",
            "families": ["openai"],
            "parameter_size": "N/A",
            "quantization_level": "N/A"
        }

        # 添加所有原始模型
        for model in openai_models.get("data", []):
            model_id = model.get("id", "")
            if not model_id:  # 如果 model_id 为空则跳过
                continue

            name_with_tag = model_id
            created_timestamp = model.get("created")

            if created_timestamp:
                modified_at_iso = _format_ollama_timestamp(created_timestamp)
            else:
                modified_at_iso = _format_ollama_timestamp()

            current_details = base_model_details.copy()
            models.append({
                "name": name_with_tag,
                "model": name_with_tag,
                "modified_at": modified_at_iso,
                "size": 0,
                "digest": "",
                "details": current_details
            })

            # 添加映射的别名
            aliases = [k for k, v in config.model_mapping.items()
                       if v == model_id]
            for alias in aliases:
                alias_details = base_model_details.copy()
                # 别名也使用相同的 details 结构和时间戳
                models.append({
                    "name": f"{alias}",
                    "model": f"{alias}",
                    "modified_at": modified_at_iso,  # 使用原始模型的时间戳
                    "size": 0,
                    "digest": "",
                    "details": alias_details
                })

        return {"models": models}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """兼容 Ollama 的 /api/chat 接口"""
    if request.stream:
        return await _handle_ollama_stream_request(request, "chat")
    else:
        return await _handle_ollama_non_stream_request(request, "chat")


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """兼容 Ollama 的 /api/generate 接口"""
    if request.stream:
        return await _handle_ollama_stream_request(request, "generate")
    else:
        return await _handle_ollama_non_stream_request(request, "generate")


@app.post("/api/embeddings")
async def create_embedding(request: EmbeddingRequest):
    try:
        # 将 Ollama 格式转换为 OpenAI 格式
        openai_body = {
            "model": config.model_mapping.get(request.model, request.model),
            "input": request.prompt
        }

        if request.options:
            # 转换 Ollama 选项到 OpenAI 参数
            if "dimensions" in request.options:
                openai_body["dimensions"] = request.options["dimensions"]

        # 准备请求头
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"发送 embedding 请求到 OpenAI: {openai_body}")  # 添加日志

        # 发送请求到 Open AI 兼容接口
        response = await client.post(
            f"{config.openai_api_base}/v1/embeddings",
            json=openai_body,
            headers=headers
        )

        if response.status_code != 200:
            error_detail = await response.text()
            logger.error(f"OpenAI API 错误: {error_detail}")  # 添加日志
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API 错误: {error_detail}"
            )

        # 处理响应
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            embeddings = [item["embedding"] for item in data["data"]]
            return {
                "embedding": embeddings[0] if isinstance(request.prompt, str) else embeddings
            }
        else:
            logger.error(f"OpenAI 响应缺少 embeddings: {data}")  # 添加日志
            raise HTTPException(
                status_code=500, detail="OpenAI API 返回了无效的响应格式")

    except httpx.RequestError as e:
        logger.error(f"请求错误: {str(e)}")  # 添加日志
        raise HTTPException(
            status_code=500, detail=f"请求 OpenAI API 失败: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {str(e)}")  # 添加日志
        raise HTTPException(
            status_code=500, detail=f"解析 OpenAI 响应失败: {str(e)}")
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")  # 添加日志
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@app.post("/api/show")
async def show_model(req: ShowRequest):
    # 从模板文件读取响应
    template_path = os.path.join(
        "templates", "response-api-show.json")
    with open(template_path, "r", encoding="utf-8") as f:
        resp = json.load(f)
    # 修改一些必要信息
    resp["model_info"]["general.basename"] = req.model
    if req.verbose:
        resp["model_info"]["tokenizer.ggml.merges"] = []
        resp["model_info"]["tokenizer.ggml.token_type"] = []
        resp["model_info"]["tokenizer.ggml.tokens"] = []
    else:
        resp["model_info"]["tokenizer.ggml.merges"] = None
        resp["model_info"]["tokenizer.ggml.token_type"] = None
        resp["model_info"]["tokenizer.ggml.tokens"] = None
    return resp

# --- OpenAI 直通 API ---
@app.post("/v1/chat/completions")
async def forward_chat(request: Request):
    """一个简单的 OpenAI /v1/chat/completions 直通代理"""
    # 调整请求头
    headers = {k: v for k, v in request.headers.items() if k.lower() not in [
        "host", "content-length", "authorization"]}
    headers["authorization"] = f"Bearer {config.openai_api_key}"

    # 调整请求体
    openai_body = await request.json()
    openai_body["model"] = config.model_mapping.get(
        openai_body["model"], openai_body["model"])

    if openai_body.get("stream", False):
        async def stream_generator():
            try:
                async with client.stream("POST", f"{config.openai_api_base}/v1/chat/completions", json=openai_body, headers=headers) as response:
                    async for chunk in response.aiter_lines():
                        if chunk:  # 忽略按行读取时的空行
                            chunk = chunk.removeprefix("data: ")
                            if chunk.strip() != "[DONE]":  # 结束标志不处理
                                # 修改响应格式
                                data = json.loads(chunk)
                                data["id"] = "chatcmpl-133"
                                data["system_fingerprint"] = "fp_ollama"
                                if "choices" in data:
                                    for choice in data["choices"]:
                                        del_key(choice, "text")
                                chunk = json.dumps(data)

                            yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"转发流式请求时发生错误: {str(e)}")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            response = await client.post(f"{config.openai_api_base}/v1/chat/completions", json=openai_body, headers=headers)
            response.raise_for_status()
            data = response.json()
            data["id"] = "chatcmpl-106"
            data["system_fingerprint"] = "fp_ollama"
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"转发请求失败: {str(e)}")


@app.get("/v1/models")
async def get_models(request: Request):
    """一个简单的 OpenAI /v1/models 直通代理，并进行格式清理"""
    try:
        # 调整请求头
        headers = {k: v for k, v in request.headers.items() if k.lower() not in [
            "host", "content-length", "authorization"]}
        headers["authorization"] = f"Bearer {config.openai_api_key}"

        # 转发请求到上游
        response = await client.get(
            f"{config.openai_api_base}/v1/models",
            headers=headers
        )

        # 处理响应数据
        resp = response.json()
        resp["object"] = "list"
        if "data" in resp:
            for _d in resp["data"]:
                del_key(_d, "permission")
                del_key(_d, "root")
                del_key(_d, "parent")
        del_key(resp, "success")

        return resp
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"转发请求失败: {str(e)}")

# --- OPTIONS 请求处理 ---
@app.options("/{path:path}")
async def handle_options(path: str):
    """处理所有OPTIONS请求，返回204状态码"""
    return Response(status_code=204)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
