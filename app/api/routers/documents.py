"""文档管理接口（仅管理员）。

    POST   /documents                上传并异步入库
    GET    /documents/{task_id}      查询进度（不计入每分钟额度）
    DELETE /documents/{source_file}  删除该文件的全部切块

权限设计：
- 名单外的账号一律 404，不是 403。403 等于告诉对方「这里有个接口你没权限」；
  404 与访问一个不存在的路径表现完全一致，不泄露任何信息。
- 前端靠 /auth/me 的 is_admin 决定是否渲染入口，后端这道校验是底线、不是可选项：
  前端隐藏挡不住直接发请求。
"""

import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from app.api.deps import get_resources, require_user_id
from app.schemas.common import ErrorResponse
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentTaskResponse,
    DocumentUploadResponse,
)
from app.services.document_ingest import (
    UploadRejected,
    matches_signature,
    remove_temp_file,
    run_ingest_task,
    sanitize_filename,
    save_upload_to_temp,
    validate_extension,
)
from src.ingest_index_chunks import count_chunks_by_source_file

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["文档管理"],
    responses={
        404: {"model": ErrorResponse, "description": "资源不存在或无权访问"},
        500: {"model": ErrorResponse, "description": "服务器内部处理异常"},
    },
)

_NOT_FOUND = {"error_code": "NOT_FOUND", "message": "请求的资源不存在"}


def require_admin(
    request: Request,
    user_id: str = Depends(require_user_id),
) -> str:
    """管理员校验。名单外一律 404。

    单独做成依赖而不是塞进 require_user_id：认证回答「你是谁」，
    授权回答「你能干什么」。将来换真实 JWT、加角色表时只替换这个函数。
    """
    resources = get_resources(request)
    if not resources.is_admin(user_id):
        logger.warning(
            "非管理员访问文档管理接口 user=%s path=%s", user_id, request.url.path
        )
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return user_id


@router.post(
    "",
    status_code=202,
    response_model=DocumentUploadResponse,
    summary="上传文档并入库",
    responses={
        400: {"model": ErrorResponse, "description": "文件名为空或文件为空"},
        409: {"model": ErrorResponse, "description": "同名文件已入库或正在处理"},
        413: {"model": ErrorResponse, "description": "文件超过大小上限"},
        415: {"model": ErrorResponse, "description": "格式不支持或文件头不匹配"},
    },
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="待入库的 .pdf 或 .docx"),
    admin_id: str = Depends(require_admin),
) -> DocumentUploadResponse:
    resources = get_resources(request)
    store = resources.ingest_task_store

    # 一、文件名清洗（含路径穿越断言）
    try:
        source_file = sanitize_filename(file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_FILENAME", "message": f"文件名不可用：{exc}"},
        ) from exc

    # 二、扩展名白名单，落盘前拒绝
    try:
        ext = validate_extension(source_file, resources.allowed_upload_extensions)
    except UploadRejected as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc

    # 三、同名检测：查库。>0 说明这个文件名已经入库
    existing = await count_chunks_by_source_file(resources.vector_store, source_file)
    if existing > 0:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "DOCUMENT_ALREADY_EXISTS",
                "message": (
                    f"该文件已入库（现有 {existing} 个块），本阶段不支持覆盖更新。"
                    "请重命名后上传，或联系管理员先删除旧数据"
                ),
            },
        )

    # 四、并发占位：两个请求同时传同名文件时只放行一个
    claim_token = store.claim(source_file)
    if claim_token is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "DOCUMENT_PROCESSING",
                "message": "同名文件正在处理中，请等待当前任务结束后再试",
            },
        )

    tmp_path = None
    try:
        # 五、先落盘到受控临时文件：请求结束后 UploadFile 就不可用了
        try:
            tmp_path, _size, header = await save_upload_to_temp(
                file,
                dest_dir=resources.upload_temp_dir,
                filename=source_file,
                max_bytes=resources.upload_max_bytes,
            )
        except UploadRejected as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"error_code": exc.error_code, "message": exc.message},
            ) from exc

        if not matches_signature(header, ext):
            raise HTTPException(
                status_code=415,
                detail={
                    "error_code": "FILE_TYPE_MISMATCH",
                    "message": f"文件内容与扩展名 {ext} 不符，请确认文件未损坏、未被改名",
                },
            )

        # 六、建任务
        task = store.create(
            owner_user_id=admin_id,
            source_file=source_file,
            claim_token=claim_token,
        )
    except BaseException:
        # 任务没建成：必须释放占位，否则这个文件名再也传不上去
        remove_temp_file(tmp_path)
        store.release(source_file, claim_token)
        raise

    # 七、交给后台。用 BackgroundTasks 而不是 create_task：
    # 前者由框架管理生命周期，响应发出后才开始执行，异常不会静默丢失
    background_tasks.add_task(
        run_ingest_task,
        task_id=task.task_id,
        tmp_path=tmp_path,
        source_file=source_file,
        claim_key=source_file,
        claim_token=claim_token,
        resources=resources,
        store=store,
    )

    return DocumentUploadResponse(
        task_id=task.task_id,
        status=task.status.value,
        stage=task.stage.value,
        source_file=source_file,
        created_at=task.created_at,
        # 用请求路径拼，不写死前缀：改了 RAG_API_V1_PREFIX 这里不会失效
        poll_url=f"{request.url.path.rstrip('/')}/{task.task_id}",
    )


@router.get(
    "/{task_id}",
    response_model=DocumentTaskResponse,
    summary="查询入库任务状态",
    description=(
        "任务记录只保留 60 分钟（进程内存），过期后返回 404。\n\n"
        "**已入库的数据不受影响**：查不到任务不等于没写进库，可用检索接口确认。\n\n"
        "本接口不计入每分钟请求额度，前端可 1 秒轮询一次。"
    ),
)
async def get_document_task(
    request: Request,
    task_id: str,
    admin_id: str = Depends(require_admin),
) -> DocumentTaskResponse:
    resources = get_resources(request)
    store = resources.ingest_task_store

    # 顺手清理过期记录：没有独立定时器，靠访问驱动
    store.sweep()

    task = store.get(task_id, admin_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "TASK_NOT_FOUND",
                "message": (
                    "任务不存在、不属于当前账号，或记录已过期（只保留 60 分钟）。"
                    "已入库的数据不受影响，可用检索接口确认"
                ),
            },
        )

    return DocumentTaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        stage=task.stage.value,
        failed_stage=task.failed_stage.value if task.failed_stage else None,
        source_file=task.source_file,
        document_type=task.document_type,
        chunks_submitted=task.chunks_submitted,
        chunks_inserted=task.chunks_inserted,
        error_code=task.error_code,
        message=task.message,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.delete(
    "/{source_file}",
    response_model=DocumentDeleteResponse,
    summary="删除某文件的全部切块",
    description=(
        "按文件名删除数据库里的所有切块。磁盘上的原文件不受影响"
        "（上传件在任务结束后已删除；data/documents 是 CLI 的源目录，不归这里管）。\n\n"
        "历史会话里的引文不会消失：assistant 消息存的是当时的检索快照，"
        "这是刻意设计，历史记录不该被改写。"
    ),
)
async def delete_document(
    request: Request,
    source_file: str,
    admin_id: str = Depends(require_admin),
) -> DocumentDeleteResponse:
    resources = get_resources(request)
    store = resources.ingest_task_store

    try:
        name = sanitize_filename(source_file)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_FILENAME", "message": f"文件名不可用：{exc}"},
        ) from exc

    # 与上传抢同一个占用位：文件正在入库时删，会「删完又回来」
    claim_token = store.claim(name)
    if claim_token is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "DOCUMENT_PROCESSING",
                "message": "该文件正在入库，请等任务结束后再删除",
            },
        )

    try:
        deleted = await resources.vector_store.delete_by_source_file(name)
    finally:
        store.release(name, claim_token)

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "DOCUMENT_NOT_FOUND",
                "message": f"知识库中没有名为「{name}」的文档",
            },
        )

    logger.info("删除文档 user=%s file=%s chunks=%s", admin_id, name, deleted)
    return DocumentDeleteResponse(source_file=name, deleted_chunks=deleted)
