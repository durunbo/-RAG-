import os
from typing import List
from src.models import DocumentBlock


def load_document(file_path: str) -> List[DocumentBlock]:
    """
    文档解析统一入口
    根据文件扩展名自动分发到对应的格式解析器，输出统一的 DocumentBlock 结构
    新增格式时只需新增 elif 分支，上层调用无需修改
    :param file_path: 待解析文件的完整路径
    :return: 解析后的 DocumentBlock 列表
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Word 文档解析
    if ext in ['.docx', '.doc']:
        from .word_loader import load_word
        return load_word(file_path)

    # PDF 文档解析
    elif ext == '.pdf':
        from .pdf_loader import load_pdf
        return load_pdf(file_path)

    # Excel 表格解析
    elif ext in ['.xlsx', '.xls']:
        from .excel_loader import load_excel
        return load_excel(file_path)

    # 不支持的格式抛出异常
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


