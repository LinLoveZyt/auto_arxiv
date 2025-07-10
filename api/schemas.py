# api/schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict
from datetime import datetime, date

class CustomFetchRequest(BaseModel):
    """自定义获取论文接口 (/fetch/custom) 的请求体。"""
    domains: List[str] = Field(
        ...,
        min_length=1,
        description="要检索的arXiv领域列表。",
        examples=[["cs.AI", "cs.LG"]]
    )
    start_date: date = Field(..., description="搜索范围的开始日期 (YYYY-MM-DD)。")
    end_date: date = Field(..., description="搜索范围的结束日期 (YYYY-MM-DD)。")
    max_results: int = Field(
        default=50,
        gt=0,
        le=200,
        description="希望获取的最大结果数量。"
    )
    
    @field_validator('end_date')
    @classmethod
    def end_date_must_be_after_start_date(cls, v: date, values):
        if 'start_date' in values.data and v < values.data['start_date']:
            raise ValueError('结束日期不能早于开始日期')
        return v
        


class QueryRequest(BaseModel):
    """用户问答接口 (/query) 的请求体。"""
    query_text: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="用户的自然语言问题。",
        examples=["介绍一下Transformer模型及其在目标检测领域的应用"]
    )
    online_search_enabled: bool = Field(
        default=False, 
        description="是否启用在线搜索功能。"
    )
    user_id: Optional[str] = Field(
        None,
        description="用于追踪的可选用户ID。",
        examples=["user-12345"]
    )

# ==============================================================================
#  响应体模型 (Response Body Models)
# ==============================================================================


class PaperSource(BaseModel):
    arxiv_id: str = Field(description="论文的arXiv ID。")
    title: str = Field(description="论文标题。")
    authors: List[str] = Field(description="作者列表。")
    summary: str = Field(description="论文摘要。")
    pdf_url: str = Field(description="用于在UI中查看或下载PDF的内部API链接。")

class GeneralStatusResponse(BaseModel):
    message: str = Field(description="描述操作结果的信息性消息。")

# [v3.0 修改] QueryResponse现在返回结构化的来源列表
class QueryResponse(GeneralStatusResponse):
    """用户问答接口 (/query) 的成功响应体。"""
    answer: str = Field(description="由LLM生成的、综合性的自然语言回答。")
    sources: List[PaperSource] = Field(description="构成答案依据的来源论文列表。")

class DailyRunResponse(GeneralStatusResponse):
    papers_processed: int = Field(description="本次运行成功处理并入库的论文数量。")
    report_path: Optional[str] = Field(None, description="生成的每日报告的路径（如果已实现）。")
    
class CustomFetchResponse(GeneralStatusResponse):
    papers_found: int = Field(description="根据条件找到的论文数量。")
    papers: List[Dict[str, Any]] = Field(description="找到的论文信息列表。")

class DailyRunRequest(BaseModel):
    """每日工作流接口 (/run/daily_workflow) 的请求体。"""
    research_plan: Optional[str] = Field(
        None,
        description="用户本次临时的调研计划，用于辅助AI筛选论文。",
        examples=["我最近在关注多模态大模型在自动驾驶领域的应用..."]
    )
    start_date: Optional[date] = Field(None, description="搜索范围的开始日期 (YYYY-MM-DD)。")
    end_date: Optional[date] = Field(None, description="搜索范围的结束日期 (YYYY-MM-DD)。")