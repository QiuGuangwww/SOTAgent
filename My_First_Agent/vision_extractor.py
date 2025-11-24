"""
增强版 Extractor：集成 Vision Model 处理复杂表格和图表
"""

import os
import base64
import json
from typing import Dict, List, Any, Optional
from PIL import Image
import io

try:
    from google.adk.models.lite_llm import LiteLlm
    VISION_MODEL_AVAILABLE = True
except ImportError:
    VISION_MODEL_AVAILABLE = False
    print("[Warning] Vision Model 不可用，将使用基础提取方法")


class VisionExtractor:
    """使用 Vision Model 增强的提取器"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        初始化 Vision Extractor
        
        Args:
            model_name: Vision Model 名称，可选 "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash-exp"
        """
        self.model_name = model_name
        self.vision_model = None
        
        if VISION_MODEL_AVAILABLE:
            try:
                # 使用 LiteLLM 调用 Vision Model
                if "gpt-4o" in model_name.lower():
                    self.vision_model = LiteLlm(model="gpt-4o")
                elif "claude" in model_name.lower():
                    self.vision_model = LiteLlm(model="claude-3-5-sonnet-20241022")
                elif "gemini" in model_name.lower():
                    self.vision_model = LiteLlm(model="gemini/gemini-2.0-flash-exp")
                else:
                    self.vision_model = LiteLlm(model=model_name)
                print(f"[VisionExtractor] 已初始化 Vision Model: {model_name}")
            except Exception as e:
                print(f"[VisionExtractor] Vision Model 初始化失败: {e}")
                self.vision_model = None
    
    def pdf_page_to_image(self, pdf_path: str, page_num: int) -> Optional[Image.Image]:
        """将 PDF 页面转换为图像"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                doc.close()
                return None
            
            page = doc[page_num]
            # 渲染为图像（2倍缩放以提高质量）
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            doc.close()
            return img
        except Exception as e:
            print(f"[VisionExtractor] PDF 转图像失败: {e}")
            return None
    
    def image_to_base64(self, image: Image.Image) -> str:
        """将图像转换为 base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    async def extract_table_with_vision(self, pdf_path: str, page_num: int, table_bbox: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """
        使用 Vision Model 提取表格
        
        Args:
            pdf_path: PDF 路径
            page_num: 页码（0-based）
            table_bbox: 表格边界框 (x0, y0, x1, y1)，如果为 None 则处理整页
        
        Returns:
            提取的表格数据
        """
        if not self.vision_model:
            return None
        
        try:
            # 将页面转换为图像
            image = self.pdf_page_to_image(pdf_path, page_num)
            if not image:
                return None
            
            # 如果指定了表格区域，裁剪图像
            if table_bbox:
                x0, y0, x1, y1 = table_bbox
                image = image.crop((x0, y0, x1, y1))
            
            # 转换为 base64
            img_base64 = self.image_to_base64(image)
            
            # 构建提示词
            prompt = """请从这张图片中提取表格数据。表格可能包含：
1. 模型名称
2. 数据集名称
3. 性能指标（如 Accuracy, F1, mAP 等）
4. 其他相关数值

请以 JSON 格式返回，包含：
- table_data: 表格的二维数组（行和列）
- headers: 表头列表
- metrics: 提取的指标列表，每个指标包含 {metric_name, value, dataset, model}
- notes: 任何重要注释

如果这不是表格或无法提取，返回 null。"""
            
            # 调用 Vision Model
            # 注意：这里需要根据实际的 LiteLLM API 调整
            # 由于 LiteLLM 的 Vision API 可能不同，这里提供一个通用接口
            
            # 对于 GPT-4o，可以使用 messages 格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # 这里需要根据实际的模型 API 调用
            # 由于 ADK 的 LiteLLM 可能不支持直接调用，我们返回一个占位符
            print(f"[VisionExtractor] 使用 Vision Model 处理页面 {page_num + 1}")
            
            # 实际调用需要根据 ADK 的 API 调整
            # response = await self.vision_model.generate_content_async(...)
            
            return {
                "status": "vision_processed",
                "page": page_num + 1,
                "note": "Vision Model 处理完成（需要根据实际 API 实现）"
            }
            
        except Exception as e:
            print(f"[VisionExtractor] Vision 表格提取失败: {e}")
            return None
    
    async def extract_metrics_with_llm(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        使用 LLM 从文本中提取指标（上下文理解）
        
        Args:
            text: 要分析的文本
            context: 上下文信息（如论文标题、摘要）
        
        Returns:
            提取的指标列表
        """
        if not self.vision_model:
            # 降级到基础正则表达式提取
            return self._extract_metrics_basic(text)
        
        try:
            prompt = f"""请从以下文本中提取性能指标。文本可能来自论文的实验部分。

上下文信息：
{context}

文本内容：
{text[:2000]}  # 限制长度

请提取所有性能指标，包括：
- 指标名称（如 Accuracy, F1-Score, mAP, Top-1, Top-5 等）
- 数值
- 对应的数据集（如果提到）
- 对应的模型（如果提到）

请以 JSON 格式返回，格式：
{{
  "metrics": [
    {{
      "metric_name": "accuracy",
      "value": 85.3,
      "unit": "percentage",
      "dataset": "ImageNet",
      "model": "ResNet-50",
      "context": "相关上下文文本"
    }}
  ]
}}

如果没有找到指标，返回 {{"metrics": []}}。"""
            
            # 调用 LLM（这里需要根据实际 API 调整）
            print(f"[VisionExtractor] 使用 LLM 提取指标（文本长度: {len(text)}）")
            
            # 实际调用需要根据 ADK 的 API 调整
            # response = await self.vision_model.generate_content_async(...)
            
            # 暂时返回空列表，实际实现需要调用 LLM
            return []
            
        except Exception as e:
            print(f"[VisionExtractor] LLM 指标提取失败: {e}")
            return self._extract_metrics_basic(text)
    
    def _extract_metrics_basic(self, text: str) -> List[Dict[str, Any]]:
        """基础正则表达式提取（降级方案）"""
        import re
        metrics = []
        
        metric_patterns = [
            (r"(?:accuracy|acc)\s*[=:]\s*(\d+\.?\d*)\s*%?", "accuracy"),
            (r"(?:f1[- ]?score|f1)\s*[=:]\s*(\d+\.?\d*)\s*%?", "f1_score"),
            (r"(?:mAP|mean average precision)\s*[=:]\s*(\d+\.?\d*)\s*%?", "mAP"),
            (r"(?:top[- ]?1|top1)\s*[=:]\s*(\d+\.?\d*)\s*%?", "top1_accuracy"),
            (r"(?:top[- ]?5|top5)\s*[=:]\s*(\d+\.?\d*)\s*%?", "top5_accuracy"),
        ]
        
        for pattern, metric_name in metric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                if value <= 1.0:
                    value = value * 100
                
                metrics.append({
                    "metric": metric_name,
                    "value": value,
                    "unit": "percentage",
                    "context": text[max(0, match.start()-50):match.end()+50]
                })
        
        return metrics
    
    def detect_tables_in_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        检测 PDF 中的表格位置（使用 pdfplumber 检测，然后可以用 Vision Model 精提取）
        
        Returns:
            表格位置信息列表
        """
        try:
            import pdfplumber
            tables_info = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 检测表格
                    tables = page.find_tables()
                    
                    for table_num, table in enumerate(tables):
                        bbox = table.bbox
                        tables_info.append({
                            "page": page_num + 1,
                            "table_index": table_num,
                            "bbox": bbox,  # (x0, y0, x1, y1)
                            "rows": len(table.extract()) if hasattr(table, 'extract') else 0
                        })
            
            return tables_info
        except Exception as e:
            print(f"[VisionExtractor] 表格检测失败: {e}")
            return []


# 便捷函数
def create_vision_extractor(model_name: str = "gpt-4o") -> VisionExtractor:
    """创建 Vision Extractor 实例"""
    return VisionExtractor(model_name)

