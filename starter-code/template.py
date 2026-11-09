"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng của hệ sinh thái Vingroup (VinFast, Vinpearl).
- Giọng nói: Chuyên nghiệp, lịch sự, thân thiện, chính xác và tận tâm.

## 2. AVAILABLE TOOLS
- search_product_catalog(category: str, max_price: int): Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục ('xe_dien', 'du_lich') và giá tối đa.
- submit_support_ticket(customer_name: str, issue_description: str, priority: str): Ghi nhận phản ánh, khiếu nại hoặc yêu cầu hỗ trợ kỹ thuật vào hệ thống ticket.

## 3. CORE RULES
- TUYỆT ĐỐI KHÔNG tự bịa đặt thông tin sản phẩm hoặc chính sách (Zero Hallucination).
- BẮT BUỘC gọi tool khi người dùng cần tra cứu danh mục hoặc tạo yêu cầu hỗ trợ.
- Nếu không tìm thấy kết quả phù hợp, phản hồi thông báo lịch sự cho khách hàng.

## 4. OPERATIONAL BOUNDARIES
- Chỉ trả lời và hỗ trợ các sản phẩm, dịch vụ trong hệ sinh thái Vingroup (VinFast, Vinpearl).
- Từ chối lịch sự các chủ đề không liên quan đến Vingroup.

## 5. OUTPUT CONTRACT
- Tuân thủ cấu trúc ReAct:
  * Thought: Phân tích nhu cầu của người dùng.
  * Action: Tên tool và tham số gọi (nếu có).
  * Observation: Dữ liệu nhận về từ tool.
  * Final Answer: Phản hồi hoàn chỉnh, thân thiện tới khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Phân tích câu hỏi người dùng để xác định intent và tham số cho tool."""
        lower_input = user_input.lower()

        # 1. FAQ Intent
        is_faq = False
        if any(kw in lower_input for kw in ["bảo hành", "chính sách"]) and not any(kw in lower_input for kw in ["lỗi", "hỏng", "khiếu nại", "gấp"]):
            is_faq = True

        # 2. Ticket Intent
        needs_ticket = False
        if any(kw in lower_input for kw in ["lỗi", "sự cố", "khiếu nại", "phản hồi", "hỗ trợ", "bị ẩm mốc", "gấp", "nghiêm trọng", "ticket", "vấn đề"]):
            needs_ticket = True

        # 3. Catalog Intent
        needs_catalog = False
        if any(kw in lower_input for kw in ["xem", "tìm", "có xe", "có resort", "giá", "mua", "danh mục", "catalog", "dưới"]):
            if any(kw in lower_input for kw in ["xe điện", "xe dien", "vinfast", "resort", "vinpearl", "du lịch", "du lich", "triệu", "tỷ"]):
                needs_catalog = True

        # Trích xuất tham số catalog
        catalog_args = {}
        if needs_catalog:
            category = "xe_dien"
            if any(kw in lower_input for kw in ["du lịch", "du lich", "resort", "vinpearl", "phòng"]):
                category = "du_lich"
            
            max_price = 999999999999
            price_match = re.search(r"(?:dưới|tầm|khoảng|mức giá)\s*(\d+(?:[.,]\d+)?)\s*(triệu|trieu|tr|tỷ|ty)?", lower_input)
            if price_match:
                val = float(price_match.group(1).replace(",", "."))
                unit = (price_match.group(2) or "").lower()
                if "tỷ" in unit or "ty" in unit:
                    max_price = int(val * 1_000_000_000)
                else:
                    max_price = int(val * 1_000_000)
            catalog_args = {"category": category, "max_price": max_price}

        # Trích xuất tham số ticket
        ticket_args = {}
        if needs_ticket:
            customer_name = "Khách hàng"
            name_match = re.search(r"(?:tôi tên là|tôi tên|tên tôi là)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|$|\.|\bvà\b)", user_input, re.IGNORECASE)
            if name_match:
                customer_name = name_match.group(1).strip()

            priority = "medium"
            if any(kw in lower_input for kw in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
                priority = "high"
            elif any(kw in lower_input for kw in ["thấp", "low"]):
                priority = "low"

            issue_desc = user_input
            issue_match = re.search(r"(?:xe [^,.]+ bị [^,.]+|phòng [^,.]+ bị [^,.]+|bị lỗi [^,.]+)", user_input, re.IGNORECASE)
            if issue_match:
                issue_desc = issue_match.group(0).strip()

            ticket_args = {
                "customer_name": customer_name,
                "issue_description": issue_desc,
                "priority": priority
            }

        return {
            "is_faq": is_faq,
            "needs_catalog": needs_catalog,
            "catalog_args": catalog_args,
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)
        iteration = 1

        # Case 1: FAQ không cần gọi tool
        if intents["is_faq"]:
            thought = "Người dùng hỏi về chính sách bảo hành. Dùng tri thức có sẵn để trả lời, không cần gọi tool."
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km tuỳ điều kiện nào đến trước."
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": None,
                "observation": None,
                "answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Danh sách các action cần thực hiện
        actions_to_execute = []
        if intents["needs_catalog"]:
            actions_to_execute.append(("search_product_catalog", intents["catalog_args"]))
        if intents["needs_ticket"]:
            actions_to_execute.append(("submit_support_ticket", intents["ticket_args"]))

        # Nếu không nhận diện được tool nào phù hợp
        if not actions_to_execute:
            return {
                "answer": f"VinAssistant đã ghi nhận câu hỏi của bạn: {user_input}",
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        observations = []
        answers_parts = []

        # Chạy từng iteration
        for tool_name, tool_kwargs in actions_to_execute:
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": self.max_iterations,
                    "status": "max_iterations_reached"
                }

            tool_fn = TOOL_MAP.get(tool_name)
            observation = tool_fn(**tool_kwargs) if tool_fn else None

            # Xử lý kết quả quan sát
            if tool_name == "search_product_catalog":
                if not observation or len(observation) == 0:
                    answers_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.")
                else:
                    product_list_str = ", ".join([f"{p['name']} ({p['price_vnd']:,} VNĐ)" for p in observation])
                    answers_parts.append(f"Dưới đây là các sản phẩm phù hợp: {product_list_str}.")

            elif tool_name == "submit_support_ticket":
                t_id = observation.get("ticket_id", "")
                c_name = observation.get("customer_name", "")
                answers_parts.append(f"Yêu cầu hỗ trợ của khách hàng {c_name} đã được ghi nhận thành công với mã ticket: {t_id}.")

            self.trace.append({
                "iteration": iteration,
                "thought": f"Cần thực thi tool {tool_name} với tham số {tool_kwargs}.",
                "action": {"tool": tool_name, "parameters": tool_kwargs},
                "observation": observation
            })

            # Tăng iteration nếu còn action tiếp theo
            if len(actions_to_execute) > 1 and iteration < len(actions_to_execute):
                iteration += 1

        final_answer = "\n".join(answers_parts)

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
