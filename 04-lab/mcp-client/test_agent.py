"""Kiểm tra end-to-end: DeepSeek (qua ADK) khám phá & gọi tool trên MCP server.

Chạy MCP server trước (PORT=8085), rồi:
    uv run python test_agent.py
"""
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from weather_agent import root_agent

APP = "weather_app"
USER = "user1"
SESSION = "session1"


async def ask(runner: Runner, prompt: str) -> None:
    print(f"\n{'='*60}\nUser: {prompt}\n{'='*60}")
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(
        user_id=USER, session_id=SESSION, new_message=content
    ):
        # In ra tool calls và tool results để thấy vòng lặp function calling
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    print(f"  [DeepSeek gọi tool] {part.function_call.name}({dict(part.function_call.args)})")
                if part.function_response:
                    resp = str(part.function_response.response)
                    print(f"  [MCP trả kết quả] {resp[:200]}")
        if event.is_final_response() and event.content:
            print(f"\nAgent: {event.content.parts[0].text}")


async def main() -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP, user_id=USER, session_id=SESSION
    )
    runner = Runner(
        agent=root_agent, app_name=APP, session_service=session_service
    )

    # health_check không cần WEATHERAPI_KEY -> chứng minh trọn vòng MCP
    await ask(runner, "Kiểm tra xem MCP weather server có đang chạy không?")
    # get_current_weather -> cần WEATHERAPI_KEY (nếu chưa set sẽ báo chưa cấu hình)
    await ask(runner, "Thời tiết Hà Nội bây giờ thế nào?")


if __name__ == "__main__":
    asyncio.run(main())
